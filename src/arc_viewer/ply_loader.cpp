// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ply_loader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace lfs::arc {
    namespace {

        constexpr std::uint64_t kMaxHeaderBytes = 1024 * 1024;
        constexpr std::size_t kMaxHeaderLines = 4096;
        constexpr std::size_t kMaxElements = 64;
        constexpr std::size_t kMaxPropertiesPerElement = 256;
        constexpr std::uint64_t kMaxVertices = 50'000'000;
        constexpr std::uint64_t kMaxElementRecords = 100'000'000;
        constexpr std::uint64_t kMaxTotalRecords = 100'000'000;
        constexpr std::uint64_t kMaxListEntries = 10'000'000;
        constexpr std::uint64_t kMaxFileBytes = 64ULL * 1024 * 1024 * 1024;
        constexpr double kShBasisC0 = 0.28209479177387814;
        constexpr std::size_t kRobustFramingMinimumVertices = 1'000;
        constexpr std::size_t kFramingTrimDivisor = 100;
        constexpr float kFramingPaddingFraction = 0.05F;

        enum class PlyFormat {
            Ascii,
            BinaryLittleEndian,
        };

        enum class ScalarType {
            Int8,
            UInt8,
            Int16,
            UInt16,
            Int32,
            UInt32,
            Float32,
            Float64,
        };

        struct Property {
            std::string name;
            bool isList = false;
            ScalarType scalarType = ScalarType::Float32;
            ScalarType countType = ScalarType::UInt8;
            ScalarType itemType = ScalarType::Float32;
        };

        struct Element {
            std::string name;
            std::uint64_t count = 0;
            std::vector<Property> properties;
        };

        struct Header {
            PlyFormat format = PlyFormat::Ascii;
            std::vector<Element> elements;
        };

        [[noreturn]] void fail(const std::string& message) {
            throw std::runtime_error("PLY parse error: " + message);
        }

        bool readBoundedLine(std::istream& input, std::string& line) {
            line.clear();
            line.reserve(256);
            char character = 0;
            while (input.get(character)) {
                if (character == '\n') {
                    return true;
                }
                if (line.size() >= kMaxHeaderBytes) {
                    fail("header line exceeds safety limit");
                }
                line.push_back(character);
            }
            if (input.bad()) {
                fail("I/O error while reading header");
            }
            return !line.empty();
        }

        void stripCarriageReturn(std::string& line) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
        }

        std::string lowerAscii(std::string value) {
            for (char& character : value) {
                if (character >= 'A' && character <= 'Z') {
                    character = static_cast<char>(character - 'A' + 'a');
                }
            }
            return value;
        }

        std::uint64_t parseUnsigned(std::string_view token, std::string_view what) {
            if (!token.empty() && token.front() == '+') {
                token.remove_prefix(1);
            }
            std::uint64_t value = 0;
            const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
            if (token.empty() || result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
                fail(std::string(what) + " is not a valid unsigned integer");
            }
            return value;
        }

        ScalarType parseScalarType(const std::string& rawName) {
            const std::string name = lowerAscii(rawName);
            if (name == "char" || name == "int8") {
                return ScalarType::Int8;
            }
            if (name == "uchar" || name == "uint8") {
                return ScalarType::UInt8;
            }
            if (name == "short" || name == "int16") {
                return ScalarType::Int16;
            }
            if (name == "ushort" || name == "uint16") {
                return ScalarType::UInt16;
            }
            if (name == "int" || name == "int32") {
                return ScalarType::Int32;
            }
            if (name == "uint" || name == "uint32") {
                return ScalarType::UInt32;
            }
            if (name == "float" || name == "float32") {
                return ScalarType::Float32;
            }
            if (name == "double" || name == "float64") {
                return ScalarType::Float64;
            }
            fail("unsupported scalar type '" + rawName + "'");
        }

        bool isInteger(ScalarType type) {
            return type != ScalarType::Float32 && type != ScalarType::Float64;
        }

        std::size_t scalarSize(ScalarType type) {
            switch (type) {
            case ScalarType::Int8:
            case ScalarType::UInt8:
                return 1;
            case ScalarType::Int16:
            case ScalarType::UInt16:
                return 2;
            case ScalarType::Int32:
            case ScalarType::UInt32:
            case ScalarType::Float32:
                return 4;
            case ScalarType::Float64:
                return 8;
            }
            fail("invalid scalar type");
        }

        std::uint64_t minimumPayloadBytes(const Header& header) {
            std::uint64_t total = 0;
            for (const Element& element : header.elements) {
                std::uint64_t recordBytes = 0;
                for (const Property& property : element.properties) {
                    const std::uint64_t propertyBytes = header.format == PlyFormat::Ascii
                                                            ? 1
                                                            : scalarSize(property.isList ? property.countType : property.scalarType);
                    if (recordBytes > std::numeric_limits<std::uint64_t>::max() - propertyBytes) {
                        fail("minimum record byte count overflows");
                    }
                    recordBytes += propertyBytes;
                }
                if (recordBytes != 0 &&
                    element.count > std::numeric_limits<std::uint64_t>::max() / recordBytes) {
                    fail("minimum payload byte count overflows");
                }
                const std::uint64_t elementBytes = element.count * recordBytes;
                if (total > std::numeric_limits<std::uint64_t>::max() - elementBytes) {
                    fail("minimum payload byte count overflows");
                }
                total += elementBytes;
            }
            return total;
        }

        Header readHeader(std::istream& input) {
            std::string line;
            if (!readBoundedLine(input, line)) {
                fail("file is empty");
            }
            stripCarriageReturn(line);
            if (line != "ply") {
                fail("missing 'ply' magic line");
            }

            Header header;
            bool formatSeen = false;
            bool endSeen = false;
            std::uint64_t headerBytes = line.size() + 1;
            std::size_t headerLines = 1;
            std::uint64_t totalRecords = 0;
            Element* currentElement = nullptr;

            while (readBoundedLine(input, line)) {
                ++headerLines;
                headerBytes += line.size() + 1;
                if (headerLines > kMaxHeaderLines || headerBytes > kMaxHeaderBytes) {
                    fail("header exceeds safety limit");
                }
                stripCarriageReturn(line);
                if (line.find('\0') != std::string::npos) {
                    fail("header contains a NUL byte");
                }

                std::istringstream words(line);
                std::string keyword;
                words >> keyword;
                keyword = lowerAscii(keyword);
                if (keyword.empty() || keyword == "comment" || keyword == "obj_info") {
                    continue;
                }
                if (keyword == "format") {
                    std::string formatName;
                    std::string version;
                    std::string extra;
                    if (formatSeen || !(words >> formatName >> version) || (words >> extra)) {
                        fail("malformed or duplicate format declaration");
                    }
                    formatName = lowerAscii(formatName);
                    if (version != "1.0") {
                        fail("only PLY format version 1.0 is supported");
                    }
                    if (formatName == "ascii") {
                        header.format = PlyFormat::Ascii;
                    } else if (formatName == "binary_little_endian") {
                        header.format = PlyFormat::BinaryLittleEndian;
                    } else if (formatName == "binary_big_endian") {
                        fail("binary_big_endian is not supported");
                    } else {
                        fail("unsupported format '" + formatName + "'");
                    }
                    formatSeen = true;
                    continue;
                }
                if (keyword == "element") {
                    if (!formatSeen) {
                        fail("element declaration appears before format");
                    }
                    std::string name;
                    std::string countToken;
                    std::string extra;
                    if (!(words >> name >> countToken) || (words >> extra)) {
                        fail("malformed element declaration");
                    }
                    if (header.elements.size() >= kMaxElements) {
                        fail("too many element declarations");
                    }
                    Element element;
                    element.name = lowerAscii(name);
                    element.count = parseUnsigned(countToken, "element count");
                    const std::uint64_t limit = element.name == "vertex" ? kMaxVertices : kMaxElementRecords;
                    if (element.count > limit) {
                        fail("element '" + element.name + "' exceeds the supported record limit");
                    }
                    if (element.count > kMaxTotalRecords - totalRecords) {
                        fail("aggregate element record count exceeds the safety limit");
                    }
                    totalRecords += element.count;
                    header.elements.push_back(std::move(element));
                    currentElement = &header.elements.back();
                    continue;
                }
                if (keyword == "property") {
                    if (!currentElement) {
                        fail("property declaration appears before an element");
                    }
                    if (currentElement->properties.size() >= kMaxPropertiesPerElement) {
                        fail("element '" + currentElement->name + "' has too many properties");
                    }

                    std::string first;
                    if (!(words >> first)) {
                        fail("malformed property declaration");
                    }
                    Property property;
                    if (lowerAscii(first) == "list") {
                        std::string countType;
                        std::string itemType;
                        std::string name;
                        std::string extra;
                        if (!(words >> countType >> itemType >> name) || (words >> extra)) {
                            fail("malformed list property declaration");
                        }
                        property.isList = true;
                        property.countType = parseScalarType(countType);
                        property.itemType = parseScalarType(itemType);
                        property.name = lowerAscii(name);
                        if (!isInteger(property.countType)) {
                            fail("list count type must be an integer");
                        }
                    } else {
                        std::string name;
                        std::string extra;
                        if (!(words >> name) || (words >> extra)) {
                            fail("malformed scalar property declaration");
                        }
                        property.scalarType = parseScalarType(first);
                        property.name = lowerAscii(name);
                    }

                    const auto duplicate = std::find_if(
                        currentElement->properties.begin(), currentElement->properties.end(),
                        [&](const Property& existing) { return existing.name == property.name; });
                    if (duplicate != currentElement->properties.end()) {
                        fail("duplicate property '" + property.name + "' in element '" +
                             currentElement->name + "'");
                    }
                    currentElement->properties.push_back(std::move(property));
                    continue;
                }
                if (keyword == "end_header") {
                    std::string extra;
                    if (words >> extra) {
                        fail("unexpected text after end_header");
                    }
                    endSeen = true;
                    break;
                }
                fail("unsupported header directive '" + keyword + "'");
            }

            if (!formatSeen) {
                fail("missing format declaration");
            }
            if (!endSeen) {
                fail("missing end_header");
            }
            for (const Element& element : header.elements) {
                if (element.count > 0 && element.properties.empty()) {
                    fail("non-empty element '" + element.name + "' has no properties");
                }
            }
            return header;
        }

        std::string readAsciiToken(std::istream& input) {
            std::string token;
            token.reserve(32);
            char character = 0;
            while (input.get(character)) {
                if (std::isspace(static_cast<unsigned char>(character)) != 0) {
                    if (!token.empty()) {
                        return token;
                    }
                    continue;
                }
                if (token.size() >= 128) {
                    fail("ASCII scalar token exceeds safety limit");
                }
                token.push_back(character);
            }
            if (input.bad()) {
                fail("I/O error while reading ASCII payload");
            }
            if (token.empty()) {
                fail("ASCII payload is truncated");
            }
            return token;
        }

        template <typename Value>
        Value parseIntegerToken(std::string_view token, Value minimum, Value maximum) {
            if (!token.empty() && token.front() == '+') {
                token.remove_prefix(1);
            }
            Value value{};
            const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
            if (token.empty() || result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
                value < minimum || value > maximum) {
                fail("ASCII integer is invalid or outside its declared type");
            }
            return value;
        }

        double readAsciiScalar(std::istream& input, ScalarType type) {
            const std::string storage = readAsciiToken(input);
            std::string_view token(storage);
            switch (type) {
            case ScalarType::Int8:
                return static_cast<double>(parseIntegerToken<std::int64_t>(token, -128, 127));
            case ScalarType::UInt8:
                return static_cast<double>(parseIntegerToken<std::uint64_t>(token, 0, 255));
            case ScalarType::Int16:
                return static_cast<double>(parseIntegerToken<std::int64_t>(token, -32768, 32767));
            case ScalarType::UInt16:
                return static_cast<double>(parseIntegerToken<std::uint64_t>(token, 0, 65535));
            case ScalarType::Int32:
                return static_cast<double>(parseIntegerToken<std::int64_t>(
                    token, std::numeric_limits<std::int32_t>::min(),
                    std::numeric_limits<std::int32_t>::max()));
            case ScalarType::UInt32:
                return static_cast<double>(parseIntegerToken<std::uint64_t>(
                    token, 0, std::numeric_limits<std::uint32_t>::max()));
            case ScalarType::Float32:
            case ScalarType::Float64:
                break;
            }

            if (!token.empty() && token.front() == '+') {
                token.remove_prefix(1);
            }
            double value = 0.0;
            const auto result = std::from_chars(
                token.data(), token.data() + token.size(), value, std::chars_format::general);
            if (token.empty() || result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
                fail("ASCII floating-point scalar is invalid");
            }
            if (type == ScalarType::Float32 && std::isfinite(value)) {
                if (std::abs(value) > std::numeric_limits<float>::max()) {
                    fail("ASCII float32 scalar is outside its declared type");
                }
                value = static_cast<float>(value);
            }
            return value;
        }

        std::uint64_t readLittleEndianBits(std::istream& input, std::size_t byteCount) {
            std::array<unsigned char, 8> bytes{};
            input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byteCount));
            if (input.gcount() != static_cast<std::streamsize>(byteCount)) {
                fail("binary payload is truncated");
            }
            std::uint64_t value = 0;
            for (std::size_t index = 0; index < byteCount; ++index) {
                value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
            }
            return value;
        }

        double readBinaryScalar(std::istream& input, ScalarType type) {
            const std::uint64_t bits = readLittleEndianBits(input, scalarSize(type));
            switch (type) {
            case ScalarType::Int8:
                return std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(bits));
            case ScalarType::UInt8:
                return static_cast<std::uint8_t>(bits);
            case ScalarType::Int16:
                return std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(bits));
            case ScalarType::UInt16:
                return static_cast<std::uint16_t>(bits);
            case ScalarType::Int32:
                return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
            case ScalarType::UInt32:
                return static_cast<std::uint32_t>(bits);
            case ScalarType::Float32:
                return std::bit_cast<float>(static_cast<std::uint32_t>(bits));
            case ScalarType::Float64:
                return std::bit_cast<double>(bits);
            }
            fail("invalid binary scalar type");
        }

        double readScalar(std::istream& input, PlyFormat format, ScalarType type) {
            return format == PlyFormat::Ascii ? readAsciiScalar(input, type)
                                              : readBinaryScalar(input, type);
        }

        std::uint64_t readListCount(std::istream& input, PlyFormat format, ScalarType type) {
            const double value = readScalar(input, format, type);
            if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value ||
                value > static_cast<double>(kMaxListEntries)) {
                fail("list count is invalid or exceeds the safety limit");
            }
            return static_cast<std::uint64_t>(value);
        }

        std::optional<std::size_t> findProperty(
            const Element& element, std::initializer_list<std::string_view> names) {
            for (const std::string_view name : names) {
                for (std::size_t index = 0; index < element.properties.size(); ++index) {
                    if (element.properties[index].name == name) {
                        return index;
                    }
                }
            }
            return std::nullopt;
        }

        float normalizedColor(double value, ScalarType type) {
            if (!std::isfinite(value)) {
                fail("vertex color contains a non-finite value");
            }
            double maximum = 1.0;
            switch (type) {
            case ScalarType::Int8:
                maximum = 127.0;
                break;
            case ScalarType::UInt8:
                maximum = 255.0;
                break;
            case ScalarType::Int16:
                maximum = 32767.0;
                break;
            case ScalarType::UInt16:
                maximum = 65535.0;
                break;
            case ScalarType::Int32:
                maximum = static_cast<double>(std::numeric_limits<std::int32_t>::max());
                break;
            case ScalarType::UInt32:
                maximum = static_cast<double>(std::numeric_limits<std::uint32_t>::max());
                break;
            case ScalarType::Float32:
            case ScalarType::Float64:
                maximum = 1.0;
                break;
            }
            if (value < 0.0 || value > maximum) {
                fail("vertex color is outside its declared numeric range");
            }
            return static_cast<float>(value / maximum);
        }

        float sphericalHarmonicDcColor(double value) {
            if (!std::isfinite(value)) {
                fail("vertex spherical harmonic color contains a non-finite value");
            }
            return static_cast<float>(std::clamp(0.5 + kShBasisC0 * value, 0.0, 1.0));
        }

        void skipBinaryBytes(
            std::istream& input, std::uint64_t byteCount, std::streampos payloadEnd) {
            if (byteCount > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                fail("binary skip size exceeds stream limits");
            }
            const std::streampos current = input.tellg();
            if (current < 0 || payloadEnd < current ||
                static_cast<std::uint64_t>(payloadEnd - current) < byteCount) {
                fail("binary payload is truncated");
            }
            input.seekg(static_cast<std::streamoff>(byteCount), std::ios::cur);
            if (!input) {
                fail("could not seek across binary payload");
            }
        }

        void skipProperty(
            std::istream& input,
            PlyFormat format,
            const Property& property,
            std::streampos payloadEnd) {
            if (!property.isList) {
                (void)readScalar(input, format, property.scalarType);
                return;
            }
            const std::uint64_t count = readListCount(input, format, property.countType);
            if (format == PlyFormat::BinaryLittleEndian) {
                const std::size_t itemBytes = scalarSize(property.itemType);
                if (count > std::numeric_limits<std::uint64_t>::max() / itemBytes) {
                    fail("binary list byte size overflows");
                }
                skipBinaryBytes(input, count * itemBytes, payloadEnd);
                return;
            }
            for (std::uint64_t index = 0; index < count; ++index) {
                (void)readScalar(input, format, property.itemType);
            }
        }

        void computeFramingBounds(PointCloud& cloud) {
            cloud.framingBoundsMin = cloud.boundsMin;
            cloud.framingBoundsMax = cloud.boundsMax;
            if (cloud.vertices.size() < kRobustFramingMinimumVertices) {
                return;
            }

            const std::size_t trimCount = cloud.vertices.size() / kFramingTrimDivisor;
            const std::size_t upperIndex = cloud.vertices.size() - trimCount - 1;
            std::vector<float> values;
            values.reserve(cloud.vertices.size());
            for (std::size_t axis = 0; axis < 3; ++axis) {
                values.clear();
                for (const Vertex& vertex : cloud.vertices) {
                    const std::array<float, 3> position{vertex.x, vertex.y, vertex.z};
                    values.push_back(position[axis]);
                }

                std::nth_element(values.begin(), values.begin() + trimCount, values.end());
                const double lower = values[trimCount];
                std::nth_element(values.begin(), values.begin() + upperIndex, values.end());
                const double upper = values[upperIndex];
                const double padding = (upper - lower) * kFramingPaddingFraction;
                const double floatMaximum = std::numeric_limits<float>::max();
                cloud.framingBoundsMin[axis] = static_cast<float>(
                    std::clamp(lower - padding, -floatMaximum, floatMaximum));
                cloud.framingBoundsMax[axis] = static_cast<float>(
                    std::clamp(upper + padding, -floatMaximum, floatMaximum));
            }
        }

    } // namespace

    PointCloud loadPly(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Could not open PLY file: " + path.string());
        }

        const Header header = readHeader(input);
        const std::streampos payloadStart = input.tellg();
        input.seekg(0, std::ios::end);
        const std::streampos payloadEnd = input.tellg();
        if (payloadStart < 0 || payloadEnd < payloadStart ||
            static_cast<std::uint64_t>(payloadEnd) > kMaxFileBytes) {
            fail("file size is invalid or exceeds the safety limit");
        }
        const std::uint64_t availablePayloadBytes =
            static_cast<std::uint64_t>(payloadEnd - payloadStart);
        if (availablePayloadBytes < minimumPayloadBytes(header)) {
            fail("payload is truncated for the declared element counts");
        }
        input.seekg(payloadStart);
        if (!input) {
            fail("could not seek to PLY payload");
        }
        const Element* vertexElement = nullptr;
        for (const Element& element : header.elements) {
            if (element.name == "vertex") {
                if (vertexElement) {
                    fail("multiple vertex elements are not supported");
                }
                vertexElement = &element;
            }
        }
        if (!vertexElement) {
            fail("missing vertex element");
        }
        if (vertexElement->count == 0) {
            fail("vertex element is empty");
        }

        const auto xIndex = findProperty(*vertexElement, {"x"});
        const auto yIndex = findProperty(*vertexElement, {"y"});
        const auto zIndex = findProperty(*vertexElement, {"z"});
        if (!xIndex || !yIndex || !zIndex) {
            fail("vertex element must declare scalar x, y, and z properties");
        }
        for (const std::size_t index : {*xIndex, *yIndex, *zIndex}) {
            if (vertexElement->properties[index].isList) {
                fail("vertex x, y, and z properties must be scalar");
            }
        }

        enum class ColorEncoding {
            Default,
            Rgb,
            SphericalHarmonicDc,
        };

        std::array<std::optional<std::size_t>, 3> colorIndices{
            findProperty(*vertexElement, {"red", "r", "diffuse_red"}),
            findProperty(*vertexElement, {"green", "g", "diffuse_green"}),
            findProperty(*vertexElement, {"blue", "b", "diffuse_blue"})};
        ColorEncoding colorEncoding = ColorEncoding::Default;
        if (colorIndices[0] && colorIndices[1] && colorIndices[2]) {
            colorEncoding = ColorEncoding::Rgb;
        } else {
            colorIndices = {
                findProperty(*vertexElement, {"f_dc_0"}),
                findProperty(*vertexElement, {"f_dc_1"}),
                findProperty(*vertexElement, {"f_dc_2"})};
            if (colorIndices[0] && colorIndices[1] && colorIndices[2]) {
                colorEncoding = ColorEncoding::SphericalHarmonicDc;
            }
        }
        if (colorEncoding != ColorEncoding::Default) {
            for (const auto& index : colorIndices) {
                if (!index) {
                    fail("internal vertex color property selection is incomplete");
                }
                if (vertexElement->properties[*index].isList) {
                    fail("vertex color properties must be scalar");
                }
            }
        }

        PointCloud cloud;
        try {
            cloud.vertices.reserve(static_cast<std::size_t>(vertexElement->count));
        } catch (const std::bad_alloc&) {
            throw std::runtime_error("Not enough memory for PLY vertex data: " + path.string());
        }

        std::array<double, 3> minimum{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
        std::array<double, 3> maximum{
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};

        for (const Element& element : header.elements) {
            const bool isVertex = &element == vertexElement;
            for (std::uint64_t record = 0; record < element.count; ++record) {
                if (!isVertex) {
                    for (const Property& property : element.properties) {
                        skipProperty(input, header.format, property, payloadEnd);
                    }
                    continue;
                }

                std::array<double, 3> position{};
                std::array<double, 3> color{};
                for (std::size_t propertyIndex = 0;
                     propertyIndex < element.properties.size(); ++propertyIndex) {
                    const Property& property = element.properties[propertyIndex];
                    if (property.isList) {
                        skipProperty(input, header.format, property, payloadEnd);
                        continue;
                    }
                    const double value = readScalar(input, header.format, property.scalarType);
                    if (propertyIndex == *xIndex) {
                        position[0] = value;
                    } else if (propertyIndex == *yIndex) {
                        position[1] = value;
                    } else if (propertyIndex == *zIndex) {
                        position[2] = value;
                    } else if (colorEncoding != ColorEncoding::Default &&
                               propertyIndex == *colorIndices[0]) {
                        color[0] = value;
                    } else if (colorEncoding != ColorEncoding::Default &&
                               propertyIndex == *colorIndices[1]) {
                        color[1] = value;
                    } else if (colorEncoding != ColorEncoding::Default &&
                               propertyIndex == *colorIndices[2]) {
                        color[2] = value;
                    }
                }

                for (std::size_t axis = 0; axis < position.size(); ++axis) {
                    if (!std::isfinite(position[axis]) ||
                        std::abs(position[axis]) > std::numeric_limits<float>::max()) {
                        fail("vertex coordinate is non-finite or outside float32 range");
                    }
                    minimum[axis] = std::min(minimum[axis], position[axis]);
                    maximum[axis] = std::max(maximum[axis], position[axis]);
                }

                Vertex vertex{
                    static_cast<float>(position[0]),
                    static_cast<float>(position[1]),
                    static_cast<float>(position[2]),
                    0.82F,
                    0.87F,
                    0.95F};
                if (colorEncoding == ColorEncoding::Rgb) {
                    vertex.red = normalizedColor(
                        color[0], element.properties[*colorIndices[0]].scalarType);
                    vertex.green = normalizedColor(
                        color[1], element.properties[*colorIndices[1]].scalarType);
                    vertex.blue = normalizedColor(
                        color[2], element.properties[*colorIndices[2]].scalarType);
                } else if (colorEncoding == ColorEncoding::SphericalHarmonicDc) {
                    vertex.red = sphericalHarmonicDcColor(color[0]);
                    vertex.green = sphericalHarmonicDcColor(color[1]);
                    vertex.blue = sphericalHarmonicDcColor(color[2]);
                }
                cloud.vertices.push_back(vertex);
            }
        }

        if (input.bad()) {
            fail("I/O error while reading payload");
        }
        for (std::size_t axis = 0; axis < minimum.size(); ++axis) {
            cloud.boundsMin[axis] = static_cast<float>(minimum[axis]);
            cloud.boundsMax[axis] = static_cast<float>(maximum[axis]);
        }
        computeFramingBounds(cloud);
        return cloud;
    }

} // namespace lfs::arc
