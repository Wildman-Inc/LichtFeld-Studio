// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ply_loader.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

    constexpr float kShBasisC0 = 0.28209479177387814F;

    class TemporaryPly {
    public:
        explicit TemporaryPly(std::string_view label) {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            path_ = std::filesystem::temp_directory_path() /
                    ("lfs_arc_" + std::string(label) + "_" + std::to_string(nonce) + ".ply");
        }

        TemporaryPly(const TemporaryPly&) = delete;
        TemporaryPly& operator=(const TemporaryPly&) = delete;

        ~TemporaryPly() {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }

        const std::filesystem::path& path() const { return path_; }

    private:
        std::filesystem::path path_;
    };

    void expect(bool condition, std::string_view message) {
        if (!condition) {
            throw std::runtime_error(std::string(message));
        }
    }

    void expectNear(float actual, float expected, std::string_view message) {
        if (std::abs(actual - expected) > 1.0e-5F) {
            throw std::runtime_error(
                std::string(message) + ": expected " + std::to_string(expected) +
                ", got " + std::to_string(actual));
        }
    }

    void writeAscii(const std::filesystem::path& path, std::string_view payload) {
        std::ofstream output(path, std::ios::binary);
        expect(static_cast<bool>(output), "could not create ASCII fixture");
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        expect(static_cast<bool>(output), "could not write ASCII fixture");
    }

    void writeLittleEndian32(std::ostream& output, std::uint32_t value) {
        for (unsigned int byte = 0; byte < 4; ++byte) {
            output.put(static_cast<char>((value >> (byte * 8)) & 0xFFU));
        }
    }

    void writeFloat32(std::ostream& output, float value) {
        writeLittleEndian32(output, std::bit_cast<std::uint32_t>(value));
    }

    void testAsciiRgbAndElementSkipping() {
        TemporaryPly file("ascii_rgb");
        writeAscii(file.path(),
                   "ply\r\n"
                   "format ascii 1.0\r\n"
                   "element face 1\r\n"
                   "property list uchar int vertex_indices\r\n"
                   "element vertex 2\r\n"
                   "property float z\r\n"
                   "property uchar blue\r\n"
                   "property float x\r\n"
                   "property uchar red\r\n"
                   "property float y\r\n"
                   "property uchar green\r\n"
                   "end_header\r\n"
                   "3 0 1 1\n"
                   "3 30 1 10 2 20\n"
                   "6 255 -2 128 4 0\n");

        const lfs::arc::PointCloud cloud = lfs::arc::loadPly(file.path());
        expect(cloud.vertices.size() == 2, "ASCII fixture vertex count mismatch");
        const auto& first = cloud.vertices[0];
        expectNear(first.x, 1.0F, "first x");
        expectNear(first.y, 2.0F, "first y");
        expectNear(first.z, 3.0F, "first z");
        expectNear(first.red, 10.0F / 255.0F, "first red");
        expectNear(first.green, 20.0F / 255.0F, "first green");
        expectNear(first.blue, 30.0F / 255.0F, "first blue");

        const auto& second = cloud.vertices[1];
        expectNear(second.red, 128.0F / 255.0F, "second red");
        expectNear(second.green, 0.0F, "second green");
        expectNear(second.blue, 1.0F, "second blue");
        expectNear(cloud.boundsMin[0], -2.0F, "minimum x");
        expectNear(cloud.boundsMax[2], 6.0F, "maximum z");
    }

    void testBinaryGaussianSphericalHarmonicColor() {
        TemporaryPly file("binary_gaussian");
        std::ofstream output(file.path(), std::ios::binary);
        expect(static_cast<bool>(output), "could not create binary fixture");
        output << "ply\n"
                  "format binary_little_endian 1.0\n"
                  "element vertex 2\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "property float f_dc_0\n"
                  "property float f_dc_1\n"
                  "property float f_dc_2\n"
                  "element face 1\n"
                  "property list uchar int vertex_indices\n"
                  "end_header\n";
        const std::array<std::array<float, 6>, 2> vertices{{
            {1.0F, 2.0F, 3.0F, 0.5F, -0.5F, 2.0F},
            {-1.0F, -2.0F, -3.0F, -2.0F, 0.0F, 0.25F},
        }};
        for (const auto& vertex : vertices) {
            for (const float value : vertex) {
                writeFloat32(output, value);
            }
        }
        output.put(3);
        writeLittleEndian32(output, 0);
        writeLittleEndian32(output, 1);
        writeLittleEndian32(output, 1);
        output.close();
        expect(static_cast<bool>(output), "could not write binary fixture");

        const lfs::arc::PointCloud cloud = lfs::arc::loadPly(file.path());
        expect(cloud.vertices.size() == 2, "binary fixture vertex count mismatch");
        expectNear(cloud.vertices[0].red, 0.5F + kShBasisC0 * 0.5F, "SH red");
        expectNear(cloud.vertices[0].green, 0.5F - kShBasisC0 * 0.5F, "SH green");
        expectNear(cloud.vertices[0].blue, 1.0F, "SH blue clamp");
        expectNear(cloud.vertices[1].red, 0.0F, "SH red lower clamp");
        expectNear(cloud.vertices[1].green, 0.5F, "SH neutral green");
        expectNear(
            cloud.vertices[1].blue, 0.5F + kShBasisC0 * 0.25F, "SH second blue");
    }

    void testTruncatedBinaryIsRejected() {
        TemporaryPly file("truncated");
        std::ofstream output(file.path(), std::ios::binary);
        expect(static_cast<bool>(output), "could not create truncated fixture");
        output << "ply\n"
                  "format binary_little_endian 1.0\n"
                  "element vertex 1\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "end_header\n";
        writeFloat32(output, 1.0F);
        writeFloat32(output, 2.0F);
        output.close();

        try {
            (void)lfs::arc::loadPly(file.path());
        } catch (const std::runtime_error& error) {
            expect(std::string_view(error.what()).find("truncated") != std::string_view::npos,
                   "truncated payload reported an unrelated error");
            return;
        }
        throw std::runtime_error("truncated binary payload was accepted");
    }

    void testOversizedAsciiTokenIsRejected() {
        TemporaryPly file("oversized_token");
        std::string payload =
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n";
        payload.append(129, '1');
        payload += " 0 0\n";
        writeAscii(file.path(), payload);

        try {
            (void)lfs::arc::loadPly(file.path());
        } catch (const std::runtime_error& error) {
            expect(std::string_view(error.what()).find("token exceeds safety limit") !=
                       std::string_view::npos,
                   "oversized token reported an unrelated error");
            return;
        }
        throw std::runtime_error("oversized ASCII token was accepted");
    }

    void testOversizedHeaderLineIsRejected() {
        TemporaryPly file("oversized_header");
        std::string payload = "ply\nformat ascii 1.0\ncomment ";
        payload.append(1024 * 1024 + 1, 'x');
        payload +=
            "\nelement vertex 1\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n0 0 0\n";
        writeAscii(file.path(), payload);

        try {
            (void)lfs::arc::loadPly(file.path());
        } catch (const std::runtime_error& error) {
            expect(std::string_view(error.what()).find("header line exceeds safety limit") !=
                       std::string_view::npos,
                   "oversized header line reported an unrelated error");
            return;
        }
        throw std::runtime_error("oversized header line was accepted");
    }

    void testRobustFramingBoundsIgnoreSparseOutliers() {
        TemporaryPly file("robust_framing");
        std::string payload =
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1002\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "-1000000 0 0\n";
        for (int value = 0; value < 1000; ++value) {
            payload += std::to_string(value) + " " + std::to_string(value % 10) +
                       " " + std::to_string(value % 20) + "\n";
        }
        payload += "1000000 0 0\n";
        writeAscii(file.path(), payload);

        const lfs::arc::PointCloud cloud = lfs::arc::loadPly(file.path());
        expectNear(cloud.boundsMin[0], -1000000.0F, "exact minimum keeps outlier");
        expectNear(cloud.boundsMax[0], 1000000.0F, "exact maximum keeps outlier");
        expect(cloud.framingBoundsMin[0] > -100.0F,
               "framing minimum did not trim sparse outlier");
        expect(cloud.framingBoundsMax[0] < 1100.0F,
               "framing maximum did not trim sparse outlier");
        expect(cloud.framingBoundsMin[1] <= 0.0F && cloud.framingBoundsMax[1] >= 9.0F,
               "framing bounds lost the dense y range");
    }

    void testHugeTruncatedDeclarationIsRejectedBeforeAllocation() {
        TemporaryPly file("huge_truncated");
        writeAscii(file.path(),
                   "ply\n"
                   "format ascii 1.0\n"
                   "element vertex 50000000\n"
                   "property float x\n"
                   "property float y\n"
                   "property float z\n"
                   "end_header\n");

        try {
            (void)lfs::arc::loadPly(file.path());
        } catch (const std::runtime_error& error) {
            expect(std::string_view(error.what()).find("payload is truncated") !=
                       std::string_view::npos,
                   "huge truncated declaration reported an unrelated error");
            return;
        }
        throw std::runtime_error("huge truncated declaration was accepted");
    }

    void testRobustFramingBoundsRemainFiniteAtFloatLimits() {
        TemporaryPly file("finite_framing");
        std::string payload =
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1000\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n";
        for (int index = 0; index < 20; ++index) {
            payload += "-3.402823466e+38 0 0\n";
        }
        for (int index = 0; index < 960; ++index) {
            payload += "0 0 0\n";
        }
        for (int index = 0; index < 20; ++index) {
            payload += "3.402823466e+38 0 0\n";
        }
        writeAscii(file.path(), payload);

        const lfs::arc::PointCloud cloud = lfs::arc::loadPly(file.path());
        expect(std::isfinite(cloud.framingBoundsMin[0]),
               "framing minimum overflowed at float limits");
        expect(std::isfinite(cloud.framingBoundsMax[0]),
               "framing maximum overflowed at float limits");
        expectNear(cloud.framingBoundsMin[0], -std::numeric_limits<float>::max(),
                   "framing minimum clamp");
        expectNear(cloud.framingBoundsMax[0], std::numeric_limits<float>::max(),
                   "framing maximum clamp");
    }

} // namespace

int main() {
    const std::vector<std::pair<std::string_view, std::function<void()>>> tests{
        {"ASCII RGB and element skipping", testAsciiRgbAndElementSkipping},
        {"binary Gaussian SH color", testBinaryGaussianSphericalHarmonicColor},
        {"truncated binary rejection", testTruncatedBinaryIsRejected},
        {"oversized ASCII token rejection", testOversizedAsciiTokenIsRejected},
        {"oversized header line rejection", testOversizedHeaderLineIsRejected},
        {"robust framing bounds", testRobustFramingBoundsIgnoreSparseOutliers},
        {"huge truncated declaration rejection",
         testHugeTruncatedDeclarationIsRejectedBeforeAllocation},
        {"finite framing at float limits",
         testRobustFramingBoundsRemainFiniteAtFloatLimits},
    };

    std::size_t failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
