// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "arc_viewer.hpp"

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

    void printHelp(std::ostream& output) {
        output
            << "Usage: LichtFeld-Studio-Arc-Viewer [OPTIONS] [PLY_PATH]\n"
            << "\n"
            << "Minimal SDL3/Vulkan PLY point viewer. With no path, drop a PLY file\n"
            << "onto the window. ASCII and binary_little_endian PLY 1.0 are supported.\n"
            << "\n"
            << "Options:\n"
            << "  -h, --help     Show this help and exit\n"
            << "  --gpu INDEX    Select a Vulkan GPU by enumeration index\n"
            << "  --             Treat the remaining argument as PLY_PATH\n"
            << "\n"
            << "Controls:\n"
            << "  Left drag     Orbit\n"
            << "  Middle drag   Pan\n"
            << "  Right drag    Zoom\n"
            << "  Mouse wheel   Change point size\n"
            << "  Drop PLY      Load another point cloud\n"
            << "  Escape        Exit\n";
    }

#if defined(_WIN32)
    using ArgumentChar = wchar_t;
    using ArgumentView = std::wstring_view;

    constexpr ArgumentView kShortHelp = L"-h";
    constexpr ArgumentView kLongHelp = L"--help";
    constexpr ArgumentView kGpuOption = L"--gpu";
    constexpr ArgumentView kEndOfOptions = L"--";

    std::filesystem::path pathFromArgument(const ArgumentChar* text) {
        return std::filesystem::path(text);
    }

    std::string argumentForError(ArgumentView argument) {
        const std::u8string utf8 = std::filesystem::path(argument).u8string();
        std::string result;
        result.reserve(utf8.size());
        for (const char8_t byte : utf8) {
            result.push_back(static_cast<char>(byte));
        }
        return result;
    }
#else
    using ArgumentChar = char;
    using ArgumentView = std::string_view;

    constexpr ArgumentView kShortHelp = "-h";
    constexpr ArgumentView kLongHelp = "--help";
    constexpr ArgumentView kGpuOption = "--gpu";
    constexpr ArgumentView kEndOfOptions = "--";

    std::filesystem::path pathFromArgument(const ArgumentChar* text) {
        std::u8string utf8;
        const std::string_view bytes(text);
        utf8.reserve(bytes.size());
        for (const unsigned char byte : bytes) {
            utf8.push_back(static_cast<char8_t>(byte));
        }
        return std::filesystem::path(utf8);
    }

    std::string argumentForError(ArgumentView argument) {
        return std::string(argument);
    }
#endif

    struct Options {
        std::optional<std::filesystem::path> initialPath;
        std::optional<std::uint32_t> gpuIndex;
    };

    std::uint32_t parseGpuIndex(ArgumentView argument) {
        if (argument.empty()) {
            throw std::runtime_error("--gpu requires a non-negative decimal INDEX");
        }

        std::uint32_t value = 0;
        for (const ArgumentChar character : argument) {
            if (character < static_cast<ArgumentChar>('0') ||
                character > static_cast<ArgumentChar>('9')) {
                throw std::runtime_error("--gpu requires a non-negative decimal INDEX");
            }
            const std::uint32_t digit =
                static_cast<std::uint32_t>(character - static_cast<ArgumentChar>('0'));
            if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / 10U) {
                throw std::runtime_error("--gpu INDEX is too large");
            }
            value = value * 10U + digit;
        }
        return value;
    }

    Options parseArguments(int argc, ArgumentChar* argv[]) {
        Options options;
        bool parseOptions = true;
        for (int index = 1; index < argc; ++index) {
            const ArgumentView argument(argv[index]);
            if (parseOptions && (argument == kShortHelp || argument == kLongHelp)) {
                printHelp(std::cout);
                std::exit(0);
            }
            if (parseOptions && argument == kGpuOption) {
                if (options.gpuIndex) {
                    throw std::runtime_error("--gpu may only be specified once");
                }
                if (++index >= argc) {
                    throw std::runtime_error("--gpu requires an INDEX");
                }
                options.gpuIndex = parseGpuIndex(argv[index]);
                continue;
            }
            if (parseOptions && argument == kEndOfOptions) {
                parseOptions = false;
                continue;
            }
            if (parseOptions && !argument.empty() &&
                argument.front() == static_cast<ArgumentChar>('-')) {
                throw std::runtime_error("Unknown option: " + argumentForError(argument));
            }
            if (options.initialPath) {
                throw std::runtime_error("Only one PLY_PATH may be specified");
            }
            options.initialPath = pathFromArgument(argv[index]);
        }
        return options;
    }

    int reportFatal(const std::string& message) {
        std::cerr << "Error: " << message << '\n';
        if (!SDL_ShowSimpleMessageBox(
                SDL_MESSAGEBOX_ERROR, "LichtFeld Studio Arc Viewer", message.c_str(), nullptr)) {
            std::cerr << "SDL_ShowSimpleMessageBox failed: " << SDL_GetError() << '\n';
        }
        return 1;
    }

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    SDL_SetMainReady();
    try {
        const Options options = parseArguments(argc, argv);
        lfs::arc::ArcViewer viewer(options.gpuIndex);
        if (options.initialPath) {
            viewer.loadPointCloud(*options.initialPath);
        }
        return viewer.run();
    } catch (const std::exception& error) {
        printHelp(std::cerr);
        return reportFatal(error.what());
    }
}
