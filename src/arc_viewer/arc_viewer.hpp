// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

namespace lfs::arc {

    class ArcViewer {
    public:
        explicit ArcViewer(std::optional<std::uint32_t> gpuIndex = std::nullopt);
        ~ArcViewer();

        ArcViewer(const ArcViewer&) = delete;
        ArcViewer& operator=(const ArcViewer&) = delete;

        void loadPointCloud(const std::filesystem::path& path);
        int run();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::arc
