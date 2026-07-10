// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <vector>

namespace lfs::arc {

    struct Vertex {
        float x;
        float y;
        float z;
        float red;
        float green;
        float blue;
    };

    static_assert(sizeof(Vertex) == 6 * sizeof(float));

    struct PointCloud {
        std::vector<Vertex> vertices;
        std::array<float, 3> boundsMin{};
        std::array<float, 3> boundsMax{};
        std::array<float, 3> framingBoundsMin{};
        std::array<float, 3> framingBoundsMax{};
    };

    PointCloud loadPly(const std::filesystem::path& path);

} // namespace lfs::arc
