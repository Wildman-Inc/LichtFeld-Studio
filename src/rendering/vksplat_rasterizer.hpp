/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering.hpp"

#include <glm/glm.hpp>
#include <vector>

namespace lfs::rendering {

    using lfs::core::Tensor;

    struct VkSplatRasterizeRequest {
        int sh_degree = 3;
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};
        bool transparent_background = false;
        const std::vector<glm::mat4>* model_transforms = nullptr;
    };

    struct VkSplatRasterizeOutput {
        Tensor image;
    };

    [[nodiscard]] bool vksplat_is_available();

    Result<VkSplatRasterizeOutput> vksplat_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const VkSplatRasterizeRequest& request);

} // namespace lfs::rendering
