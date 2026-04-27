/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <rasterization_api.h>

namespace lfs::training::optimizer {

    struct HipFusedAdamParam {
        float* param = nullptr;
        float* exp_avg = nullptr;
        float* exp_avg_sq = nullptr;
        float lr = 0.0f;
        float bias_correction1_rcp = 1.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
        int64_t n_elements = 0;
    };

    struct HipFusedProjectionBackwardOptimizerConfig {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;

        float* densification_info = nullptr;
        const float* densification_error_map = nullptr;
        const float* grad_image = nullptr;
        const float* grad_alpha = nullptr;
        const float* image = nullptr;
        const float* alpha = nullptr;

        const float* w2c = nullptr;
        const float* cam_position = nullptr;
        fast_lfs::rasterization::ForwardContext forward_ctx{};

        float* grad_opacity = nullptr;
        float* grad_sh0 = nullptr;
        float* grad_shN = nullptr;

        HipFusedAdamParam means;
        HipFusedAdamParam scaling;
        HipFusedAdamParam rotation;
        HipFusedAdamParam opacity;
        HipFusedAdamParam sh0;
        HipFusedAdamParam shN;

        int n_primitives = 0;
        int active_sh_bases = 0;
        int total_bases_sh_rest = 0;
        int width = 0;
        int height = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        bool mip_filter = false;
        DensificationType densification_type = DensificationType::None;

        float scale_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        bool update_shN = false;
    };

    void launch_hip_fused_projection_backward_optimizer(
        const HipFusedProjectionBackwardOptimizerConfig& config,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::optimizer
