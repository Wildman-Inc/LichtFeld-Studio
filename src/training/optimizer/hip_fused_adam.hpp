/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::optimizer {

    constexpr int HIP_FUSED_ADAM_MAX_GROUPS = 6;

    struct HipFusedAdamGroup {
        float* param = nullptr;
        float* exp_avg = nullptr;
        float* exp_avg_sq = nullptr;
        const float* grad = nullptr;
        int64_t n_elements = 0;
        float lr = 0.0f;
        float bias_correction1_rcp = 1.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
    };

    struct HipFusedAdamLaunchConfig {
        HipFusedAdamGroup groups[HIP_FUSED_ADAM_MAX_GROUPS];
        int group_count = 0;
        int64_t total_elements = 0;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
    };

    void launch_hip_fused_adam_step(
        const HipFusedAdamLaunchConfig& config,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::optimizer
