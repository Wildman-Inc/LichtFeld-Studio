/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hip_fused_adam.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace lfs::training::optimizer {

    namespace {
        constexpr int BLOCK_SIZE = 256;

        __global__ void hip_fused_adam_kernel(HipFusedAdamLaunchConfig config) {
            int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (linear_idx >= config.total_elements) {
                return;
            }

            const float beta1_comp = 1.0f - config.beta1;
            const float beta2_comp = 1.0f - config.beta2;

            for (int group_idx = 0; group_idx < config.group_count; ++group_idx) {
                const auto& group = config.groups[group_idx];
                if (linear_idx >= group.n_elements) {
                    linear_idx -= group.n_elements;
                    continue;
                }

                const float grad = group.grad[linear_idx];
                const float moment1 = config.beta1 * group.exp_avg[linear_idx] + beta1_comp * grad;
                const float moment2 = config.beta2 * group.exp_avg_sq[linear_idx] + beta2_comp * grad * grad;
                const float denom = sqrtf(moment2) * group.bias_correction2_sqrt_rcp + config.eps;
                const float step_size = group.lr * group.bias_correction1_rcp;

                group.param[linear_idx] -= step_size * moment1 / denom;
                group.exp_avg[linear_idx] = moment1;
                group.exp_avg_sq[linear_idx] = moment2;
                return;
            }
        }
    } // namespace

    void launch_hip_fused_adam_step(
        const HipFusedAdamLaunchConfig& config,
        cudaStream_t stream) {
        if (config.group_count <= 0 || config.total_elements <= 0) {
            return;
        }

        const int blocks = static_cast<int>((config.total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
        hip_fused_adam_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(config);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("hip_fused_adam_kernel launch failed: ") +
                                     cudaGetErrorString(err));
        }
    }

} // namespace lfs::training::optimizer
