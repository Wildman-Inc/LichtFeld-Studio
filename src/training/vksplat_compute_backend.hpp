/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <cstdlib>
#include <expected>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>

namespace lfs::training::vksplat_compute {

    struct Progress {
        int iteration = 0;
        int total_iterations = 0;
        int num_gaussians = 0;
        float loss = 0.0f;
        float l1 = 0.0f;
        float psnr = 0.0f;
        float ssim = 0.0f;
        bool has_metrics = false;
    };

    struct StepControl {
        std::function<bool(int)> before_step;
        std::function<void(const Progress&)> after_step;
    };

    [[nodiscard]] inline bool env_flag_enabled(const char* name) {
        const char* raw = std::getenv(name);
        if (!raw) {
            return false;
        }

        const std::string_view value(raw);
        return !value.empty() && value != "0" && value != "false" && value != "FALSE" &&
               value != "off" && value != "OFF" && value != "no" && value != "NO";
    }

    [[nodiscard]] inline bool is_requested() {
        if (env_flag_enabled("LFS_USE_VKSPLAT_COMPUTE")) {
            return true;
        }

        const char* raw = std::getenv("LFS_TRAINING_BACKEND");
        if (!raw) {
            return false;
        }

        const std::string_view backend(raw);
        return backend == "vksplat" || backend == "VKSPLAT" ||
               backend == "vulkan" || backend == "VULKAN";
    }

#if defined(LFS_VULKAN_COMPUTE_BACKEND)
    [[nodiscard]] std::expected<void, std::string> run_training(
        const lfs::core::param::TrainingParameters& params,
        std::stop_token stop_token,
        StepControl control);
#else
    [[nodiscard]] inline std::expected<void, std::string> run_training(
        const lfs::core::param::TrainingParameters&,
        std::stop_token,
        StepControl) {
        return std::unexpected("vksplat Vulkan compute backend is not compiled. Reconfigure with -DLFS_ENABLE_VKSPLAT=ON.");
    }
#endif

} // namespace lfs::training::vksplat_compute
