/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/pipelined_image_loader.hpp"

#include <cstddef>
#include <string_view>

namespace lfs::training::hip_fused_splat_backend {

    struct Options {
        bool resident_gpu_cache = true;
        bool fused_optimizer = true;
        bool loss_raster_boundary_fusion = true;
        bool rocprim_sort_scan = true;
        size_t resident_gpu_cache_bytes = 0;
        size_t prefetch_count = 0;
    };

    [[nodiscard]] bool is_requested();
    [[nodiscard]] bool resident_pipeline_requested();
    [[nodiscard]] Options read_options();
    [[nodiscard]] bool fused_optimizer_enabled();
    [[nodiscard]] bool loss_raster_boundary_fusion_enabled();
    [[nodiscard]] std::string_view sort_scan_policy_name();

    void configure_pipelined_loader(
        lfs::io::PipelinedLoaderConfig& config,
        size_t train_dataset_size);

} // namespace lfs::training::hip_fused_splat_backend
