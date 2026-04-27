/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hip_fused_splat_backend.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <initializer_list>
#include <string_view>

namespace lfs::training::hip_fused_splat_backend {

    namespace {
        constexpr double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;

        [[nodiscard]] bool env_flag_enabled(const char* name) {
            const char* raw = std::getenv(name);
            if (!raw) {
                return false;
            }

            const std::string_view value(raw);
            return !value.empty() && value != "0" && value != "false" && value != "FALSE" &&
                   value != "off" && value != "OFF" && value != "no" && value != "NO";
        }

        [[nodiscard]] bool env_flag_disabled(const char* name) {
            const char* raw = std::getenv(name);
            if (!raw) {
                return false;
            }

            const std::string_view value(raw);
            return value.empty() || value == "0" || value == "false" || value == "FALSE" ||
                   value == "off" || value == "OFF" || value == "no" || value == "NO";
        }

        [[nodiscard]] bool backend_matches_any(std::initializer_list<std::string_view> names) {
            const char* raw = std::getenv("LFS_TRAINING_BACKEND");
            if (!raw) {
                return false;
            }

            const std::string_view backend(raw);
            return std::find(names.begin(), names.end(), backend) != names.end();
        }

        [[nodiscard]] size_t read_cache_bytes_from_env() {
            const char* raw = std::getenv("LFS_HIP_FUSED_CACHE_GB");
            if (!raw || raw[0] == '\0') {
                raw = std::getenv("LFS_HIP_RESIDENT_CACHE_GB");
            }
            if (!raw || raw[0] == '\0') {
                return 0;
            }

            char* end = nullptr;
            const double value_gb = std::strtod(raw, &end);
            if (end != raw && std::isfinite(value_gb) && value_gb > 0.0) {
                return static_cast<size_t>(value_gb * BYTES_PER_GIB);
            }
            return 0;
        }

        [[nodiscard]] size_t read_prefetch_from_env() {
            const char* raw = std::getenv("LFS_HIP_FUSED_PREFETCH");
            if (!raw || raw[0] == '\0') {
                raw = std::getenv("LFS_HIP_RESIDENT_PREFETCH");
            }
            if (!raw || raw[0] == '\0') {
                return 0;
            }

            char* end = nullptr;
            const auto requested_prefetch = std::strtoull(raw, &end, 10);
            return (end != raw && requested_prefetch > 0)
                       ? static_cast<size_t>(requested_prefetch)
                       : size_t{0};
        }

        [[nodiscard]] size_t default_resident_cache_bytes() {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) {
                return 0;
            }

            constexpr size_t SAFETY_BYTES = 2ULL * 1024 * 1024 * 1024;
            const size_t cache_budget =
                free_bytes > SAFETY_BYTES
                    ? (free_bytes - SAFETY_BYTES) / 2
                    : free_bytes / 4;
            return std::max<size_t>(cache_budget, 512ULL * 1024 * 1024);
        }
    } // namespace

    bool is_requested() {
        if (env_flag_enabled("LFS_USE_HIP_FUSED_SPLAT") ||
            env_flag_enabled("LFS_USE_HIP_FUSED_BACKEND")) {
            return true;
        }

        return backend_matches_any({"hip-fused", "HIP-FUSED"});
    }

    bool resident_pipeline_requested() {
        if (is_requested()) {
            return true;
        }

        if (env_flag_enabled("LFS_USE_HIP_RESIDENT_PIPELINE") ||
            env_flag_enabled("LFS_HIP_RESIDENT")) {
            return true;
        }

        return backend_matches_any({
            "hip-resident",
            "HIP-RESIDENT",
            "hip-gpu-resident",
            "HIP-GPU-RESIDENT",
        });
    }

    Options read_options() {
        Options options;
        options.resident_gpu_cache = !env_flag_disabled("LFS_HIP_FUSED_RESIDENT_CACHE");
        options.projection_backward_optimizer = !env_flag_disabled("LFS_HIP_FUSED_PROJECTION_OPTIMIZER");
        options.fused_optimizer = !env_flag_disabled("LFS_HIP_FUSED_OPTIMIZER");
        options.loss_raster_boundary_fusion = !env_flag_disabled("LFS_HIP_FUSED_LOSS_RASTER");
        options.rocprim_sort_scan = !env_flag_disabled("LFS_HIP_FUSED_ROCPRIM_SORT_SCAN");
        options.resident_gpu_cache_bytes = read_cache_bytes_from_env();
        options.prefetch_count = read_prefetch_from_env();
        return options;
    }

    bool projection_backward_optimizer_enabled() {
        return is_requested() && read_options().projection_backward_optimizer;
    }

    bool fused_optimizer_enabled() {
        return is_requested() && read_options().fused_optimizer;
    }

    bool loss_raster_boundary_fusion_enabled() {
        return is_requested() && read_options().loss_raster_boundary_fusion;
    }

    std::string_view sort_scan_policy_name() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || (defined(LFS_USE_HIP) && LFS_USE_HIP)
        return "rocPRIM/hipCUB";
#else
        return "CUB";
#endif
    }

    void configure_pipelined_loader(
        lfs::io::PipelinedLoaderConfig& config,
        const size_t train_dataset_size) {
        const Options options = read_options();

        if (options.resident_gpu_cache) {
            config.resident_gpu_cache = true;
            config.resident_gpu_cache_max_bytes = options.resident_gpu_cache_bytes;
            if (config.resident_gpu_cache_max_bytes == 0) {
                config.resident_gpu_cache_max_bytes = default_resident_cache_bytes();
            }
            if (config.resident_gpu_cache_max_bytes == 0) {
                config.resident_gpu_cache_max_bytes = config.max_cache_bytes;
            }
        }

        if (options.prefetch_count > 0) {
            const size_t target_prefetch = std::clamp<size_t>(
                options.prefetch_count,
                size_t{1},
                std::max<size_t>(train_dataset_size, 1));
            config.prefetch_count = std::max(config.prefetch_count, target_prefetch);
            config.jpeg_batch_size = std::max(
                config.jpeg_batch_size,
                std::min<size_t>(16, config.prefetch_count));
            config.output_queue_size = std::max(
                config.output_queue_size,
                std::min<size_t>(16, std::max<size_t>(1, config.prefetch_count / 2)));
        }

        config.decoder_pool_size = std::max(
            config.decoder_pool_size,
            config.jpeg_batch_size);

        LOG_INFO(
            "HIP fused splat backend enabled (resident_cache={}, limit={:.1f} GB, projection_optimizer={}, fused_optimizer={}, loss_raster_boundary={}, sort_scan={})",
            config.resident_gpu_cache,
            config.resident_gpu_cache_max_bytes / BYTES_PER_GIB,
            options.projection_backward_optimizer,
            options.fused_optimizer,
            options.loss_raster_boundary_fusion,
            options.rocprim_sort_scan ? sort_scan_policy_name() : std::string_view{"fastgs-default"});
    }

} // namespace lfs::training::hip_fused_splat_backend
