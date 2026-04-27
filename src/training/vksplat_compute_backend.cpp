/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_compute_backend.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "gs_trainer.h"

namespace lfs::training::vksplat_compute {

    namespace {
        struct SampleMetrics {
            float loss = 0.0f;
            float l1 = 0.0f;
            float psnr = 0.0f;
            float ssim = 0.0f;
            bool valid = false;
        };

        [[nodiscard]] std::filesystem::path shader_root() {
#if defined(LFS_VKSPLAT_SHADER_PATH)
            return std::filesystem::path{LFS_VKSPLAT_SHADER_PATH};
#else
            return {};
#endif
        }

        [[nodiscard]] std::string with_trailing_slash(const std::filesystem::path& path) {
            auto normalized = path.generic_string();
            if (!normalized.empty() && normalized.back() != '/') {
                normalized.push_back('/');
            }
            return normalized;
        }

        [[nodiscard]] int metric_interval_from_env() {
            const char* raw = std::getenv("LFS_VKSPLAT_METRIC_INTERVAL");
            if (!raw || raw[0] == '\0') {
                return 100;
            }
            try {
                return std::max(0, std::stoi(raw));
            } catch (...) {
                return 100;
            }
        }

        [[nodiscard]] TrainerConfig::CacheImage image_cache_from_env() {
            const char* raw = std::getenv("LFS_VKSPLAT_IMAGE_CACHE");
            if (!raw || raw[0] == '\0') {
                return TrainerConfig::CacheImage::CPU;
            }

            const std::string_view value(raw);
            if (value == "cpu" || value == "CPU") {
                return TrainerConfig::CacheImage::CPU;
            }
            if (value == "gpu" || value == "GPU" || value == "auto" || value == "AUTO") {
                return TrainerConfig::CacheImage::GPU;
            }

            LOG_WARN("Unknown LFS_VKSPLAT_IMAGE_CACHE='{}'; using CPU image cache", raw);
            return TrainerConfig::CacheImage::CPU;
        }

        [[nodiscard]] const char* image_cache_name(const TrainerConfig::CacheImage cache) {
            return cache == TrainerConfig::CacheImage::GPU ? "gpu" : "cpu";
        }

        [[nodiscard]] double clamp_unit_finite(const double value) {
            if (!std::isfinite(value)) {
                return 0.0;
            }
            return std::clamp(value, 0.0, 1.0);
        }

        [[nodiscard]] std::map<std::string, std::string> make_shader_map() {
            const std::filesystem::path root = shader_root();
            if (root.empty()) {
                throw std::runtime_error("vksplat shader root was not configured");
            }

            const std::array<std::string_view, 44> shader_names{
                "projection_forward",
                "generate_keys",
                "compute_tile_ranges",
                "rasterize_forward",
                "rasterize_backward_0",
                "rasterize_backward_1",
                "rasterize_backward_2",
                "rasterize_backward_3",
                "rasterize_backward_4",
                "cumsum_single_pass",
                "cumsum_block_scan",
                "cumsum_scan_block_sums",
                "cumsum_add_block_offsets",
                "radix_sort/upsweep",
                "radix_sort/spine",
                "radix_sort/downsweep",
                "ssim_forward",
                "ssim_backward",
                "fused_projection_backward_optimizer",
                "sum",
                "where",
                "default_update_state",
                "default_compute_grow_mask",
                "default_duplicate",
                "default_split",
                "default_compute_prune_mask",
                "default_prune",
                "default_prune_mean",
                "default_prune_sh",
                "default_reset_opa",
                "mcmc_inject_noise",
                "mcmc_compute_probs",
                "mcmc_compute_relocation_index_map",
                "mcmc_compute_relocation",
                "mcmc_update_relocation",
                "mcmc_compute_add_index_map",
                "mcmc_compute_add",
                "mcmc_update_add",
                "morton_sort_compute_stats",
                "morton_sort_generate_keys",
                "morton_sort_apply_indices",
                "morton_sort_apply_indices_sh",
                "morton_sort_update_buffer",
                "morton_sort_update_buffer_sh",
            };

            std::map<std::string, std::string> spirv_paths;
            for (const auto name_view : shader_names) {
                const std::string name{name_view};
                const std::filesystem::path path =
                    name.starts_with("radix_sort/")
                        ? root / std::filesystem::path{name + ".spv"}
                        : root / "generated" / std::filesystem::path{name + ".spv"};
                if (!std::filesystem::exists(path)) {
                    throw std::runtime_error(std::format("vksplat shader is missing: {}", path.generic_string()));
                }
                spirv_paths[name] = path.generic_string();
            }
            return spirv_paths;
        }

        [[nodiscard]] std::filesystem::path choose_sparse_dir(const std::filesystem::path& dataset_dir) {
            const auto sparse0 = dataset_dir / "sparse" / "0";
            if (std::filesystem::exists(sparse0 / "cameras.bin") ||
                std::filesystem::exists(sparse0 / "cameras.txt")) {
                return sparse0;
            }

            const auto sparse = dataset_dir / "sparse";
            if (std::filesystem::exists(sparse / "cameras.bin") ||
                std::filesystem::exists(sparse / "cameras.txt")) {
                return sparse;
            }

            return sparse0;
        }

        [[nodiscard]] std::filesystem::path choose_output_dir(
            const lfs::core::param::TrainingParameters& params) {
            if (!params.dataset.output_path.empty()) {
                return params.dataset.output_path;
            }
            return lfs::core::param::default_dataset_output_path(params.dataset.data_path);
        }

        [[nodiscard]] std::string choose_output_name(
            const lfs::core::param::TrainingParameters& params) {
            if (!params.dataset.output_name.empty()) {
                std::filesystem::path name{params.dataset.output_name};
                if (name.extension().empty()) {
                    name += ".ply";
                }
                return name.generic_string();
            }
            return "splat.ply";
        }

        [[nodiscard]] int clamp_to_int(const size_t value) {
            return static_cast<int>(std::min<size_t>(value, static_cast<size_t>(std::numeric_limits<int>::max())));
        }

        [[nodiscard]] TrainerConfig make_config(const lfs::core::param::TrainingParameters& params) {
            const auto& opt = params.optimization;
            const auto output_dir = choose_output_dir(params);
            const auto output_ply = output_dir / choose_output_name(params);
            std::filesystem::create_directories(output_dir);

            TrainerConfig config{};
            config.output_dir = with_trailing_slash(output_dir);
            config.output_ply = output_ply.generic_string();
            config.dataset_dir = with_trailing_slash(params.dataset.data_path);
            config.image_dir = with_trailing_slash(params.dataset.data_path / params.dataset.images);
            config.mask_dir = "";
            config.sparse_dir = with_trailing_slash(choose_sparse_dir(params.dataset.data_path));
            config.eval_interval = std::max(2, params.dataset.test_every);
            config.image_cache_device = image_cache_from_env();

            config.global_scale = 1.0f;
            config.init_scale = opt.init_scaling;
            config.init_opacity = opt.init_opacity;

            if (lfs::core::param::strategy_names_match(opt.strategy, lfs::core::param::kStrategyMCMC)) {
                config.strategy = TrainerConfig::Strategy::MCMC;
            } else {
                if (!lfs::core::param::strategy_names_match(opt.strategy, "default")) {
                    LOG_WARN("vksplat compute supports default/MCMC densification; using default for '{}'", opt.strategy);
                }
                config.strategy = TrainerConfig::Strategy::Default;
            }

            const int total_steps = std::max(1, opt.resolved_total_iterations());
            config.max_steps = total_steps;
            config.ssim_lambda = opt.lambda_dssim;
            config.means_lr = opt.means_lr;
            config.means_lr_final = opt.means_lr_end;
            config.features_dc_lr = opt.shs_lr;
            config.features_rest_lr = opt.shs_lr / 20.0f;
            config.opacities_lr = opt.opacity_lr;
            config.scales_lr = opt.scaling_lr;
            config.quats_lr = opt.rotation_lr;
            config.scale_reg = opt.scale_reg;
            config.opacity_reg = opt.opacity_reg;

            config.refine_start_iter = clamp_to_int(opt.start_refine);
            config.refine_stop_iter = clamp_to_int(opt.stop_refine);
            config.refine_every = clamp_to_int(opt.refine_every);

            config.prune_opa = opt.prune_opacity;
            config.grow_grad2d = opt.grad_threshold;
            config.grow_scale3d = opt.grow_scale3d;
            config.grow_scale2d = opt.grow_scale2d;
            config.prune_scale3d = opt.prune_scale3d;
            config.prune_scale2d = opt.prune_scale2d;
            config.refine_scale2d_stop_iter = 0;
            config.reset_every = clamp_to_int(opt.reset_every);
            config.stop_reset_at = config.refine_stop_iter;
            config.pause_refine_after_reset = clamp_to_int(opt.pause_refine_after_reset);

            config.noise_lr = 5.0e5f;
            config.min_opacity = opt.min_opacity;
            config.grow_factor = 1.05f;
            config.cap_max = std::max(opt.max_cap, 1);
            return config;
        }

        void run_forward(VulkanGSTrainer& trainer, VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers,
                         const TrainerConfig& config) {
            const size_t reserve = config.strategy == TrainerConfig::Strategy::MCMC
                                       ? static_cast<size_t>(std::max(config.cap_max, 0))
                                       : 0u;
            uniforms.num_splats = static_cast<uint32_t>(std::min<size_t>(
                buffers.num_splats, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
            trainer.executeProjectionForward(uniforms, buffers, reserve);
            trainer.executeCalculateIndexBufferOffset(buffers);
            if (buffers.num_indices != 0) {
                trainer.executeGenerateKeys(uniforms, buffers);
                trainer.executeSort(uniforms, buffers, -1);
                trainer.executeComputeTileRanges(uniforms, buffers);
            }
            trainer.executeRasterizeForward(uniforms, buffers);
        }

        void cap_initial_splats(VulkanGSPipelineBuffers& buffers, const TrainerConfig& config) {
            if (config.cap_max <= 0) {
                return;
            }

            const size_t target_count = static_cast<size_t>(config.cap_max);
            if (buffers.num_splats <= target_count) {
                return;
            }

            std::vector<size_t> selected(buffers.num_splats);
            std::iota(selected.begin(), selected.end(), size_t{0});
            std::mt19937 rng{0};
            std::shuffle(selected.begin(), selected.end(), rng);
            selected.resize(target_count);
            std::sort(selected.begin(), selected.end());

            auto compact = [&selected, target_count](auto& buffer, const size_t stride) {
                for (size_t dst = 0; dst < target_count; ++dst) {
                    const size_t src = selected[dst];
                    if (dst == src) {
                        continue;
                    }
                    std::copy_n(
                        buffer.begin() + static_cast<std::ptrdiff_t>(src * stride),
                        stride,
                        buffer.begin() + static_cast<std::ptrdiff_t>(dst * stride));
                }
                buffer.resize(target_count * stride);
            };

            compact(buffers.xyz_ws, 3);
            compact(buffers.sh_coeffs, 16 * 3);
            compact(buffers.rotations, 4);
            compact(buffers.scales_opacs, 4);
            buffers.num_splats = target_count;

            LOG_INFO("vksplat initial splats capped to {} from COLMAP input", target_count);
        }

        SampleMetrics sample_metrics(VulkanGSTrainer& trainer, VulkanGSPipelineBuffers& buffers,
                                     const TrainerConfig& config, const size_t train_idx) {
            trainer.copyFromDevice(buffers.pixel_state);

            const auto& rendered = buffers.pixel_state;
            const auto& reference = trainer.get_train_image(train_idx).buffer;
            const size_t pixels = std::min(rendered.size(), reference.size()) / 4;
            if (pixels == 0) {
                return {};
            }

            double l1_sum = 0.0;
            double mse_sum = 0.0;
            std::array<double, 3> render_sum{};
            std::array<double, 3> ref_sum{};
            std::array<double, 3> render_sq_sum{};
            std::array<double, 3> ref_sq_sum{};
            std::array<double, 3> cross_sum{};

            for (size_t i = 0; i < pixels; ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    const double render_value = clamp_unit_finite(static_cast<double>(rendered[4 * i + c]));
                    const double ref_value = clamp_unit_finite(static_cast<double>(reference[4 * i + c]) / 255.0);
                    const double delta = render_value - ref_value;
                    l1_sum += std::abs(delta);
                    mse_sum += delta * delta;
                    render_sum[c] += render_value;
                    ref_sum[c] += ref_value;
                    render_sq_sum[c] += render_value * render_value;
                    ref_sq_sum[c] += ref_value * ref_value;
                    cross_sum[c] += render_value * ref_value;
                }
            }

            const double sample_count = static_cast<double>(pixels);
            const double channel_count = sample_count * 3.0;
            const double l1 = l1_sum / channel_count;
            const double mse = mse_sum / channel_count;
            const double psnr = mse <= 0.0 ? 99.0 : 10.0 * std::log10(1.0 / mse);

            constexpr double c1 = 0.01 * 0.01;
            constexpr double c2 = 0.03 * 0.03;
            double ssim_sum = 0.0;
            for (size_t c = 0; c < 3; ++c) {
                const double render_mean = render_sum[c] / sample_count;
                const double ref_mean = ref_sum[c] / sample_count;
                const double render_var = std::max(0.0, render_sq_sum[c] / sample_count - render_mean * render_mean);
                const double ref_var = std::max(0.0, ref_sq_sum[c] / sample_count - ref_mean * ref_mean);
                const double covariance = cross_sum[c] / sample_count - render_mean * ref_mean;
                const double numerator = (2.0 * render_mean * ref_mean + c1) * (2.0 * covariance + c2);
                const double denominator = (render_mean * render_mean + ref_mean * ref_mean + c1) *
                                           (render_var + ref_var + c2);
                ssim_sum += denominator > 0.0 ? numerator / denominator : 0.0;
            }
            const double ssim = std::clamp(ssim_sum / 3.0, -1.0, 1.0);
            const double lambda = std::clamp(static_cast<double>(config.ssim_lambda), 0.0, 1.0);
            const double loss = (1.0 - lambda) * l1 + lambda * (1.0 - ssim);

            return SampleMetrics{
                .loss = static_cast<float>(loss),
                .l1 = static_cast<float>(l1),
                .psnr = static_cast<float>(psnr),
                .ssim = static_cast<float>(ssim),
                .valid = std::isfinite(loss) && std::isfinite(l1) && std::isfinite(psnr) && std::isfinite(ssim)};
        }

        SampleMetrics run_train_step(VulkanGSTrainer& trainer, VulkanGSRendererUniforms& uniforms,
                                     VulkanGSPipelineBuffers& buffers, const TrainerConfig& config,
                                     const size_t train_idx, const int step, const bool measure_metrics) {
            trainer.get_train_camera(train_idx, uniforms);
            uniforms.step = static_cast<uint32_t>(step);

            auto device_guard = DeviceGuard(&trainer);
            run_forward(trainer, uniforms, buffers, config);
            SampleMetrics metrics;
            if (measure_metrics) {
                metrics = sample_metrics(trainer, buffers, config, train_idx);
            }
            if (buffers.num_indices != 0) {
                trainer.executeComputeSSIMGradient(config, uniforms, buffers, train_idx);
            }
            trainer.executeRasterizeBackward(uniforms, buffers);
            trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step + 1);
            if (config.strategy == TrainerConfig::Strategy::MCMC) {
                trainer.executeMCMCPostBackward(config, uniforms, buffers, step);
            } else {
                trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
            }
            return metrics;
        }

    } // namespace

    std::expected<void, std::string> run_training(
        const lfs::core::param::TrainingParameters& params,
        std::stop_token stop_token,
        StepControl control) {
        try {
            VulkanGSTrainer trainer;
            VulkanGSRendererUniforms uniforms{};
            VulkanGSPipelineBuffers buffers;
            TrainerConfig config = make_config(params);

            trainer.initialize(make_shader_map(), 0);
            auto cleanup = [&] {
                try {
                    trainer.cleanupBuffers(buffers);
                    trainer.cleanup();
                } catch (const std::exception& e) {
                    LOG_WARN("vksplat cleanup failed: {}", e.what());
                }
            };

            try {
                trainer.load_colmap_dataset(config, buffers);
                cap_initial_splats(buffers, config);
                if (trainer.num_train() == 0) {
                    cleanup();
                    return std::unexpected("vksplat dataset split produced no training images");
                }

                std::vector<size_t> order(trainer.num_train());
                std::iota(order.begin(), order.end(), size_t{0});
                std::mt19937 rng{0};

                const int total_steps = std::max(1, params.optimization.resolved_total_iterations());
                const int sh_interval = static_cast<int>(std::max<size_t>(1, params.optimization.sh_degree_interval));
                const uint32_t max_sh = static_cast<uint32_t>(std::clamp(params.optimization.sh_degree, 0, 3));
                const int metric_interval = metric_interval_from_env();
                SampleMetrics last_metrics;

                LOG_INFO("Starting fused vksplat Vulkan compute training: {} steps, {} train images, image_cache={}, output {}",
                         total_steps, trainer.num_train(), image_cache_name(config.image_cache_device), config.output_ply);

                for (int step = 0; step < total_steps; ++step) {
                    if (stop_token.stop_requested()) {
                        break;
                    }
                    const int iteration = step + 1;
                    if (control.before_step && !control.before_step(iteration)) {
                        break;
                    }

                    if (step > 0 && step % static_cast<int>(order.size()) == 0) {
                        std::shuffle(order.begin(), order.end(), rng);
                    }

                    uniforms.active_sh = std::min<uint32_t>(static_cast<uint32_t>(step / sh_interval), max_sh);
                    const size_t train_idx = order[step % order.size()];
                    const bool measure_metrics =
                        metric_interval > 0 &&
                        (iteration == 1 || iteration == total_steps || iteration % metric_interval == 0);
                    const SampleMetrics metrics =
                        run_train_step(trainer, uniforms, buffers, config, train_idx, step, measure_metrics);
                    if (metrics.valid) {
                        last_metrics = metrics;
                        LOG_INFO("vksplat sample metrics iter {}: loss={:.6f}, l1={:.6f}, psnr={:.2f} dB, global_ssim={:.4f}",
                                 iteration, metrics.loss, metrics.l1, metrics.psnr, metrics.ssim);
                    } else if (measure_metrics) {
                        LOG_WARN("vksplat sample metrics skipped at iter {}: rendered_values={}, reference_values={}, tile_indices={}",
                                 iteration,
                                 buffers.pixel_state.deviceSize(),
                                 trainer.get_train_image(train_idx).buffer.size(),
                                 buffers.num_indices);
                    }

                    if (control.after_step) {
                        control.after_step(Progress{
                            .iteration = iteration,
                            .total_iterations = total_steps,
                            .num_gaussians = static_cast<int>(buffers.num_splats),
                            .loss = last_metrics.loss,
                            .l1 = last_metrics.l1,
                            .psnr = last_metrics.psnr,
                            .ssim = last_metrics.ssim,
                            .has_metrics = last_metrics.valid});
                    }
                }

                if (!stop_token.stop_requested()) {
                    trainer.writePLY(config.output_ply, buffers);
                    LOG_INFO("vksplat Vulkan compute training wrote {}", config.output_ply);
                }

                cleanup();
                return {};
            } catch (...) {
                cleanup();
                throw;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("vksplat Vulkan compute training failed: {}", e.what()));
        }
    }

} // namespace lfs::training::vksplat_compute
