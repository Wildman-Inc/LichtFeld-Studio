/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_rasterizer.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <mutex>
#include <string>

#if defined(LFS_HAS_VKSPLAT) && LFS_HAS_VKSPLAT
#include "gs_renderer.h"
#endif

namespace lfs::rendering {

    namespace {
        constexpr size_t kVkSplatShCoefficients = 16;
        constexpr size_t kVkSplatShChannels = 3;

        [[nodiscard]] size_t hashBytes(const void* data, const size_t size, size_t seed) {
            constexpr size_t fnv_prime = 1099511628211ull;
            const auto* bytes = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < size; ++i) {
                seed ^= static_cast<size_t>(bytes[i]);
                seed *= fnv_prime;
            }
            return seed;
        }

        template <typename T>
        [[nodiscard]] size_t hashValue(const T& value, const size_t seed) {
            return hashBytes(&value, sizeof(T), seed);
        }

        [[nodiscard]] size_t hashModelUpload(
            const lfs::core::SplatData& model,
            const std::vector<glm::mat4>* const transforms) {
            size_t hash = 1469598103934665603ull;
            const size_t count = static_cast<size_t>(model.size());
            const int max_sh_degree = model.get_max_sh_degree();

            hash = hashValue(&model, hash);
            hash = hashValue(count, hash);
            hash = hashValue(max_sh_degree, hash);
            hash = hashValue(model.means_raw().storage_ptr(), hash);
            hash = hashValue(model.scaling_raw().storage_ptr(), hash);
            hash = hashValue(model.rotation_raw().storage_ptr(), hash);
            hash = hashValue(model.opacity_raw().storage_ptr(), hash);
            hash = hashValue(model.sh0_raw().storage_ptr(), hash);
            hash = hashValue(model.shN_raw().storage_ptr(), hash);

            const size_t transform_count = transforms ? transforms->size() : 0;
            hash = hashValue(transform_count, hash);
            if (transform_count == 1) {
                hash = hashBytes(&(*transforms)[0], sizeof(glm::mat4), hash);
            }
            return hash;
        }

        [[nodiscard]] glm::mat4 cameraWorldViewMatrix(const lfs::core::Camera& camera) {
            const auto w2c_cpu = camera.world_view_transform().squeeze(0).cpu().contiguous();
            const float* const w2c = w2c_cpu.ptr<float>();
            glm::mat4 world_view{1.0f};
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    world_view[col][row] = w2c[row * 4 + col];
                }
            }
            return world_view;
        }

#if defined(LFS_HAS_VKSPLAT) && LFS_HAS_VKSPLAT
        struct VkSplatContext {
            std::mutex mutex;
            VulkanGSRenderer renderer;
            VulkanGSPipelineBuffers buffers;
            bool initialized = false;
            bool disabled = false;
            size_t upload_hash = 0;
        };

        [[nodiscard]] VkSplatContext& context() {
            static VkSplatContext instance;
            return instance;
        }

        [[nodiscard]] std::filesystem::path shaderRoot() {
#if defined(LFS_VKSPLAT_SHADER_PATH)
            return std::filesystem::path{LFS_VKSPLAT_SHADER_PATH};
#else
            return {};
#endif
        }

        [[nodiscard]] Result<void> initialize(VkSplatContext& ctx) {
            if (ctx.initialized) {
                return {};
            }
            if (ctx.disabled) {
                return std::unexpected("vksplat was disabled after an earlier initialization failure");
            }

            const std::filesystem::path root = shaderRoot();
            if (root.empty() || !std::filesystem::exists(root / "generated" / "projection_forward.spv")) {
                ctx.disabled = true;
                return std::unexpected("vksplat SPIR-V shaders were not found");
            }

            const std::array<std::string, 17> shader_names{
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
                "sum"};

            std::map<std::string, std::string> spirv_paths;
            for (const auto& name : shader_names) {
                const std::filesystem::path path =
                    name.starts_with("radix_sort/")
                        ? root / std::filesystem::path{name + ".spv"}
                        : root / "generated" / std::filesystem::path{name + ".spv"};
                spirv_paths[name] = path.generic_string();
            }
            spirv_paths["where"] = (root / "generated" / "where.spv").generic_string();

            try {
                ctx.renderer.initialize(spirv_paths, 0);
                ctx.initialized = true;
                LOG_INFO("Initialized vksplat Vulkan rasterizer");
                return {};
            } catch (const std::exception& e) {
                ctx.disabled = true;
                return std::unexpected(std::format("vksplat initialization failed: {}", e.what()));
            }
        }

        [[nodiscard]] Result<void> uploadModelIfNeeded(
            VkSplatContext& ctx,
            const lfs::core::SplatData& model,
            const VkSplatRasterizeRequest& request) {
            const auto* const transforms = request.model_transforms;
            if (transforms && transforms->size() > 1) {
                return std::unexpected("vksplat supports only a single model transform");
            }

            const size_t upload_hash = hashModelUpload(model, transforms);
            if (ctx.upload_hash == upload_hash && ctx.buffers.num_splats == static_cast<size_t>(model.size())) {
                return {};
            }

            const size_t count = static_cast<size_t>(model.size());
            if (count == 0) {
                return std::unexpected("vksplat cannot render an empty model");
            }

            auto means_cpu = model.means_raw().cpu().contiguous();
            auto scales_cpu = model.get_scaling().cpu().contiguous();
            auto rotations_cpu = model.get_rotation().cpu().contiguous();
            auto opacities_cpu = model.get_opacity().cpu().contiguous();
            auto sh_cpu = model.get_shs().cpu().contiguous();

            if (means_cpu.ndim() != 2 || means_cpu.size(1) != 3 ||
                scales_cpu.ndim() != 2 || scales_cpu.size(1) != 3 ||
                rotations_cpu.ndim() != 2 || rotations_cpu.size(1) != 4 ||
                sh_cpu.ndim() != 3 || sh_cpu.size(2) != 3) {
                return std::unexpected("vksplat model tensors have unsupported shapes");
            }
            if (means_cpu.size(0) != count || scales_cpu.size(0) != count ||
                rotations_cpu.size(0) != count || opacities_cpu.numel() != count || sh_cpu.size(0) != count) {
                return std::unexpected("vksplat model tensor counts do not match");
            }

            const float* const means = means_cpu.ptr<float>();
            const float* const scales = scales_cpu.ptr<float>();
            const float* const rotations = rotations_cpu.ptr<float>();
            const float* const opacities = opacities_cpu.ptr<float>();
            const float* const sh = sh_cpu.ptr<float>();
            if (!means || !scales || !rotations || !opacities || !sh) {
                return std::unexpected("vksplat model tensor data is not available");
            }

            std::vector<float> sh_coeffs(count * kVkSplatShCoefficients * kVkSplatShChannels, 0.0f);
            const size_t sh_coeffs_in = std::min(sh_cpu.size(1), kVkSplatShCoefficients);
            for (size_t i = 0; i < count; ++i) {
                for (size_t coeff = 0; coeff < sh_coeffs_in; ++coeff) {
                    for (size_t channel = 0; channel < kVkSplatShChannels; ++channel) {
                        sh_coeffs[(i * kVkSplatShCoefficients + coeff) * kVkSplatShChannels + channel] =
                            sh[(i * sh_cpu.size(1) + coeff) * kVkSplatShChannels + channel];
                    }
                }
            }

            ctx.buffers.num_splats = count;
            ctx.buffers.xyz_ws.assign(means, means + count * 3);
            ctx.buffers.rotations.assign(rotations, rotations + count * 4);
            ctx.buffers.assignScalesOpacs(ctx.buffers.scales_opacs, count, scales, opacities);
            ctx.buffers.sh_coeffs.assign(sh_coeffs.begin(), sh_coeffs.end());
            ctx.buffers.reorderSH(ctx.buffers.sh_coeffs);

            ctx.renderer.copyToDevice(ctx.buffers.xyz_ws);
            ctx.renderer.copyToDevice(ctx.buffers.sh_coeffs);
            ctx.renderer.copyToDevice(ctx.buffers.rotations);
            ctx.renderer.copyToDevice(ctx.buffers.scales_opacs);

            ctx.upload_hash = upload_hash;
            LOG_INFO("Uploaded {} gaussians to vksplat", count);
            return {};
        }

        [[nodiscard]] VulkanGSRendererUniforms makeUniforms(
            const lfs::core::Camera& camera,
            const lfs::core::SplatData& model,
            const VkSplatRasterizeRequest& request) {
            VulkanGSRendererUniforms uniforms{};
            uniforms.image_height = static_cast<uint32_t>(camera.camera_height());
            uniforms.image_width = static_cast<uint32_t>(camera.camera_width());
            uniforms.grid_height = (uniforms.image_height + 15u) / 16u;
            uniforms.grid_width = (uniforms.image_width + 15u) / 16u;
            uniforms.num_splats = static_cast<uint32_t>(model.size());
            uniforms.active_sh = static_cast<uint32_t>(std::clamp(request.sh_degree, 0, std::min(3, model.get_max_sh_degree())));
            uniforms.step = 0;
            uniforms.camera_model = 0;
            uniforms.fx = camera.focal_x();
            uniforms.fy = camera.focal_y();
            uniforms.cx = camera.center_x();
            uniforms.cy = camera.center_y();

            glm::mat4 world_view = cameraWorldViewMatrix(camera);
            if (request.model_transforms && request.model_transforms->size() == 1) {
                world_view = world_view * (*request.model_transforms)[0];
            }

            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    uniforms.world_view_transform[col * 4 + row] = world_view[col][row];
                }
            }
            return uniforms;
        }

        [[nodiscard]] Tensor composeImage(
            const Buffer<float>& pixel_state,
            const uint32_t width,
            const uint32_t height,
            const glm::vec3& background,
            const bool transparent_background) {
            const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
            const size_t channels = transparent_background ? 4u : 3u;
            std::vector<float> image(pixel_count * channels, 0.0f);

            for (size_t i = 0; i < pixel_count; ++i) {
                const float r = pixel_state[i * 4 + 0];
                const float g = pixel_state[i * 4 + 1];
                const float b = pixel_state[i * 4 + 2];
                const float transmittance = std::clamp(pixel_state[i * 4 + 3], 0.0f, 1.0f);

                if (transparent_background) {
                    const float alpha = 1.0f - transmittance;
                    const float inv_alpha = alpha > 1.0e-6f ? 1.0f / alpha : 0.0f;
                    image[i * 4 + 0] = r * inv_alpha;
                    image[i * 4 + 1] = g * inv_alpha;
                    image[i * 4 + 2] = b * inv_alpha;
                    image[i * 4 + 3] = alpha;
                } else {
                    image[i * 3 + 0] = r + transmittance * background.r;
                    image[i * 3 + 1] = g + transmittance * background.g;
                    image[i * 3 + 2] = b + transmittance * background.b;
                }
            }

            return Tensor::from_vector(
                image,
                {static_cast<size_t>(height), static_cast<size_t>(width), channels},
                lfs::core::Device::CPU);
        }
#endif
    } // namespace

    bool vksplat_is_available() {
#if defined(LFS_HAS_VKSPLAT) && LFS_HAS_VKSPLAT
        return !context().disabled;
#else
        return false;
#endif
    }

    Result<VkSplatRasterizeOutput> vksplat_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const VkSplatRasterizeRequest& request) {
#if defined(LFS_HAS_VKSPLAT) && LFS_HAS_VKSPLAT
        auto& ctx = context();
        std::lock_guard lock(ctx.mutex);

        if (auto init = initialize(ctx); !init) {
            return std::unexpected(init.error());
        }
        if (auto upload = uploadModelIfNeeded(ctx, model, request); !upload) {
            return std::unexpected(upload.error());
        }

        try {
            auto uniforms = makeUniforms(camera, model, request);
            ctx.renderer.executeProjectionForward(uniforms, ctx.buffers);
            ctx.renderer.executeCalculateIndexBufferOffset(ctx.buffers);
            if (ctx.buffers.num_indices > 0) {
                ctx.renderer.executeGenerateKeys(uniforms, ctx.buffers);
                ctx.renderer.executeSort(uniforms, ctx.buffers, -1);
                ctx.renderer.executeComputeTileRanges(uniforms, ctx.buffers);
                ctx.renderer.executeRasterizeForward(uniforms, ctx.buffers);
                ctx.renderer.copyFromDevice(ctx.buffers.pixel_state);
            } else {
                ctx.buffers.pixel_state.assign(
                    static_cast<size_t>(uniforms.image_width) * static_cast<size_t>(uniforms.image_height) * 4u,
                    0.0f);
                for (size_t i = 0; i < ctx.buffers.pixel_state.size() / 4u; ++i) {
                    ctx.buffers.pixel_state[i * 4u + 3u] = 1.0f;
                }
            }

            return VkSplatRasterizeOutput{
                .image = composeImage(
                    ctx.buffers.pixel_state,
                    uniforms.image_width,
                    uniforms.image_height,
                    request.background_color,
                    request.transparent_background)};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("vksplat rasterization failed: {}", e.what()));
        }
#else
        (void)camera;
        (void)model;
        (void)request;
        return std::unexpected("vksplat is not enabled in this build");
#endif
    }

} // namespace lfs::rendering
