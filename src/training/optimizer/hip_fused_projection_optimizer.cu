/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hip_fused_projection_optimizer.hpp"

#include "buffer_utils.h"
#include "kernels_backward.cuh"
#include "utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace lfs::training::optimizer {

    namespace {
        constexpr int BLOCK_SIZE = 256;

        void check_cuda(const cudaError_t err, const char* label) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(err));
            }
        }

        __device__ inline float sigmoid(const float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ inline void adam_update(
            float& param,
            float& exp_avg,
            float& exp_avg_sq,
            const float grad,
            const HipFusedAdamParam& adam,
            const HipFusedProjectionBackwardOptimizerConfig& config) {
            const float moment1 = config.beta1 * exp_avg + (1.0f - config.beta1) * grad;
            const float moment2 = config.beta2 * exp_avg_sq + (1.0f - config.beta2) * grad * grad;
            const float denom = sqrtf(moment2) * adam.bias_correction2_sqrt_rcp + config.eps;
            const float step_size = adam.lr * adam.bias_correction1_rcp;

            param -= step_size * moment1 / denom;
            exp_avg = moment1;
            exp_avg_sq = moment2;
        }

        __device__ inline void adam_update_at(
            const HipFusedAdamParam& adam,
            const int64_t idx,
            const float grad,
            const HipFusedProjectionBackwardOptimizerConfig& config) {
            if (!adam.param || !adam.exp_avg || !adam.exp_avg_sq || idx < 0 || idx >= adam.n_elements) {
                return;
            }
            adam_update(adam.param[idx], adam.exp_avg[idx], adam.exp_avg_sq[idx], grad, adam, config);
        }

        __global__ void fused_projection_backward_optimizer_kernel(
            HipFusedProjectionBackwardOptimizerConfig config,
            const uint* __restrict__ primitive_n_touched_tiles,
            const float2* __restrict__ grad_mean2d,
            const float* __restrict__ grad_conic) {
            using namespace fast_lfs::rasterization;
            using namespace fast_lfs::rasterization::kernels;
            using namespace fast_lfs::rasterization::kernels::backward;

            const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (primitive_idx >= static_cast<uint>(config.n_primitives)) {
                return;
            }

            const bool touched = primitive_n_touched_tiles[primitive_idx] != 0;
            const float4* const w2c = reinterpret_cast<const float4*>(config.w2c);
            const float3* const cam_position = reinterpret_cast<const float3*>(config.cam_position);
            float3* const means = reinterpret_cast<float3*>(config.means.param);
            float3* const raw_scales = reinterpret_cast<float3*>(config.scaling.param);
            float4* const raw_rotations = reinterpret_cast<float4*>(config.rotation.param);
            float* const raw_opacities = config.opacity.param;
            float3* const sh0 = reinterpret_cast<float3*>(config.sh0.param);
            float3* const shN = reinterpret_cast<float3*>(config.shN.param);
            const float3* const shN_const = reinterpret_cast<const float3*>(config.shN.param);

            const float3 mean3d = means[primitive_idx];
            float3 dL_dmean3d = make_float3(0.0f);
            float3 dL_draw_scale = make_float3(0.0f);
            float4 dL_draw_rotation = make_float4(0.0f);

            if (touched) {
                const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
                    shN_const,
                    reinterpret_cast<float3*>(config.grad_sh0),
                    reinterpret_cast<float3*>(config.grad_shN),
                    mean3d,
                    cam_position[0],
                    primitive_idx,
                    static_cast<uint>(config.active_sh_bases),
                    static_cast<uint>(config.total_bases_sh_rest));

                const float4 w2c_r3 = w2c[2];
                const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
                const float depth_safe = fmaxf(depth, 1e-4f);
                const float4 w2c_r1 = w2c[0];
                const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth_safe;
                const float4 w2c_r2 = w2c[1];
                const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth_safe;

                const float3 raw_scale = raw_scales[primitive_idx];
                const float3 clamped_scale = make_float3(
                    fminf(raw_scale.x, fast_lfs::rasterization::config::max_raw_scale),
                    fminf(raw_scale.y, fast_lfs::rasterization::config::max_raw_scale),
                    fminf(raw_scale.z, fast_lfs::rasterization::config::max_raw_scale));
                const float3 variance = make_float3(
                    expf(2.0f * clamped_scale.x),
                    expf(2.0f * clamped_scale.y),
                    expf(2.0f * clamped_scale.z));
                auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
                const float qrr_raw = qr * qr;
                const float qxx_raw = qx * qx;
                const float qyy_raw = qy * qy;
                const float qzz_raw = qz * qz;
                const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
                const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-7f);
                const float qxx = 2.0f * qxx_raw / q_norm_sq_safe;
                const float qyy = 2.0f * qyy_raw / q_norm_sq_safe;
                const float qzz = 2.0f * qzz_raw / q_norm_sq_safe;
                const float qxy = 2.0f * qx * qy / q_norm_sq_safe;
                const float qxz = 2.0f * qx * qz / q_norm_sq_safe;
                const float qyz = 2.0f * qy * qz / q_norm_sq_safe;
                const float qrx = 2.0f * qr * qx / q_norm_sq_safe;
                const float qry = 2.0f * qr * qy / q_norm_sq_safe;
                const float qrz = 2.0f * qr * qz / q_norm_sq_safe;
                const mat3x3 rotation = {
                    1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
                    qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
                    qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
                const mat3x3 rotation_scaled = {
                    rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
                    rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
                    rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
                const mat3x3_triu cov3d{
                    rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
                    rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
                    rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
                    rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
                    rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
                    rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
                };

                const float w = static_cast<float>(config.width);
                const float h = static_cast<float>(config.height);
                const float clip_left = (-0.15f * w - config.center_x) / config.focal_x;
                const float clip_right = (1.15f * w - config.center_x) / config.focal_x;
                const float clip_top = (-0.15f * h - config.center_y) / config.focal_y;
                const float clip_bottom = (1.15f * h - config.center_y) / config.focal_y;
                const float tx = clamp(x, clip_left, clip_right);
                const float ty = clamp(y, clip_top, clip_bottom);
                const float j11 = config.focal_x / depth_safe;
                const float j13 = -j11 * tx;
                const float j22 = config.focal_y / depth_safe;
                const float j23 = -j22 * ty;
                const float3 jw_r1 = make_float3(
                    j11 * w2c_r1.x + j13 * w2c_r3.x,
                    j11 * w2c_r1.y + j13 * w2c_r3.y,
                    j11 * w2c_r1.z + j13 * w2c_r3.z);
                const float3 jw_r2 = make_float3(
                    j22 * w2c_r2.x + j23 * w2c_r3.x,
                    j22 * w2c_r2.y + j23 * w2c_r3.y,
                    j22 * w2c_r2.z + j23 * w2c_r3.z);
                const float3 jwc_r1 = make_float3(
                    jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
                    jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
                    jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
                const float3 jwc_r2 = make_float3(
                    jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
                    jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
                    jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);

                const float kernel_size = config.mip_filter
                                              ? fast_lfs::rasterization::config::dilation_mip_filter
                                              : fast_lfs::rasterization::config::dilation;
                const float a = dot(jwc_r1, jw_r1) + kernel_size;
                const float b = dot(jwc_r1, jw_r2);
                const float c = dot(jwc_r2, jw_r2) + kernel_size;
                const float aa = a * a;
                const float bb = b * b;
                const float cc = c * c;
                const float ac = a * c;
                const float ab = a * b;
                const float bc = b * c;
                const float determinant = ac - bb;
                const float determinant_safe = fmaxf(
                    determinant,
                    fast_lfs::rasterization::config::min_cov2d_determinant);
                const float determinant_rcp = 1.0f / determinant_safe;
                const float determinant_rcp_sq = determinant_rcp * determinant_rcp;
                const float3 dL_dconic = make_float3(
                    grad_conic[primitive_idx],
                    grad_conic[config.n_primitives + primitive_idx],
                    grad_conic[2 * config.n_primitives + primitive_idx]);
                const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                    2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                    bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
                    2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

                const mat3x3_triu dL_dcov3d = {
                    (jw_r1.x * jw_r1.x) * dL_dcov2d.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.x) * dL_dcov2d.z,
                    (jw_r1.x * jw_r1.y) * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.y) * dL_dcov2d.z,
                    (jw_r1.x * jw_r1.z) * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.z) * dL_dcov2d.z,
                    (jw_r1.y * jw_r1.y) * dL_dcov2d.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.y) * dL_dcov2d.z,
                    (jw_r1.y * jw_r1.z) * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.z) * dL_dcov2d.z,
                    (jw_r1.z * jw_r1.z) * dL_dcov2d.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d.y + (jw_r2.z * jw_r2.z) * dL_dcov2d.z,
                };

                const float3 dL_djw_r1 = 2.0f * make_float3(
                                                   jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                                                   jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                                                   jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
                const float3 dL_djw_r2 = 2.0f * make_float3(
                                                   jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                                                   jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                                                   jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);

                const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
                const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
                const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
                const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

                const float2 dL_dmean2d = grad_mean2d[primitive_idx];
                float3 dL_dmean3d_cam = make_float3(
                    j11 * dL_dmean2d.x,
                    j22 * dL_dmean2d.y,
                    -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y);
                const bool valid_x = x >= clip_left && x <= clip_right;
                const bool valid_y = y >= clip_top && y <= clip_bottom;
                if (valid_x) {
                    dL_dmean3d_cam.x -= j11 * dL_dj13 / depth_safe;
                }
                if (valid_y) {
                    dL_dmean3d_cam.y -= j22 * dL_dj23 / depth_safe;
                }
                const float factor_x = 1.0f + static_cast<float>(valid_x);
                const float factor_y = 1.0f + static_cast<float>(valid_y);
                dL_dmean3d_cam.z +=
                    (j11 * (factor_x * tx * dL_dj13 - dL_dj11) +
                     j22 * (factor_y * ty * dL_dj23 - dL_dj22)) /
                    depth_safe;

                const float3 dL_dmean3d_from_splatting = make_float3(
                    w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
                    w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
                    w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

                dL_dmean3d = clamp_grad3(dL_dmean3d_from_splatting + dL_dmean3d_from_color);

                const float dL_dvariance_x =
                    rotation.m11 * rotation.m11 * dL_dcov3d.m11 +
                    rotation.m21 * rotation.m21 * dL_dcov3d.m22 +
                    rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                    2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 +
                            rotation.m11 * rotation.m31 * dL_dcov3d.m13 +
                            rotation.m21 * rotation.m31 * dL_dcov3d.m23);
                const float dL_dvariance_y =
                    rotation.m12 * rotation.m12 * dL_dcov3d.m11 +
                    rotation.m22 * rotation.m22 * dL_dcov3d.m22 +
                    rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                    2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 +
                            rotation.m12 * rotation.m32 * dL_dcov3d.m13 +
                            rotation.m22 * rotation.m32 * dL_dcov3d.m23);
                const float dL_dvariance_z =
                    rotation.m13 * rotation.m13 * dL_dcov3d.m11 +
                    rotation.m23 * rotation.m23 * dL_dcov3d.m22 +
                    rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                    2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 +
                            rotation.m13 * rotation.m33 * dL_dcov3d.m13 +
                            rotation.m23 * rotation.m33 * dL_dcov3d.m23);
                dL_draw_scale = clamp_grad3(make_float3(
                    (raw_scale.x < fast_lfs::rasterization::config::max_raw_scale) ? 2.0f * variance.x * dL_dvariance_x : 0.0f,
                    (raw_scale.y < fast_lfs::rasterization::config::max_raw_scale) ? 2.0f * variance.y * dL_dvariance_y : 0.0f,
                    (raw_scale.z < fast_lfs::rasterization::config::max_raw_scale) ? 2.0f * variance.z * dL_dvariance_z : 0.0f));

                const mat3x3 dL_drotation = {
                    2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
                    2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
                    2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
                    2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
                    2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
                    2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
                    2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
                    2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
                    2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};
                const float dL_dqxx = -dL_drotation.m22 - dL_drotation.m33;
                const float dL_dqyy = -dL_drotation.m11 - dL_drotation.m33;
                const float dL_dqzz = -dL_drotation.m11 - dL_drotation.m22;
                const float dL_dqxy = dL_drotation.m12 + dL_drotation.m21;
                const float dL_dqxz = dL_drotation.m13 + dL_drotation.m31;
                const float dL_dqyz = dL_drotation.m23 + dL_drotation.m32;
                const float dL_dqrx = dL_drotation.m32 - dL_drotation.m23;
                const float dL_dqry = dL_drotation.m13 - dL_drotation.m31;
                const float dL_dqrz = dL_drotation.m21 - dL_drotation.m12;
                const float dL_dq_norm_helper =
                    qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz +
                    qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz +
                    qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
                dL_draw_rotation = clamp_grad4(
                    2.0f *
                    make_float4(
                        qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper,
                        2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper,
                        2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper,
                        2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) /
                    q_norm_sq_safe);

                if (config.densification_info &&
                    config.densification_error_map == nullptr &&
                    config.densification_type == DensificationType::None) {
                    config.densification_info[primitive_idx] += 1.0f;
                    config.densification_info[config.n_primitives + primitive_idx] +=
                        length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
                }
            }

            const int mean_base = static_cast<int>(primitive_idx) * 3;
            adam_update_at(config.means, mean_base + 0, dL_dmean3d.x, config);
            adam_update_at(config.means, mean_base + 1, dL_dmean3d.y, config);
            adam_update_at(config.means, mean_base + 2, dL_dmean3d.z, config);

            const float3 raw_scale_for_reg = raw_scales[primitive_idx];
            const float scale_reg_grad_scale =
                config.scale_reg_weight > 0.0f
                    ? config.scale_reg_weight / static_cast<float>(config.n_primitives * 3)
                    : 0.0f;
            adam_update_at(
                config.scaling,
                mean_base + 0,
                dL_draw_scale.x + scale_reg_grad_scale * expf(raw_scale_for_reg.x),
                config);
            adam_update_at(
                config.scaling,
                mean_base + 1,
                dL_draw_scale.y + scale_reg_grad_scale * expf(raw_scale_for_reg.y),
                config);
            adam_update_at(
                config.scaling,
                mean_base + 2,
                dL_draw_scale.z + scale_reg_grad_scale * expf(raw_scale_for_reg.z),
                config);

            const int rotation_base = static_cast<int>(primitive_idx) * 4;
            adam_update_at(config.rotation, rotation_base + 0, dL_draw_rotation.x, config);
            adam_update_at(config.rotation, rotation_base + 1, dL_draw_rotation.y, config);
            adam_update_at(config.rotation, rotation_base + 2, dL_draw_rotation.z, config);
            adam_update_at(config.rotation, rotation_base + 3, dL_draw_rotation.w, config);

            const float raw_opacity = raw_opacities[primitive_idx];
            const float opacity = sigmoid(raw_opacity);
            const float opacity_reg_grad =
                config.opacity_reg_weight > 0.0f
                    ? (config.opacity_reg_weight / static_cast<float>(config.n_primitives)) *
                          opacity * (1.0f - opacity)
                    : 0.0f;
            const float dL_draw_opacity =
                (config.grad_opacity ? config.grad_opacity[primitive_idx] : 0.0f) + opacity_reg_grad;
            adam_update_at(config.opacity, primitive_idx, dL_draw_opacity, config);

            const float3 sh0_grad =
                config.grad_sh0 ? reinterpret_cast<float3*>(config.grad_sh0)[primitive_idx] : make_float3(0.0f);
            adam_update_at(config.sh0, mean_base + 0, sh0_grad.x, config);
            adam_update_at(config.sh0, mean_base + 1, sh0_grad.y, config);
            adam_update_at(config.sh0, mean_base + 2, sh0_grad.z, config);

            if (config.update_shN && shN && config.grad_shN && config.total_bases_sh_rest > 0) {
                const int shn_base = static_cast<int>(primitive_idx) * config.total_bases_sh_rest * 3;
                const int shn_count = config.total_bases_sh_rest * 3;
                for (int i = 0; i < shn_count; ++i) {
                    adam_update_at(config.shN, shn_base + i, config.grad_shN[shn_base + i], config);
                }
            }
        }

        template <DensificationType DENSIFICATION_TYPE>
        void launch_blend_backward(
            const HipFusedProjectionBackwardOptimizerConfig& config,
            fast_lfs::rasterization::PerTileBuffers& per_tile_buffers,
            fast_lfs::rasterization::PerInstanceBuffers& per_instance_buffers,
            fast_lfs::rasterization::PerPrimitiveBuffers& per_primitive_buffers,
            fast_lfs::rasterization::PerBucketBuffers& per_bucket_buffers,
            float2* grad_mean2d_helper,
            float* grad_conic_helper,
            cudaStream_t stream) {
            fast_lfs::rasterization::kernels::backward::blend_backward_cu<DENSIFICATION_TYPE>
                <<<config.forward_ctx.n_buckets, 32, 0, stream>>>(
                    per_tile_buffers.instance_ranges,
                    per_tile_buffers.bucket_offsets,
                    per_instance_buffers.primitive_indices.Current(),
                    per_primitive_buffers.mean2d,
                    per_primitive_buffers.conic_opacity,
                    per_primitive_buffers.color,
                    config.opacity.param,
                    config.grad_image,
                    config.grad_alpha,
                    config.image,
                    config.alpha,
                    per_tile_buffers.max_n_contributions,
                    per_tile_buffers.n_contributions,
                    per_bucket_buffers.tile_index,
                    per_bucket_buffers.checkpoint_uint8,
                    grad_mean2d_helper,
                    grad_conic_helper,
                    config.grad_opacity,
                    reinterpret_cast<float3*>(config.grad_sh0),
                    config.densification_info,
                    config.densification_error_map,
                    config.forward_ctx.n_buckets,
                    config.n_primitives,
                    config.width,
                    config.height,
                    div_round_up(config.width, fast_lfs::rasterization::config::tile_width),
                    config.mip_filter);
        }
    } // namespace

    void launch_hip_fused_projection_backward_optimizer(
        const HipFusedProjectionBackwardOptimizerConfig& config,
        cudaStream_t stream) {
        using namespace fast_lfs::rasterization;

        if (config.n_primitives <= 0) {
            return;
        }
        if (!config.forward_ctx.per_primitive_buffers || !config.forward_ctx.per_tile_buffers ||
            !config.forward_ctx.grad_mean2d_helper || !config.forward_ctx.grad_conic_helper) {
            throw std::runtime_error("Invalid fastgs forward context for HIP fused projection optimizer");
        }

        const dim3 grid(
            div_round_up(config.width, fast_lfs::rasterization::config::tile_width),
            div_round_up(config.height, fast_lfs::rasterization::config::tile_height),
            1);
        const int n_tiles = grid.x * grid.y;
        const int end_bit = extract_end_bit(static_cast<uint>(n_tiles - 1));

        char* per_primitive_blob = static_cast<char*>(config.forward_ctx.per_primitive_buffers);
        char* per_tile_blob = static_cast<char*>(config.forward_ctx.per_tile_buffers);

        auto per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_blob, config.n_primitives);
        auto per_tile_buffers = PerTileBuffers::from_blob(per_tile_blob, n_tiles);

        per_primitive_buffers.primitive_indices.selector =
            config.forward_ctx.primitive_primitive_indices_selector;

        auto* grad_mean2d_helper = static_cast<float2*>(config.forward_ctx.grad_mean2d_helper);
        auto* grad_conic_helper = static_cast<float*>(config.forward_ctx.grad_conic_helper);
        check_cuda(
            cudaMemsetAsync(
                grad_mean2d_helper,
                0,
                static_cast<size_t>(config.n_primitives) * 2 * sizeof(float),
                stream),
            "hip fused grad_mean2d memset failed");
        check_cuda(
            cudaMemsetAsync(
                grad_conic_helper,
                0,
                static_cast<size_t>(config.n_primitives) * 3 * sizeof(float),
                stream),
            "hip fused grad_conic memset failed");

        if (config.forward_ctx.n_buckets > 0 &&
            config.forward_ctx.n_instances > 0 &&
            config.forward_ctx.n_visible_primitives > 0) {
            if (!config.forward_ctx.per_instance_buffers || !config.forward_ctx.per_bucket_buffers) {
                throw std::runtime_error("Invalid fastgs instance/bucket context for HIP fused projection optimizer");
            }
            char* per_instance_blob = static_cast<char*>(config.forward_ctx.per_instance_buffers);
            char* per_bucket_blob = static_cast<char*>(config.forward_ctx.per_bucket_buffers);
            auto per_instance_buffers = PerInstanceBuffers::from_blob(
                per_instance_blob,
                config.forward_ctx.n_instances,
                end_bit);
            auto per_bucket_buffers = PerBucketBuffers::from_blob(
                per_bucket_blob,
                config.forward_ctx.n_buckets);
            per_instance_buffers.primitive_indices.selector =
                config.forward_ctx.instance_primitive_indices_selector;

            if (config.densification_type == DensificationType::MRNF && config.densification_info) {
                launch_blend_backward<DensificationType::MRNF>(
                    config,
                    per_tile_buffers,
                    per_instance_buffers,
                    per_primitive_buffers,
                    per_bucket_buffers,
                    grad_mean2d_helper,
                    grad_conic_helper,
                    stream);
            } else if (config.densification_info && config.densification_error_map) {
                launch_blend_backward<DensificationType::MCMC>(
                    config,
                    per_tile_buffers,
                    per_instance_buffers,
                    per_primitive_buffers,
                    per_bucket_buffers,
                    grad_mean2d_helper,
                    grad_conic_helper,
                    stream);
            } else {
                launch_blend_backward<DensificationType::None>(
                    config,
                    per_tile_buffers,
                    per_instance_buffers,
                    per_primitive_buffers,
                    per_bucket_buffers,
                    grad_mean2d_helper,
                    grad_conic_helper,
                    stream);
            }
            check_cuda(cudaGetLastError(), "hip fused blend_backward launch failed");
        }

        const int blocks = div_round_up(config.n_primitives, BLOCK_SIZE);
        fused_projection_backward_optimizer_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            config,
            per_primitive_buffers.n_touched_tiles,
            grad_mean2d_helper,
            grad_conic_helper);
        check_cuda(cudaGetLastError(), "hip fused projection backward optimizer launch failed");
    }

} // namespace lfs::training::optimizer
