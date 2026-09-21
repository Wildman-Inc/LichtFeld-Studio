/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2025 Youyu Chen (original Lanczos implementation)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-License-Identifier: MIT (original Lanczos implementation)
 */

#pragma once

#include "core/tensor.hpp"

namespace lfs::core {

    // A single caller owns this workspace. Calls complete before returning, so
    // dimensions/streams may change between calls. Destroy on the same device.
    class LanczosResizeWorkspace {
        friend void lanczos_resize_into(const Tensor&, Tensor&, LanczosResizeWorkspace&,
                                        int, cudaStream_t);
        Tensor coefficients_x_, coefficients_y_;
        int input_h_ = 0, input_w_ = 0, output_h_ = 0, output_w_ = 0, kernel_size_ = 0;
    };

    // RGB uint8 HWC -> preallocated, exclusively owned uint8/float32 CHW.
    // Reuses coefficients and quantizes only after the full Lanczos sum.
    void lanczos_resize_into(const Tensor& input, Tensor& output,
                             LanczosResizeWorkspace& workspace,
                             int kernel_size = 2, cudaStream_t cuda_stream = nullptr);

    /**
     * High-quality Lanczos resampling on GPU
     *
     * @param input Input tensor in [H, W, C] format (uint8)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [C, H, W] format (float32)
     */
    Tensor lanczos_resize(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

    /**
     * High-quality Lanczos resampling for grayscale images on GPU
     *
     * @param input Input tensor in [H, W] format (uint8, normalized to [0,1], or float32 passed through)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [H, W] format (float32)
     */
    Tensor lanczos_resize_grayscale(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

    /**
     * High-quality Lanczos resampling for planar 3-channel float images on GPU
     *
     * @param input Input tensor in [C, H, W] format (float32, 3 channels)
     * @param output_h Target height
     * @param output_w Target width
     * @param kernel_size Lanczos kernel size (typically 2 or 3)
     * @param cuda_stream CUDA stream for async execution
     * @return Resized tensor in [C, H, W] format (float32)
     */
    Tensor lanczos_resize_float_chw(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size = 2,
        cudaStream_t cuda_stream = nullptr);

    // Bilinear prior resampling excludes invalid neighbors and carries validity
    // with nearest sampling. Depth <= 0/nonfinite and normal norms < 0.5 are
    // invalid; outputs use zero sentinels and valid normals have unit length.
    Tensor resize_depth_prior(const Tensor& input, int output_h, int output_w,
                              cudaStream_t cuda_stream = nullptr);
    Tensor resize_normal_prior(const Tensor& input, int output_h, int output_w,
                               cudaStream_t cuda_stream = nullptr);

} // namespace lfs::core
