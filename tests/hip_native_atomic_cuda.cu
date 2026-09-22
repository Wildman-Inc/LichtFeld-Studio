/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/rasterization/fastgs/rasterization/include/device_gradient_atomic.cuh"

namespace {
    template <bool Native>
    __global__ void add_gradient_probe(float* sum, int count, float value, bool alternate_sign) {
        const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index < count) {
            const float increment = alternate_sign && (index & 1) ? -value : value;
            fast_lfs::rasterization::kernels::atomic_add_device_gradient<Native>(sum, increment);
        }
    }
} // namespace

cudaError_t launch_native_gradient_atomic_probe(float* sums, int count, float value,
                                                bool alternate_sign, cudaStream_t stream) {
    const int blocks = (count + 255) / 256;
    add_gradient_probe<false><<<blocks, 256, 0, stream>>>(sums, count, value, alternate_sign);
    const auto first = cudaGetLastError();
    if (first != cudaSuccess) {
        return first;
    }
    add_gradient_probe<true><<<blocks, 256, 0, stream>>>(sums + 1, count, value, alternate_sign);
    return cudaGetLastError();
}
