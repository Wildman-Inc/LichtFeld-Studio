/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#if defined(LFS_TEST_TRAINING_WARP_REDUCE)
#include "lfs/core/warp_reduce.cuh"
#else
#include "internal/warp_reduce.cuh"
#endif

__global__ void lfs_cdna_wave_smoke(float* output) {
    const float value = static_cast<float>(threadIdx.x + 1);
    const float product_value = 1.0f + static_cast<float>(threadIdx.x & 1) * 0.001f;

    const float warp_sum = lfs::core::warp_ops::warp_reduce_sum(value);
    const float warp_max = lfs::core::warp_ops::warp_reduce_max(value);
    const float warp_min = lfs::core::warp_ops::warp_reduce_min(value);
    const float warp_product = lfs::core::warp_ops::warp_reduce_prod(product_value);

    const float block_sum = lfs::core::warp_ops::block_reduce_sum(value);
    const float block_max = lfs::core::warp_ops::block_reduce_max(value);
    const float block_min = lfs::core::warp_ops::block_reduce_min(value);
    const float block_product = lfs::core::warp_ops::block_reduce_prod(product_value);

    output[threadIdx.x] = warp_sum + warp_max + warp_min + warp_product +
                          block_sum + block_max + block_min + block_product;
}
