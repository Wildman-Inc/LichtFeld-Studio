/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cuda_bf16.h
 * @brief HIP-compatible CUDA bfloat16 shim header.
 */

#ifndef _LFS_CUDA_BF16_SHIM_H_
#define _LFS_CUDA_BF16_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    #include <hip/hip_bfloat16.h>

    // Type alias
    using nv_bfloat16 = hip_bfloat16;
    using __nv_bfloat16 = hip_bfloat16;
    // Note: hip_bfloat162 doesn't exist, use a pair struct if needed
#else
    #include <cuda_bf16.h>
#endif

#endif // _LFS_CUDA_BF16_SHIM_H_
