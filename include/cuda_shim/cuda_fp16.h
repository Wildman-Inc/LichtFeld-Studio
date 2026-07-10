/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cuda_fp16.h
 * @brief HIP-compatible CUDA FP16 shim header.
 */

#ifndef _LFS_CUDA_FP16_SHIM_H_
#define _LFS_CUDA_FP16_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    #include <hip/hip_fp16.h>
    // HIP already provides __half and half2 types, no need for extra aliases
#else
    #include <cuda_fp16.h>
#endif

#endif // _LFS_CUDA_FP16_SHIM_H_
