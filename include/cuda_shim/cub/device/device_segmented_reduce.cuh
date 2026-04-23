/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cub/device/device_segmented_reduce.cuh
 * @brief HIP-compatible CUB device segmented reduce shim header.
 */

#ifndef _LFS_CUB_DEVICE_SEGMENTED_REDUCE_SHIM_H_
#define _LFS_CUB_DEVICE_SEGMENTED_REDUCE_SHIM_H_

#if (defined(USE_HIP) && USE_HIP) || (defined(LFS_USE_HIP) && LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // hipCUB provides CUB-compatible API
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#else
    #include <cub/device/device_segmented_reduce.cuh>
#endif

#endif // _LFS_CUB_DEVICE_SEGMENTED_REDUCE_SHIM_H_
