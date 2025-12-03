/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file curand_kernel.h
 * @brief HIP-compatible cuRAND kernel shim header.
 */

#ifndef _LFS_CURAND_KERNEL_SHIM_H_
#define _LFS_CURAND_KERNEL_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    #include "kernels/hip_rand_compat.h"
    #include <hiprand/hiprand_kernel.h>
#else
    #include <curand_kernel.h>
#endif

#endif // _LFS_CURAND_KERNEL_SHIM_H_
