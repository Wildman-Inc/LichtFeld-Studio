/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file curand.h
 * @brief HIP-compatible cuRAND shim header.
 */

#ifndef _LFS_CURAND_SHIM_H_
#define _LFS_CURAND_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    #include "kernels/hip_rand_compat.h"
#else
    #include <curand.h>
#endif

#endif // _LFS_CURAND_SHIM_H_
