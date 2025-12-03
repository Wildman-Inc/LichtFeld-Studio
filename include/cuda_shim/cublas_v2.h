/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cublas_v2.h
 * @brief HIP-compatible cuBLAS shim header.
 */

#ifndef _LFS_CUBLAS_SHIM_H_
#define _LFS_CUBLAS_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    #include "kernels/hip_blas_compat.h"
#else
    #include <cublas_v2.h>
#endif

#endif // _LFS_CUBLAS_SHIM_H_
