/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cusolverDn.h
 * @brief HIP-compatible cuSOLVER Dense shim header.
 */

#ifndef _LFS_CUSOLVER_DN_SHIM_H_
#define _LFS_CUSOLVER_DN_SHIM_H_

#if defined(USE_HIP) && USE_HIP

// HIP solver is the equivalent - include it if available
#if __has_include(<hipsolver/hipsolver.h>)
    #include <hipsolver/hipsolver.h>
#endif

// Type mappings (if not already defined by hipsolver)
#ifndef cusolverDnHandle_t
    typedef void* cusolverDnHandle_t;
#endif

#ifndef cusolverStatus_t
typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT = 10,
    CUSOLVER_STATUS_INVALID_LICENSE = 11,
} cusolverStatus_t;
#endif

#else
    // CUDA path - include the real header
    #include <cusolverDn.h>
#endif

#endif // _LFS_CUSOLVER_DN_SHIM_H_
