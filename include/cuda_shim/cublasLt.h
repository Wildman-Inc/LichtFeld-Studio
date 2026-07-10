/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cublasLt.h
 * @brief HIP-compatible cuBLASLt shim header.
 *
 * cuBLASLt is CUDA's lightweight BLAS library. For HIP, we include hipblaslt
 * if available, otherwise provide empty stubs for symbols used by PyTorch.
 */

#ifndef _LFS_CUBLASLT_SHIM_H_
#define _LFS_CUBLASLT_SHIM_H_

#if defined(USE_HIP) && USE_HIP

// hipBLASLt is the HIP equivalent of cuBLASLt
// It may not be available on all ROCm installations
#if __has_include(<hipblaslt/hipblaslt.h>)
    #include <hipblaslt/hipblaslt.h>
#else
    // Minimal stub definitions for PyTorch compatibility
    // These won't actually work but allow compilation
    typedef void* cublasLtHandle_t;
    typedef void* cublasLtMatmulDesc_t;
    typedef void* cublasLtMatrixLayout_t;
    typedef void* cublasLtMatmulPreference_t;

    typedef enum {
        CUBLASLT_STATUS_SUCCESS = 0,
        CUBLASLT_STATUS_NOT_INITIALIZED = 1,
        CUBLASLT_STATUS_ALLOC_FAILED = 2,
        CUBLASLT_STATUS_INVALID_VALUE = 3,
        CUBLASLT_STATUS_ARCH_MISMATCH = 4,
        CUBLASLT_STATUS_MAPPING_ERROR = 5,
        CUBLASLT_STATUS_EXECUTION_FAILED = 6,
        CUBLASLT_STATUS_INTERNAL_ERROR = 7,
        CUBLASLT_STATUS_NOT_SUPPORTED = 8,
    } cublasLtStatus_t;
#endif

#else
    // CUDA path - include the real header
    #include <cublasLt.h>
#endif

#endif // _LFS_CUBLASLT_SHIM_H_
