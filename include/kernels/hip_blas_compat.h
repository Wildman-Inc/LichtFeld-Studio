/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_blas_compat.h
 * @brief cuBLAS/hipBLAS compatibility layer.
 *
 * This header provides unified BLAS API for both CUDA (cuBLAS)
 * and HIP (hipBLAS) backends.
 */

#ifndef LFS_HIP_BLAS_COMPAT_H
#define LFS_HIP_BLAS_COMPAT_H

#include "kernels/hip_runtime_compat.h"

#if LFS_USE_HIP

#include <hipblas/hipblas.h>

// Type mappings
using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;
using cublasOperation_t = hipblasOperation_t;
using cublasFillMode_t = hipblasFillMode_t;
using cublasDiagType_t = hipblasDiagType_t;
using cublasSideMode_t = hipblasSideMode_t;

// Status codes
constexpr auto CUBLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
constexpr auto CUBLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED;
constexpr auto CUBLAS_STATUS_ALLOC_FAILED = HIPBLAS_STATUS_ALLOC_FAILED;
constexpr auto CUBLAS_STATUS_INVALID_VALUE = HIPBLAS_STATUS_INVALID_VALUE;
constexpr auto CUBLAS_STATUS_ARCH_MISMATCH = HIPBLAS_STATUS_ARCH_MISMATCH;
constexpr auto CUBLAS_STATUS_MAPPING_ERROR = HIPBLAS_STATUS_MAPPING_ERROR;
constexpr auto CUBLAS_STATUS_EXECUTION_FAILED = HIPBLAS_STATUS_EXECUTION_FAILED;
constexpr auto CUBLAS_STATUS_INTERNAL_ERROR = HIPBLAS_STATUS_INTERNAL_ERROR;

// Operation types - use macros to avoid redefinition if already defined
#ifndef CUBLAS_OP_N
#define CUBLAS_OP_N HIPBLAS_OP_N
#endif
#ifndef CUBLAS_OP_T
#define CUBLAS_OP_T HIPBLAS_OP_T
#endif
#ifndef CUBLAS_OP_C
#define CUBLAS_OP_C HIPBLAS_OP_C
#endif

// Fill modes
#ifndef CUBLAS_FILL_MODE_UPPER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#endif
#ifndef CUBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#endif

// Side modes
#ifndef CUBLAS_SIDE_LEFT
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#endif
#ifndef CUBLAS_SIDE_RIGHT
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#endif

// Diag types
#ifndef CUBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#endif
#ifndef CUBLAS_DIAG_UNIT
#define CUBLAS_DIAG_UNIT HIPBLAS_DIAG_UNIT
#endif

// Function mappings - Handle management
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetStream hipblasGetStream
#define cublasSetMathMode hipblasSetMathMode
#define cublasGetMathMode hipblasGetMathMode

// BLAS Level 1
#define cublasScopy hipblasScopy
#define cublasDcopy hipblasDcopy
#define cublasCcopy hipblasCcopy
#define cublasZcopy hipblasZcopy

#define cublasSaxpy hipblasSaxpy
#define cublasDaxpy hipblasDaxpy
#define cublasCaxpy hipblasCaxpy
#define cublasZaxpy hipblasZaxpy

#define cublasSscal hipblasSscal
#define cublasDscal hipblasDscal
#define cublasCscal hipblasCscal
#define cublasZscal hipblasZscal

#define cublasSdot hipblasSdot
#define cublasDdot hipblasDdot

#define cublasSnrm2 hipblasSnrm2
#define cublasDnrm2 hipblasDnrm2

#define cublasSasum hipblasSasum
#define cublasDasum hipblasDasum

#define cublasIsamax hipblasIsamax
#define cublasIdamax hipblasIdamax

#define cublasIsamin hipblasIsamin
#define cublasIdamin hipblasIdamin

// BLAS Level 2
#define cublasSgemv hipblasSgemv
#define cublasDgemv hipblasDgemv
#define cublasCgemv hipblasCgemv
#define cublasZgemv hipblasZgemv

#define cublasSger hipblasSger
#define cublasDger hipblasDger

#define cublasSsymv hipblasSsymv
#define cublasDsymv hipblasDsymv

#define cublasStrmv hipblasStrmv
#define cublasDtrmv hipblasDtrmv

#define cublasStrsv hipblasStrsv
#define cublasDtrsv hipblasDtrsv

// BLAS Level 3
#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasCgemm hipblasCgemm
#define cublasZgemm hipblasZgemm

#define cublasSgemmEx hipblasSgemmEx
#define cublasGemmEx hipblasGemmEx

#define cublasSgemmBatched hipblasSgemmBatched
#define cublasDgemmBatched hipblasDgemmBatched

#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasDgemmStridedBatched hipblasDgemmStridedBatched

#define cublasSsyrk hipblasSsyrk
#define cublasDsyrk hipblasDsyrk

#define cublasSsymm hipblasSsymm
#define cublasDsymm hipblasDsymm

#define cublasStrmm hipblasStrmm
#define cublasDtrmm hipblasDtrmm

#define cublasStrsm hipblasStrsm
#define cublasDtrsm hipblasDtrsm

// Math mode (for Tensor cores)
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_TENSOR_OP_MATH
#define cublasMath_t hipblasMath_t

// Helper function to get error string
inline const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case HIPBLAS_STATUS_SUCCESS: return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE: return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "HIPBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown hipBLAS error";
    }
}

#else // CUDA backend

#include <cublas_v2.h>

#endif // LFS_USE_HIP

// Common error checking macro
#define BLAS_CHECK(call)                                                       \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "BLAS error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__,                                        \
                    cublasGetStatusString(status));                            \
        }                                                                      \
    } while (0)

#endif // LFS_HIP_BLAS_COMPAT_H
