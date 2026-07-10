/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cusparse.h
 * @brief HIP-compatible cuSPARSE shim header.
 *
 * This shim handles the complex issue where:
 * 1. ROCm 7.10's hipsparse-types.h uses #if(!defined(CUDART_VERSION)) to define hipsparseStatus_t
 * 2. PyTorch ROCm headers define CUDART_VERSION for compatibility
 * 3. This causes hipsparseStatus_t to be undefined when hipsparse.h is included after CUDART_VERSION is defined
 *
 * Solution: Always include this header before any PyTorch headers that might define CUDART_VERSION.
 */

#ifndef _LFS_CUSPARSE_SHIM_H_
#define _LFS_CUSPARSE_SHIM_H_

#if defined(USE_HIP) && USE_HIP

    // CRITICAL: We MUST ensure hipsparse types are defined BEFORE any other code
    // can include hipsparse with CUDART_VERSION defined.
    //
    // Save and temporarily undefine CUDART_VERSION to force hipsparse-types.h
    // to define the required types (hipsparseStatus_t, etc.)

    #pragma push_macro("CUDART_VERSION")
    #ifdef CUDART_VERSION
    #undef CUDART_VERSION
    #endif

    #include <hip/hip_runtime.h>
    // Include types first to ensure they're defined
    #include <hipsparse/hipsparse-types.h>
    // Now include the rest of hipsparse - types are already defined
    #include <hipsparse/hipsparse.h>

    // Restore CUDART_VERSION
    #pragma pop_macro("CUDART_VERSION")

    // Type mappings (CUDA -> HIP) - map to the types defined by hipsparse
    #define cusparseHandle_t hipsparseHandle_t
    #define cusparseMatDescr_t hipsparseMatDescr_t
    #define cusparseStatus_t hipsparseStatus_t
    #define cusparseIndexBase_t hipsparseIndexBase_t
    #define cusparseMatrixType_t hipsparseMatrixType_t
    #define cusparseFillMode_t hipsparseFillMode_t
    #define cusparseDiagType_t hipsparseDiagType_t
    #define cusparseOperation_t hipsparseOperation_t
    #define cusparsePointerMode_t hipsparsePointerMode_t
    #define cusparseAction_t hipsparseAction_t
    #define cusparseDirection_t hipsparseDirection_t

    // Status mappings
    #define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
    #define CUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
    #define CUSPARSE_STATUS_ALLOC_FAILED HIPSPARSE_STATUS_ALLOC_FAILED
    #define CUSPARSE_STATUS_INVALID_VALUE HIPSPARSE_STATUS_INVALID_VALUE
    #define CUSPARSE_STATUS_ARCH_MISMATCH HIPSPARSE_STATUS_ARCH_MISMATCH
    #define CUSPARSE_STATUS_MAPPING_ERROR HIPSPARSE_STATUS_MAPPING_ERROR
    #define CUSPARSE_STATUS_EXECUTION_FAILED HIPSPARSE_STATUS_EXECUTION_FAILED
    #define CUSPARSE_STATUS_INTERNAL_ERROR HIPSPARSE_STATUS_INTERNAL_ERROR
    #define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
    #define CUSPARSE_STATUS_ZERO_PIVOT HIPSPARSE_STATUS_ZERO_PIVOT
    #define CUSPARSE_STATUS_NOT_SUPPORTED HIPSPARSE_STATUS_NOT_SUPPORTED

    // Index base mappings
    #define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
    #define CUSPARSE_INDEX_BASE_ONE HIPSPARSE_INDEX_BASE_ONE

    // Matrix type mappings
    #define CUSPARSE_MATRIX_TYPE_GENERAL HIPSPARSE_MATRIX_TYPE_GENERAL
    #define CUSPARSE_MATRIX_TYPE_SYMMETRIC HIPSPARSE_MATRIX_TYPE_SYMMETRIC
    #define CUSPARSE_MATRIX_TYPE_HERMITIAN HIPSPARSE_MATRIX_TYPE_HERMITIAN
    #define CUSPARSE_MATRIX_TYPE_TRIANGULAR HIPSPARSE_MATRIX_TYPE_TRIANGULAR

    // Fill mode mappings
    #define CUSPARSE_FILL_MODE_LOWER HIPSPARSE_FILL_MODE_LOWER
    #define CUSPARSE_FILL_MODE_UPPER HIPSPARSE_FILL_MODE_UPPER

    // Diagonal type mappings
    #define CUSPARSE_DIAG_TYPE_NON_UNIT HIPSPARSE_DIAG_TYPE_NON_UNIT
    #define CUSPARSE_DIAG_TYPE_UNIT HIPSPARSE_DIAG_TYPE_UNIT

    // Operation mappings
    #define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
    #define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
    #define CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE

    // Pointer mode mappings
    #define CUSPARSE_POINTER_MODE_HOST HIPSPARSE_POINTER_MODE_HOST
    #define CUSPARSE_POINTER_MODE_DEVICE HIPSPARSE_POINTER_MODE_DEVICE

    // Action mappings
    #define CUSPARSE_ACTION_SYMBOLIC HIPSPARSE_ACTION_SYMBOLIC
    #define CUSPARSE_ACTION_NUMERIC HIPSPARSE_ACTION_NUMERIC

    // Direction mappings
    #define CUSPARSE_DIRECTION_ROW HIPSPARSE_DIRECTION_ROW
    #define CUSPARSE_DIRECTION_COLUMN HIPSPARSE_DIRECTION_COLUMN

    // Function mappings
    #define cusparseCreate hipsparseCreate
    #define cusparseDestroy hipsparseDestroy
    #define cusparseSetStream hipsparseSetStream
    #define cusparseGetStream hipsparseGetStream
    #define cusparseSetPointerMode hipsparseSetPointerMode
    #define cusparseGetPointerMode hipsparseGetPointerMode
    #define cusparseCreateMatDescr hipsparseCreateMatDescr
    #define cusparseDestroyMatDescr hipsparseDestroyMatDescr
    #define cusparseSetMatType hipsparseSetMatType
    #define cusparseGetMatType hipsparseGetMatType
    #define cusparseSetMatFillMode hipsparseSetMatFillMode
    #define cusparseGetMatFillMode hipsparseGetMatFillMode
    #define cusparseSetMatDiagType hipsparseSetMatDiagType
    #define cusparseGetMatDiagType hipsparseGetMatDiagType
    #define cusparseSetMatIndexBase hipsparseSetMatIndexBase
    #define cusparseGetMatIndexBase hipsparseGetMatIndexBase

#else
    #include <cusparse.h>
#endif

#endif // _LFS_CUSPARSE_SHIM_H_
