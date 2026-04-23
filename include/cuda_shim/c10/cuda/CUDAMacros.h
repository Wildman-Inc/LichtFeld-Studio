/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDAMacros.h
 * @brief Shim that provides CUDA macros for ROCm builds
 *
 * This ensures TORCH_CUDA_CPP_API and other macros are defined for HIP builds.
 */

#ifndef C10_CUDA_MACROS_SHIM_H_
#define C10_CUDA_MACROS_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, include HIP macros first
#include <c10/hip/HIPMacros.h>
#include <c10/macros/Export.h>

// Map CUDA macros to HIP macros
#ifndef C10_CUDA_API
#define C10_CUDA_API C10_HIP_API
#endif

#ifndef C10_CUDA_EXPORT
#define C10_CUDA_EXPORT C10_EXPORT
#endif

#ifndef C10_CUDA_IMPORT
#define C10_CUDA_IMPORT C10_IMPORT
#endif

// Map TORCH_CUDA_CPP_API to TORCH_HIP_CPP_API
#ifndef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API TORCH_HIP_CPP_API
#endif

// Map TORCH_CUDA_CU_API to TORCH_HIP_API
#ifndef TORCH_CUDA_CU_API
#define TORCH_CUDA_CU_API TORCH_HIP_API
#endif

// Map TORCH_CUDA_API to TORCH_HIP_API
#ifndef TORCH_CUDA_API
#define TORCH_CUDA_API TORCH_HIP_API
#endif

// Define C10_COMPILE_TIME_MAX_GPUS if not defined
#ifndef C10_COMPILE_TIME_MAX_GPUS
#define C10_COMPILE_TIME_MAX_GPUS 120
#endif

// Define C10_CUDA_BUILD_SHARED_LIBS for proper exports
#ifndef C10_CUDA_BUILD_SHARED_LIBS
#define C10_CUDA_BUILD_SHARED_LIBS
#endif

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDAMacros.h>

#endif

#endif // C10_CUDA_MACROS_SHIM_H_
