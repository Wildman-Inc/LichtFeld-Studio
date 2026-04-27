/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDACachingAllocator.h
 * @brief Shim that redirects CUDACachingAllocator to HIPCachingAllocator for ROCm builds
 *
 * This is critical because both CUDA and HIP versions define FreeMemoryCallback
 * in the c10 namespace, causing redefinition errors if both are included.
 */

#ifndef C10_CUDA_CACHING_ALLOCATOR_SHIM_H_
#define C10_CUDA_CACHING_ALLOCATOR_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, ONLY include the HIP caching allocator
// Do NOT include the CUDA version to avoid FreeMemoryCallback redefinition
#include <c10/hip/HIPCachingAllocator.h>

// Create aliases for code that uses c10::cuda:: namespace
namespace c10 {
namespace cuda {

// Forward the CUDACachingAllocator namespace to HIPCachingAllocator
namespace CUDACachingAllocator = ::c10::hip::HIPCachingAllocator;

} // namespace cuda
} // namespace c10

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDACachingAllocator.h>

#endif

#endif // C10_CUDA_CACHING_ALLOCATOR_SHIM_H_
