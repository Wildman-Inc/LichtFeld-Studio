/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDAGraphsC10Utils.h
 * @brief Shim that redirects CUDAGraphsC10Utils to HIP equivalent for ROCm builds
 */

#ifndef C10_CUDA_GRAPHS_C10_UTILS_SHIM_H_
#define C10_CUDA_GRAPHS_C10_UTILS_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP version
#include <c10/hip/HIPGraphsC10Utils.h>

// Create namespace aliases for code that uses c10::cuda namespace
namespace c10 {
namespace cuda {

// Alias the CaptureStatus enum
using CaptureStatus = ::c10::hip::CaptureStatus;

// Alias the stream capture mode guard
using CUDAStreamCaptureModeGuard = ::c10::hip::HIPStreamCaptureModeGuard;

// Forward the currentStreamCaptureStatusMayInitCtx function
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
    return ::c10::hip::currentStreamCaptureStatusMayInitCtx();
}

} // namespace cuda
} // namespace c10

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDAGraphsC10Utils.h>

#endif

#endif // C10_CUDA_GRAPHS_C10_UTILS_SHIM_H_
