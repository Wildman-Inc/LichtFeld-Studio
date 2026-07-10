/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDAStream.h
 * @brief Shim that redirects CUDAStream to HIPStream for ROCm builds
 */

#ifndef C10_CUDA_STREAM_SHIM_H_
#define C10_CUDA_STREAM_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP stream
#include <c10/hip/HIPStream.h>

// Create namespace aliases for code that uses c10::cuda namespace
namespace c10 {
namespace cuda {

// Alias the HIPStream class as CUDAStream
using CUDAStream = ::c10::hip::HIPStream;

// Forward the getCurrentCUDAStream function
inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
    return ::c10::hip::getCurrentHIPStream(device_index);
}

// Forward the setCurrentCUDAStream function
inline void setCurrentCUDAStream(CUDAStream stream) {
    ::c10::hip::setCurrentHIPStream(stream);
}

// Forward getDefaultCUDAStream
inline CUDAStream getDefaultCUDAStream(c10::DeviceIndex device_index = -1) {
    return ::c10::hip::getDefaultHIPStream(device_index);
}

} // namespace cuda
} // namespace c10

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDAStream.h>

#endif

#endif // C10_CUDA_STREAM_SHIM_H_
