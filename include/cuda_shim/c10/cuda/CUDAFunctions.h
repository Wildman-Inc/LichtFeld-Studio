/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDAFunctions.h
 * @brief Shim that redirects CUDAFunctions to HIPFunctions for ROCm builds
 */

#ifndef C10_CUDA_FUNCTIONS_SHIM_H_
#define C10_CUDA_FUNCTIONS_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP functions
#include <c10/hip/HIPFunctions.h>

// Create namespace aliases for code that uses c10::cuda namespace
namespace c10 {
namespace cuda {

// Forward device_count
inline c10::DeviceIndex device_count() noexcept {
    return ::c10::hip::device_count();
}

// Forward current_device
inline c10::DeviceIndex current_device() {
    return ::c10::hip::current_device();
}

// Forward set_device
inline void set_device(c10::DeviceIndex device) {
    ::c10::hip::set_device(device);
}

// Forward stream_synchronize
inline void stream_synchronize(hipStream_t stream) {
    ::c10::hip::stream_synchronize(stream);
}

} // namespace cuda
} // namespace c10

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDAFunctions.h>

#endif

#endif // C10_CUDA_FUNCTIONS_SHIM_H_
