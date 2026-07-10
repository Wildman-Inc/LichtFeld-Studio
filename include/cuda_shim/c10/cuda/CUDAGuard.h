/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/CUDAGuard.h
 * @brief Shim that redirects CUDAGuard to HIPGuard for ROCm builds
 */

#ifndef C10_CUDA_GUARD_SHIM_H_
#define C10_CUDA_GUARD_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP guard
#include <c10/hip/HIPGuard.h>

// Create namespace aliases for code that uses c10::cuda namespace
namespace c10 {
namespace cuda {

// Alias HIP guards as CUDA guards
using CUDAGuard = ::c10::hip::HIPGuard;
using OptionalCUDAGuard = ::c10::hip::OptionalHIPGuard;
using CUDAStreamGuard = ::c10::hip::HIPStreamGuard;
using OptionalCUDAStreamGuard = ::c10::hip::OptionalHIPStreamGuard;
using CUDAMultiStreamGuard = ::c10::hip::HIPMultiStreamGuard;

} // namespace cuda
} // namespace c10

// Create namespace aliases for code that uses at::cuda namespace
namespace at {
namespace cuda {

// Alias HIP guards for at::cuda namespace (same as c10::cuda)
using CUDAGuard = ::c10::hip::HIPGuard;
using OptionalCUDAGuard = ::c10::hip::OptionalHIPGuard;
using CUDAStreamGuard = ::c10::hip::HIPStreamGuard;
using OptionalCUDAStreamGuard = ::c10::hip::OptionalHIPStreamGuard;

} // namespace cuda
} // namespace at

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/CUDAGuard.h>

#endif

#endif // C10_CUDA_GUARD_SHIM_H_
