/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file c10/cuda/impl/CUDAGuardImpl.h
 * @brief Shim that redirects CUDAGuardImpl to HIPGuardImpl for ROCm builds
 */

#ifndef C10_CUDA_IMPL_GUARD_IMPL_SHIM_H_
#define C10_CUDA_IMPL_GUARD_IMPL_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP guard impl
#include <c10/hip/impl/HIPGuardImpl.h>

// Create namespace aliases for code that uses c10::cuda::impl namespace
namespace c10 {
namespace cuda {
namespace impl {

// Alias HIP implementations as CUDA implementations
using HIPGuardImpl = ::c10::hip::impl::HIPGuardImpl;

} // namespace impl
} // namespace cuda
} // namespace c10

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <c10/cuda/impl/CUDAGuardImpl.h>

#endif

#endif // C10_CUDA_IMPL_GUARD_IMPL_SHIM_H_
