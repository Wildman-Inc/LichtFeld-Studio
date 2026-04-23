/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file ATen/cuda/CUDAEvent.h
 * @brief Shim that redirects CUDAEvent to HIPEvent for ROCm builds
 *
 * Note: ATen/hip/HIPEvent.h actually defines `at::cuda::CUDAEvent` (not HIPEvent)
 * for hipify compatibility, so we just need to include it directly.
 */

#ifndef ATEN_CUDA_EVENT_SHIM_H_
#define ATEN_CUDA_EVENT_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, include HIPEvent.h which defines at::cuda::CUDAEvent
#include <ATen/hip/HIPEvent.h>

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <ATen/cuda/CUDAEvent.h>

#endif

#endif // ATEN_CUDA_EVENT_SHIM_H_
