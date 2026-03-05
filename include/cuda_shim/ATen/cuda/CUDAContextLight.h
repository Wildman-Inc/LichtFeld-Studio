/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file ATen/cuda/CUDAContextLight.h
 * @brief Shim that redirects CUDAContextLight to HIP equivalent for ROCm builds
 * 
 * This shim is needed because:
 * 1. PyTorch ROCm wheel includes both CUDA and HIP headers
 * 2. CUDAContextLight.h uses TORCH_CUDA_CPP_API which is not defined for HIP builds
 * 3. We need to route to the HIP version instead
 */

#ifndef ATEN_CUDA_CONTEXT_LIGHT_SHIM_H_
#define ATEN_CUDA_CONTEXT_LIGHT_SHIM_H_

#if defined(LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

// For HIP builds, redirect to HIP version which uses TORCH_HIP_CPP_API
#include <ATen/hip/HIPContextLight.h>

#else // Not HIP build

// For CUDA builds, include the original header
#include_next <ATen/cuda/CUDAContextLight.h>

#endif

#endif // ATEN_CUDA_CONTEXT_LIGHT_SHIM_H_
