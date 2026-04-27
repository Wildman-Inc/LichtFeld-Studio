/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cuda_runtime.h
 * @brief HIP-compatible CUDA runtime shim header.
 *
 * This header intercepts #include <cuda_runtime.h> and redirects
 * to the HIP compatibility layer for AMD ROCm builds.
 */

#ifndef _LFS_CUDA_RUNTIME_SHIM_H_
#define _LFS_CUDA_RUNTIME_SHIM_H_

// Include HIP runtime compatibility layer
#if __has_include("core/cuda/hip_runtime_compat.h")
#include "core/cuda/hip_runtime_compat.h"
#elif __has_include("../../src/core/include/core/cuda/hip_runtime_compat.h")
#include "../../src/core/include/core/cuda/hip_runtime_compat.h"
#else
#error "hip_runtime_compat.h not found. Add src/core/include to include paths."
#endif

#endif // _LFS_CUDA_RUNTIME_SHIM_H_
