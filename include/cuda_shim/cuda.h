/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cuda.h
 * @brief HIP-compatible CUDA driver API shim header.
 */

#ifndef _LFS_CUDA_DRIVER_SHIM_H_
#define _LFS_CUDA_DRIVER_SHIM_H_

// Include HIP runtime compatibility layer (provides driver-like APIs too)
#if __has_include("core/cuda/hip_runtime_compat.h")
#include "core/cuda/hip_runtime_compat.h"
#elif __has_include("../../src/core/include/core/cuda/hip_runtime_compat.h")
#include "../../src/core/include/core/cuda/hip_runtime_compat.h"
#else
#error "hip_runtime_compat.h not found. Add src/core/include to include paths."
#endif

// For driver API specific functionality, include HIP driver API
#if defined(USE_HIP) && USE_HIP
#include <hip/hip_runtime.h>
#endif

#endif // _LFS_CUDA_DRIVER_SHIM_H_
