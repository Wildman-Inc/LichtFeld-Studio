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
#include "core/cuda/hip_runtime_compat.h"

#endif // _LFS_CUDA_RUNTIME_SHIM_H_
