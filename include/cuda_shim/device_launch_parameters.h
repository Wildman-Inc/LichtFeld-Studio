/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file device_launch_parameters.h
 * @brief HIP-compatible CUDA device launch parameters shim header.
 *
 * This header provides compatibility with CUDA code that includes
 * device_launch_parameters.h for kernel launch built-in variables.
 * In HIP, these are provided by hip_runtime.h.
 */

#ifndef _LFS_DEVICE_LAUNCH_PARAMETERS_SHIM_H_
#define _LFS_DEVICE_LAUNCH_PARAMETERS_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    // HIP provides these through hip_runtime.h
    // threadIdx, blockIdx, blockDim, gridDim are all available
    #include <hip/hip_runtime.h>
#else
    #include <device_launch_parameters.h>
#endif

#endif // _LFS_DEVICE_LAUNCH_PARAMETERS_SHIM_H_
