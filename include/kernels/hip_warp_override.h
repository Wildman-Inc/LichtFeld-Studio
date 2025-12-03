/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_warp_override.h
 * @brief Override HIP SDK's warp sync functions BEFORE hip_runtime.h is included.
 * 
 * This header MUST be force-included via -include flag before any other headers.
 * It prevents HIP SDK's amd_warp_sync_functions.h from being used, which requires
 * 64-bit masks even on RDNA architectures that use 32-bit warp size.
 * 
 * Instead, we redirect *_sync functions to their non-sync equivalents which
 * work naturally with the architecture's warp size.
 */

#ifndef LFS_HIP_WARP_OVERRIDE_H
#define LFS_HIP_WARP_OVERRIDE_H

// Only apply these overrides for HIP device compilation
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

// Define a guard to prevent amd_warp_sync_functions.h from defining its versions
// This is a workaround for ROCm SDK 7.x which has strict 64-bit mask requirements
#ifndef AMD_WARP_SYNC_FUNCTIONS_H
#define AMD_WARP_SYNC_FUNCTIONS_H

// Forward declare HIP's native warp functions (these don't need masks)
// They will be defined by hip_runtime.h after this header
extern "C" {
// Ballot - returns mask of threads where predicate is true
__device__ unsigned long long __ballot(int predicate);

// Vote operations
__device__ int __all(int predicate);
__device__ int __any(int predicate);

// Shuffle operations (templated in HIP, but we declare base types)
}

// For device code, define the _sync variants to use non-sync versions
// These macros will shadow any later definitions
#ifdef __HIP_DEVICE_COMPILE__

// Ballot with sync - ignore mask, use native ballot
// Note: Using variadic macros to handle extra optional parameters
#define __ballot_sync(mask, pred) __ballot(pred)

// All/Any sync - ignore mask
#define __all_sync(mask, pred) __all(pred)
#define __any_sync(mask, pred) __any(pred)

// Shuffle sync - ignore mask, pass through to native shuffle
// Native HIP __shfl takes (val, src_lane, width=warpSize)
#define __shfl_sync(mask, val, src, ...) __shfl(val, src, ##__VA_ARGS__)
#define __shfl_up_sync(mask, val, delta, ...) __shfl_up(val, delta, ##__VA_ARGS__)
#define __shfl_down_sync(mask, val, delta, ...) __shfl_down(val, delta, ##__VA_ARGS__)
#define __shfl_xor_sync(mask, val, lanemask, ...) __shfl_xor(val, lanemask, ##__VA_ARGS__)

// Active mask - returns ballot of all active threads
#define __activemask() __ballot(1)

#endif // __HIP_DEVICE_COMPILE__

#endif // AMD_WARP_SYNC_FUNCTIONS_H

#endif // HIP platform

#endif // LFS_HIP_WARP_OVERRIDE_H
