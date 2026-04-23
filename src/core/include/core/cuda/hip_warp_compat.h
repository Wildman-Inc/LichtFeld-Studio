/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_warp_compat.h
 * @brief Warp-level operation compatibility layer for CUDA/HIP portability.
 *
 * This header provides portable warp operations that work across:
 * - NVIDIA CUDA GPUs (warp size 32, 32-bit mask)
 * - AMD RDNA GPUs (warp size 32, 32-bit mask)
 * - AMD CDNA/GCN GPUs (warp size 64, 64-bit mask)
 *
 * Usage:
 *   Instead of: __ballot_sync(0xFFFFFFFF, pred)
 *   Use:        warp_ballot(pred)
 *
 *   Instead of: __shfl_sync(0xFFFFFFFF, val, lane)
 *   Use:        warp_shfl(val, lane)
 */

#ifndef LFS_HIP_WARP_COMPAT_H
#define LFS_HIP_WARP_COMPAT_H

#include "hip_runtime_compat.h"

// ============================================================================
// Warp Size Detection
// ============================================================================

#if defined(__CUDA_ARCH__)
    // NVIDIA CUDA - always 32
    #define LFS_WARP_SIZE 32
    #define LFS_WARP_MASK_TYPE unsigned int
    #define LFS_FULL_WARP_MASK 0xFFFFFFFFu
#elif defined(__HIP_DEVICE_COMPILE__)
    // AMD HIP - detect based on architecture
    #if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
        defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
        defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
        // RDNA 1/2/3 - warp size 32
        #define LFS_WARP_SIZE 32
        #define LFS_WARP_MASK_TYPE unsigned int
        #define LFS_FULL_WARP_MASK 0xFFFFFFFFu
    #else
        // CDNA / GCN (MI100, MI200, MI300, older cards) - warp size 64
        #define LFS_WARP_SIZE 64
        #define LFS_WARP_MASK_TYPE unsigned long long
        #define LFS_FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFull
    #endif
#else
    // Host code - default to 32
    #define LFS_WARP_SIZE 32
    #define LFS_WARP_MASK_TYPE unsigned int
    #define LFS_FULL_WARP_MASK 0xFFFFFFFFu
#endif

// Backward compatibility with existing code
#ifndef WARP_SIZE
#define WARP_SIZE LFS_WARP_SIZE
#endif

// ============================================================================
// Device-only warp operations
// ============================================================================

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

// ----------------------------------------------------------------------------
// Ballot operations
// ----------------------------------------------------------------------------

/**
 * @brief Warp ballot - returns mask of threads where predicate is true
 * @param pred Boolean predicate
 * @return Ballot result (32-bit for CUDA/RDNA, 64-bit for CDNA/GCN)
 */
__device__ __forceinline__ LFS_WARP_MASK_TYPE warp_ballot(int pred) {
#if defined(__CUDA_ARCH__)
    return __ballot_sync(LFS_FULL_WARP_MASK, pred);
#elif LFS_WARP_SIZE == 32
    // RDNA - HIP uses 32-bit ballot
    return __ballot(pred);
#else
    // CDNA/GCN - HIP uses 64-bit ballot
    return __ballot(pred);
#endif
}

/**
 * @brief Warp ballot with explicit mask (for compatibility)
 */
__device__ __forceinline__ LFS_WARP_MASK_TYPE warp_ballot_sync(LFS_WARP_MASK_TYPE mask, int pred) {
#if defined(__CUDA_ARCH__)
    return __ballot_sync(mask, pred);
#else
    (void)mask; // HIP doesn't use explicit mask
    return __ballot(pred);
#endif
}

// ----------------------------------------------------------------------------
// Reduction operations
// ----------------------------------------------------------------------------

/**
 * @brief Warp all - true if all threads have predicate true
 */
__device__ __forceinline__ int warp_all(int pred) {
#if defined(__CUDA_ARCH__)
    return __all_sync(LFS_FULL_WARP_MASK, pred);
#else
    return __all(pred);
#endif
}

/**
 * @brief Warp any - true if any thread has predicate true
 */
__device__ __forceinline__ int warp_any(int pred) {
#if defined(__CUDA_ARCH__)
    return __any_sync(LFS_FULL_WARP_MASK, pred);
#else
    return __any(pred);
#endif
}

// ----------------------------------------------------------------------------
// Shuffle operations
// ----------------------------------------------------------------------------

/**
 * @brief Warp shuffle - get value from arbitrary lane
 */
template<typename T>
__device__ __forceinline__ T warp_shfl(T val, int src_lane) {
#if defined(__CUDA_ARCH__)
    return __shfl_sync(LFS_FULL_WARP_MASK, val, src_lane);
#else
    return __shfl(val, src_lane);
#endif
}

/**
 * @brief Warp shuffle with explicit width
 */
template<typename T>
__device__ __forceinline__ T warp_shfl(T val, int src_lane, int width) {
#if defined(__CUDA_ARCH__)
    return __shfl_sync(LFS_FULL_WARP_MASK, val, src_lane, width);
#else
    return __shfl(val, src_lane, width);
#endif
}

/**
 * @brief Warp shuffle up - get value from lower lane
 */
template<typename T>
__device__ __forceinline__ T warp_shfl_up(T val, unsigned int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_up_sync(LFS_FULL_WARP_MASK, val, delta);
#else
    return __shfl_up(val, delta);
#endif
}

/**
 * @brief Warp shuffle down - get value from higher lane
 */
template<typename T>
__device__ __forceinline__ T warp_shfl_down(T val, unsigned int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_down_sync(LFS_FULL_WARP_MASK, val, delta);
#else
    return __shfl_down(val, delta);
#endif
}

/**
 * @brief Warp shuffle XOR - butterfly exchange
 */
template<typename T>
__device__ __forceinline__ T warp_shfl_xor(T val, int lane_mask) {
#if defined(__CUDA_ARCH__)
    return __shfl_xor_sync(LFS_FULL_WARP_MASK, val, lane_mask);
#else
    return __shfl_xor(val, lane_mask);
#endif
}

// ----------------------------------------------------------------------------
// Active mask operations
// ----------------------------------------------------------------------------

/**
 * @brief Get mask of active threads in current warp
 */
__device__ __forceinline__ LFS_WARP_MASK_TYPE warp_activemask() {
#if defined(__CUDA_ARCH__)
    return __activemask();
#else
    // HIP: __ballot(1) returns mask of all active lanes
    return __ballot(1);
#endif
}

// ----------------------------------------------------------------------------
// Vote operations with sync
// ----------------------------------------------------------------------------

__device__ __forceinline__ int warp_all_sync(LFS_WARP_MASK_TYPE mask, int pred) {
#if defined(__CUDA_ARCH__)
    return __all_sync(mask, pred);
#else
    (void)mask;
    return __all(pred);
#endif
}

__device__ __forceinline__ int warp_any_sync(LFS_WARP_MASK_TYPE mask, int pred) {
#if defined(__CUDA_ARCH__)
    return __any_sync(mask, pred);
#else
    (void)mask;
    return __any(pred);
#endif
}

// ----------------------------------------------------------------------------
// Shuffle with sync (explicit mask)
// ----------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T warp_shfl_sync(LFS_WARP_MASK_TYPE mask, T val, int src_lane) {
#if defined(__CUDA_ARCH__)
    return __shfl_sync(mask, val, src_lane);
#else
    (void)mask;
    return __shfl(val, src_lane);
#endif
}

template<typename T>
__device__ __forceinline__ T warp_shfl_up_sync(LFS_WARP_MASK_TYPE mask, T val, unsigned int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_up_sync(mask, val, delta);
#else
    (void)mask;
    return __shfl_up(val, delta);
#endif
}

template<typename T>
__device__ __forceinline__ T warp_shfl_down_sync(LFS_WARP_MASK_TYPE mask, T val, unsigned int delta) {
#if defined(__CUDA_ARCH__)
    return __shfl_down_sync(mask, val, delta);
#else
    (void)mask;
    return __shfl_down(val, delta);
#endif
}

template<typename T>
__device__ __forceinline__ T warp_shfl_xor_sync(LFS_WARP_MASK_TYPE mask, T val, int lane_mask) {
#if defined(__CUDA_ARCH__)
    return __shfl_xor_sync(mask, val, lane_mask);
#else
    (void)mask;
    return __shfl_xor(val, lane_mask);
#endif
}

// ----------------------------------------------------------------------------
// Warp reduction utilities
// ----------------------------------------------------------------------------

/**
 * @brief Warp-level sum reduction
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = LFS_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += warp_shfl_down(val, offset);
    }
    return val;
}

/**
 * @brief Warp-level max reduction
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = LFS_WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = warp_shfl_down(val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

/**
 * @brief Warp-level min reduction
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = LFS_WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = warp_shfl_down(val, offset);
        val = (val < other) ? val : other;
    }
    return val;
}

/**
 * @brief Get lane ID within warp
 */
__device__ __forceinline__ int warp_lane_id() {
    return threadIdx.x % LFS_WARP_SIZE;
}

/**
 * @brief Get warp ID within block
 */
__device__ __forceinline__ int warp_id() {
    return threadIdx.x / LFS_WARP_SIZE;
}

#endif // __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__

// ============================================================================
// Compatibility macros for existing code
// ============================================================================

// These macros allow existing CUDA code to compile with minimal changes
// They convert 32-bit masks to the appropriate type for the architecture

#if defined(__CUDA_ARCH__)
    // CUDA - use native functions
    #define BALLOT_SYNC(mask, pred) __ballot_sync(mask, pred)
    #define ALL_SYNC(mask, pred) __all_sync(mask, pred)
    #define ANY_SYNC(mask, pred) __any_sync(mask, pred)
    #define SHFL_SYNC(mask, val, lane) __shfl_sync(mask, val, lane)
    #define SHFL_UP_SYNC(mask, val, delta) __shfl_up_sync(mask, val, delta)
    #define SHFL_DOWN_SYNC(mask, val, delta) __shfl_down_sync(mask, val, delta)
    #define SHFL_XOR_SYNC(mask, val, lanemask) __shfl_xor_sync(mask, val, lanemask)
#elif defined(__HIP_DEVICE_COMPILE__)
    // HIP - ignore mask parameter, cast if needed
    #if LFS_WARP_SIZE == 64
        // CDNA/GCN - need 64-bit operations
        #define BALLOT_SYNC(mask, pred) __ballot(pred)
        #define ALL_SYNC(mask, pred) __all(pred)
        #define ANY_SYNC(mask, pred) __any(pred)
        #define SHFL_SYNC(mask, val, lane) __shfl(val, lane)
        #define SHFL_UP_SYNC(mask, val, delta) __shfl_up(val, delta)
        #define SHFL_DOWN_SYNC(mask, val, delta) __shfl_down(val, delta)
        #define SHFL_XOR_SYNC(mask, val, lanemask) __shfl_xor(val, lanemask)
    #else
        // RDNA - 32-bit operations
        #define BALLOT_SYNC(mask, pred) __ballot(pred)
        #define ALL_SYNC(mask, pred) __all(pred)
        #define ANY_SYNC(mask, pred) __any(pred)
        #define SHFL_SYNC(mask, val, lane) __shfl(val, lane)
        #define SHFL_UP_SYNC(mask, val, delta) __shfl_up(val, delta)
        #define SHFL_DOWN_SYNC(mask, val, delta) __shfl_down(val, delta)
        #define SHFL_XOR_SYNC(mask, val, lanemask) __shfl_xor(val, lanemask)
    #endif
#else
    // Host code stubs
    #define BALLOT_SYNC(mask, pred) (0)
    #define ALL_SYNC(mask, pred) (0)
    #define ANY_SYNC(mask, pred) (0)
    #define SHFL_SYNC(mask, val, lane) (val)
    #define SHFL_UP_SYNC(mask, val, delta) (val)
    #define SHFL_DOWN_SYNC(mask, val, delta) (val)
    #define SHFL_XOR_SYNC(mask, val, lanemask) (val)
#endif

// Full mask for current architecture
#define FULL_WARP_MASK LFS_FULL_WARP_MASK

#endif // LFS_HIP_WARP_COMPAT_H
