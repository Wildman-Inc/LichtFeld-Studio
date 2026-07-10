/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_intrinsics_compat.h
 * @brief CUDA intrinsic functions compatibility layer for HIP.
 *
 * This header provides HIP-compatible implementations of CUDA intrinsic
 * functions that don't have direct HIP equivalents.
 */

#ifndef LFS_HIP_INTRINSICS_COMPAT_H
#define LFS_HIP_INTRINSICS_COMPAT_H

#include "hip_runtime_compat.h"

#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)

// ============================================================================
// Cache control load/store intrinsics
// ============================================================================

// CUDA provides cache control intrinsics like __ldg, __ldcs, __ldca, __ldcg
// HIP has __ldg but not all others. Provide fallbacks.

#if defined(__HIP_DEVICE_COMPILE__)

// __ldg - Load through texture cache (read-only, cached)
// HIP has native __ldg support

// __ldcs - Load cached streaming (L1 bypass, L2 cached)
// Fallback: use regular load
#ifndef __ldcs
template<typename T>
__device__ __forceinline__ T __ldcs(const T* ptr) {
    return *ptr;
}
#endif

// __ldca - Load cached all (cached in L1 and L2)
// Fallback: use regular load
#ifndef __ldca
template<typename T>
__device__ __forceinline__ T __ldca(const T* ptr) {
    return *ptr;
}
#endif

// __ldcg - Load cached global (L1 bypass, L2 cached for global)
// Fallback: use regular load
#ifndef __ldcg
template<typename T>
__device__ __forceinline__ T __ldcg(const T* ptr) {
    return *ptr;
}
#endif

// __ldcv - Load cached volatile (bypass all caches)
// Fallback: use volatile load
#ifndef __ldcv
template<typename T>
__device__ __forceinline__ T __ldcv(const T* ptr) {
    return *const_cast<volatile const T*>(ptr);
}
#endif

// __stcg - Store cached global
#ifndef __stcg
template<typename T>
__device__ __forceinline__ void __stcg(T* ptr, T val) {
    *ptr = val;
}
#endif

// __stcs - Store cached streaming
#ifndef __stcs
template<typename T>
__device__ __forceinline__ void __stcs(T* ptr, T val) {
    *ptr = val;
}
#endif

// __stwb - Store write-back
#ifndef __stwb
template<typename T>
__device__ __forceinline__ void __stwb(T* ptr, T val) {
    *ptr = val;
}
#endif

// __stwt - Store write-through
#ifndef __stwt
template<typename T>
__device__ __forceinline__ void __stwt(T* ptr, T val) {
    *ptr = val;
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Math intrinsics
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// Fast reciprocal (may have different precision on HIP)
#ifndef __frcp_rn
__device__ __forceinline__ float __frcp_rn(float x) {
    return 1.0f / x;
}
#endif

// Fast reciprocal square root
#ifndef __frsqrt_rn
__device__ __forceinline__ float __frsqrt_rn(float x) {
    return rsqrtf(x);
}
#endif

// Double precision fast intrinsics
#ifndef __drcp_rn
__device__ __forceinline__ double __drcp_rn(double x) {
    return 1.0 / x;
}
#endif

#ifndef __dsqrt_rn
__device__ __forceinline__ double __dsqrt_rn(double x) {
    return sqrt(x);
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Saturation arithmetic
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// Saturating add for unsigned int
#ifndef __uaddsat
__device__ __forceinline__ unsigned int __uaddsat(unsigned int a, unsigned int b) {
    unsigned int result = a + b;
    return (result < a) ? 0xFFFFFFFFu : result;
}
#endif

// Saturating subtract for unsigned int
#ifndef __usubsat
__device__ __forceinline__ unsigned int __usubsat(unsigned int a, unsigned int b) {
    return (a > b) ? (a - b) : 0;
}
#endif

// Saturating multiply high
#ifndef __umulhi
__device__ __forceinline__ unsigned int __umulhi(unsigned int a, unsigned int b) {
    return static_cast<unsigned int>((static_cast<unsigned long long>(a) * b) >> 32);
}
#endif

#ifndef __mul24
__device__ __forceinline__ int __mul24(int a, int b) {
    return a * b;  // Modern GPUs don't have special 24-bit multiply
}
#endif

#ifndef __umul24
__device__ __forceinline__ unsigned int __umul24(unsigned int a, unsigned int b) {
    return a * b;
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Bit manipulation intrinsics
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// Population count (HIP has __popc)
// These should be available, but provide fallbacks

// Bit reversal
#ifndef __brev
__device__ __forceinline__ unsigned int __brev(unsigned int x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}
#endif

#ifndef __brevll
__device__ __forceinline__ unsigned long long __brevll(unsigned long long x) {
    unsigned int hi = __brev(static_cast<unsigned int>(x));
    unsigned int lo = __brev(static_cast<unsigned int>(x >> 32));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

// Byte permute
#ifndef __byte_perm
__device__ __forceinline__ unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int selector) {
    unsigned int result = 0;
    for (int i = 0; i < 4; i++) {
        unsigned int sel = (selector >> (i * 4)) & 0x7;
        unsigned int byte;
        if (sel < 4) {
            byte = (a >> (sel * 8)) & 0xFF;
        } else {
            byte = (b >> ((sel - 4) * 8)) & 0xFF;
        }
        result |= (byte << (i * 8));
    }
    return result;
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Synchronization primitives
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// Named barriers (CUDA compute capability 7.0+)
// HIP doesn't have named barriers - use regular syncthreads
#ifndef __namedbarrier_sync
__device__ __forceinline__ void __namedbarrier_sync(int name, int count) {
    (void)name;
    (void)count;
    __syncthreads();
}
#endif

// Synchronize with memory fence
#ifndef __syncthreads_and
__device__ __forceinline__ int __syncthreads_and(int pred) {
    __syncthreads();
    // Use warp vote to implement block-wide AND
    // This is an approximation - full implementation would need shared memory
    return __all(pred);
}
#endif

#ifndef __syncthreads_or
__device__ __forceinline__ int __syncthreads_or(int pred) {
    __syncthreads();
    return __any(pred);
}
#endif

#ifndef __syncthreads_count
__device__ __forceinline__ int __syncthreads_count(int pred) {
    __syncthreads();
    return __popc(__ballot(pred));
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Type conversion intrinsics
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// Float to int conversions with rounding modes
// HIP may have these, but provide fallbacks

#ifndef __float2int_rd
__device__ __forceinline__ int __float2int_rd(float x) {
    return static_cast<int>(floorf(x));
}
#endif

#ifndef __float2int_ru
__device__ __forceinline__ int __float2int_ru(float x) {
    return static_cast<int>(ceilf(x));
}
#endif

#ifndef __float2int_rz
__device__ __forceinline__ int __float2int_rz(float x) {
    return static_cast<int>(truncf(x));
}
#endif

// Unsigned versions
#ifndef __float2uint_rd
__device__ __forceinline__ unsigned int __float2uint_rd(float x) {
    return static_cast<unsigned int>(fmaxf(0.0f, floorf(x)));
}
#endif

#ifndef __float2uint_ru
__device__ __forceinline__ unsigned int __float2uint_ru(float x) {
    return static_cast<unsigned int>(fmaxf(0.0f, ceilf(x)));
}
#endif

#ifndef __float2uint_rz
__device__ __forceinline__ unsigned int __float2uint_rz(float x) {
    return static_cast<unsigned int>(fmaxf(0.0f, truncf(x)));
}
#endif

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Half precision helpers
// ============================================================================

#if defined(__HIP_DEVICE_COMPILE__)

// HIP provides __half type, but some CUDA intrinsics may need mapping
// Include hip_fp16.h for native support

#endif // __HIP_DEVICE_COMPILE__

#endif // __HIP_DEVICE_COMPILE__ || __CUDA_ARCH__

#endif // LFS_HIP_INTRINSICS_COMPAT_H
