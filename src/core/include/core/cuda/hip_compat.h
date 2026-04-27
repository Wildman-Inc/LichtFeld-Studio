/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_compat.h
 * @brief HIP/CUDA compatibility layer for portable GPU code.
 *
 * This header provides a compatibility layer that allows the same code
 * to compile and run on both NVIDIA GPUs (via CUDA) and AMD GPUs (via ROCm/HIP).
 *
 * This header includes hip_runtime_compat.h for runtime API compatibility,
 * and adds device-specific utilities for kernel code.
 *
 * For host-only code, you can include hip_runtime_compat.h directly.
 */

#ifndef HIP_COMPAT_H
#define HIP_COMPAT_H

// ============================================================================
// GPU Backend Detection
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    // ROCm/HIP backend
    #ifndef USE_HIP
    #define USE_HIP 1
    #endif
    #ifndef USE_CUDA
    #define USE_CUDA 0
    #endif
    #define GPU_BACKEND_NAME "ROCm/HIP"
#elif defined(__CUDACC__) || defined(CUDA_VERSION)
    // CUDA backend
    #ifndef USE_HIP
    #define USE_HIP 0
    #endif
    #ifndef USE_CUDA
    #define USE_CUDA 1
    #endif
    #define GPU_BACKEND_NAME "CUDA"
#else
    // CPU-only or unknown backend
    #ifndef USE_HIP
    #define USE_HIP 0
    #endif
    #ifndef USE_CUDA
    #define USE_CUDA 1
    #endif
    #define GPU_BACKEND_NAME "None"
#endif

// ============================================================================
// Include Runtime Compatibility Header
// ============================================================================
// This provides CUDA->HIP API mappings for runtime functions

#include "hip_runtime_compat.h"

// ============================================================================
// HIP-Specific Includes (Device code headers)
// ============================================================================

#if USE_HIP

// HIP FP16 and cooperative groups (device code)
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>

// Cooperative groups namespace alias
namespace cg = cooperative_groups;
#endif

// ROCm math library headers
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>

// Warp size depends on architecture
// RDNA uses 32, CDNA/GCN uses 64
#if defined(__AMDGCN_WAVEFRONT_SIZE)
    #if __AMDGCN_WAVEFRONT_SIZE == 32
        #define WARP_SIZE 32
    #else
        #define WARP_SIZE 64
    #endif
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
    // RDNA 3 / RDNA 3.5 uses warp size of 32
    #define WARP_SIZE 32
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    // RDNA 2 uses warp size of 32
    #define WARP_SIZE 32
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
    // RDNA 1 uses warp size of 32
    #define WARP_SIZE 32
#else
    // CDNA and older GCN use warp size of 64
    #define WARP_SIZE 64
#endif

// cuBLAS -> hipBLAS (in addition to runtime compat)
#ifndef cublasHandle_t
#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemv hipblasSgemv
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#endif

// cuRAND -> hipRAND (in addition to runtime compat)
#ifndef curandState_t
#define curandState_t hiprandState_t
#define curandStateXORWOW_t hiprandStateXORWOW_t
#define curand_init hiprand_init
#define curand hiprand
#define curand_uniform hiprand_uniform
#define curand_normal hiprand_normal
#endif

// Math intrinsics
#ifndef __float2int_rn
#define __float2int_rn __float2int_rn  // Available in HIP
#define __int2float_rn __int2float_rn  // Available in HIP
#endif

#else // USE_CUDA

// CUDA includes
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#define WARP_SIZE 32

namespace cg = cooperative_groups;

#endif // USE_HIP

// ============================================================================
// Device function attributes (fallback for host code)
// ============================================================================

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

// ============================================================================
// Warp Operation Wrappers - PORTABLE INTERFACE
// ============================================================================
// Use these wrappers instead of __ballot_sync, __shfl_sync, etc.
// They work on both CUDA and HIP without requiring 64-bit masks on HIP.
//
// NOTE: We use LFS_ prefix to avoid collision with hipcub's WARP_BALLOT etc.

#if defined(__CUDA_ARCH__)
    // CUDA - use native _sync functions
    #define LFS_WARP_BALLOT(pred) __ballot_sync(0xFFFFFFFFu, pred)
    #define LFS_WARP_ALL(pred) __all_sync(0xFFFFFFFFu, pred)
    #define LFS_WARP_ANY(pred) __any_sync(0xFFFFFFFFu, pred)
    #define LFS_WARP_SHFL(val, lane) __shfl_sync(0xFFFFFFFFu, val, lane)
    #define LFS_WARP_SHFL_UP(val, delta) __shfl_up_sync(0xFFFFFFFFu, val, delta)
    #define LFS_WARP_SHFL_DOWN(val, delta) __shfl_down_sync(0xFFFFFFFFu, val, delta)
    #define LFS_WARP_SHFL_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFFu, val, mask)
    #define LFS_WARP_ACTIVEMASK() __activemask()
#elif defined(__HIP_DEVICE_COMPILE__)
    // HIP - use non-sync functions (they don't require mask parameter)
    #define LFS_WARP_BALLOT(pred) __ballot(pred)
    #define LFS_WARP_ALL(pred) __all(pred)
    #define LFS_WARP_ANY(pred) __any(pred)
    #define LFS_WARP_SHFL(val, lane) __shfl(val, lane)
    #define LFS_WARP_SHFL_UP(val, delta) __shfl_up(val, delta)
    #define LFS_WARP_SHFL_DOWN(val, delta) __shfl_down(val, delta)
    #define LFS_WARP_SHFL_XOR(val, mask) __shfl_xor(val, mask)
    #define LFS_WARP_ACTIVEMASK() __ballot(1)
#else
    // Host code - stubs
    #define LFS_WARP_BALLOT(pred) 0
    #define LFS_WARP_ALL(pred) 0
    #define LFS_WARP_ANY(pred) 0
    #define LFS_WARP_SHFL(val, lane) (val)
    #define LFS_WARP_SHFL_UP(val, delta) (val)
    #define LFS_WARP_SHFL_DOWN(val, delta) (val)
    #define LFS_WARP_SHFL_XOR(val, mask) (val)
    #define LFS_WARP_ACTIVEMASK() 0
#endif

// ============================================================================
// Device-only utilities (only compile when using device compiler)
// ============================================================================
#if defined(__HIPCC__) || defined(__CUDACC__)

// ============================================================================
// Thread Block Utilities
// ============================================================================

/**
 * @brief Get the linear thread index within a block
 */
__device__ __forceinline__ int get_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

/**
 * @brief Get the linear block index
 */
__device__ __forceinline__ int get_block_id() {
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

/**
 * @brief Get the global thread index
 */
__device__ __forceinline__ int get_global_thread_id() {
    return get_block_id() * (blockDim.x * blockDim.y * blockDim.z) + get_thread_id();
}

// ============================================================================
// Memory Fence Operations
// ============================================================================

__device__ __forceinline__ void thread_fence() {
    __threadfence();
}

__device__ __forceinline__ void thread_fence_block() {
    __threadfence_block();
}

__device__ __forceinline__ void thread_fence_system() {
    __threadfence_system();
}

// ============================================================================
// Fast Math Functions
// ============================================================================

// Reciprocal square root
__device__ __forceinline__ float rsqrt_fast(float x) {
#if USE_HIP
    return __frsqrt_rn(x);
#else
    return rsqrtf(x);
#endif
}

// Fast reciprocal
__device__ __forceinline__ float rcp_fast(float x) {
    return __frcp_rn(x);
}

#endif // __HIPCC__ || __CUDACC__

// ============================================================================
// Printing from device code
// ============================================================================

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    #define GPU_PRINTF(...) printf(__VA_ARGS__)
#else
    #define GPU_PRINTF(...)
#endif

#endif // HIP_COMPAT_H
