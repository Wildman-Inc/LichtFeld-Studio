/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file gpu_runtime.h
 * @brief Unified GPU runtime header that abstracts CUDA and HIP
 * 
 * This header provides a unified interface for GPU runtime functions
 * that works with both NVIDIA CUDA and AMD ROCm/HIP backends.
 * 
 * Include this instead of <cuda_runtime.h> or <hip_runtime.h> directly.
 */

#pragma once

#include "kernels/hip_compat.h"

// Re-export commonly used types and functions for convenience
#if USE_HIP

// Error handling macros
#define GPU_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    hipGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_GPU(call) GPU_CHECK(call)

// Type aliases for uniform code
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuDeviceProp_t = hipDeviceProp_t;

constexpr auto gpuSuccess = hipSuccess;

// Function aliases
#define gpuGetLastError hipGetLastError
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuGetErrorString hipGetErrorString
#define gpuGetErrorName hipGetErrorName
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync
#define gpuHostAlloc hipHostMalloc
#define gpuFreeHost hipHostFree
#define gpuMallocHost hipHostMalloc
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime

// Memory copy types
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToHost hipMemcpyHostToHost

#else // USE_CUDA

// Error handling macros
#define GPU_CHECK(call)                                                         \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_GPU(call) GPU_CHECK(call)

// Type aliases for uniform code
using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuDeviceProp_t = cudaDeviceProp;

constexpr auto gpuSuccess = cudaSuccess;

// Function aliases
#define gpuGetLastError cudaGetLastError
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuGetErrorName cudaGetErrorName
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync
#define gpuHostAlloc cudaHostAlloc
#define gpuFreeHost cudaFreeHost
#define gpuMallocHost cudaMallocHost
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime

// Memory copy types
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToHost cudaMemcpyHostToHost

#endif // USE_HIP

