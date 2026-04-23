/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_runtime_compat.h
 * @brief CUDA/HIP runtime API compatibility for host code.
 *
 * This header provides a unified interface for GPU runtime API calls
 * (memory allocation, device management, streams, events, etc.)
 * that works with both CUDA and HIP backends.
 *
 * For device code (kernels), use hip_compat.h instead.
 *
 * Include this header instead of <cuda_runtime.h> in C++ host code files.
 */

#ifndef LFS_HIP_RUNTIME_COMPAT_H
#define LFS_HIP_RUNTIME_COMPAT_H

// ============================================================================
// Backend Selection
// ============================================================================

// Check if we're building for HIP
#if defined(LFS_USE_HIP)
    // Use value from build configuration.
#elif defined(USE_HIP) && USE_HIP
    #define LFS_USE_HIP 1
#elif defined(__HIP_PLATFORM_AMD__)
    #define LFS_USE_HIP 1
#else
    #define LFS_USE_HIP 0
#endif

// ============================================================================
// Runtime API Headers
// ============================================================================

#if LFS_USE_HIP

// Prevent cuda_runtime.h from being included by defining its include guard
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__
#endif
#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H
#endif
#ifndef __CUDA_RUNTIME_API_H__
#define __CUDA_RUNTIME_API_H__
#endif

// Include HIP runtime
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hip/math_functions.h>
#if defined(__linux__) || defined(__APPLE__)
#include <hip/hip_gl_interop.h>
#endif

// On HIP device compilation (not host pass), Clang's HIP wrappers provide
// __device__ math overloads (fminf/fmaxf/sqrtf/expf/__expf/etc.) that CUDA code
// expects. Include them explicitly on Windows to avoid falling back to UCRT
// host-only declarations during kernel compilation.
#if defined(__HIP_DEVICE_COMPILE__) && defined(_WIN32)
#include <__clang_hip_math.h>
#include <__clang_hip_cmath.h>
#endif

#if defined(__HIPCC__)
#include <thrust/system/hip/execution_policy.h>
namespace thrust {
namespace cuda = hip;
} // namespace thrust
#endif

// Define CUDART_VERSION equivalent for HIP
// HIP_VERSION format: major * 10000000 + minor * 100000 + patch
// CUDART_VERSION format: major * 1000 + minor * 10
#ifndef CUDART_VERSION
#define CUDART_VERSION (HIP_VERSION_MAJOR * 1000 + HIP_VERSION_MINOR * 10)
#endif

// ============================================================================
// Type Aliases (CUDA names -> HIP types)
// ============================================================================

// Note: Use typedefs/using instead of macros where possible
// to avoid issues with HIP headers that already define these types

// Error types
using cudaError_t = hipError_t;
// Note: cudaSuccess etc are already defined by HIP's cuda compatibility layer
// or we provide them below

// Stream and event types
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

// Device properties
using cudaDeviceProp = hipDeviceProp_t;

// Memory copy kinds
using cudaMemcpyKind = hipMemcpyKind;

// Pointer attributes
using cudaPointerAttributes = hipPointerAttribute_t;

// CUDA array handle
using cudaArray_t = hipArray_t;

// ============================================================================
// Constant Mappings (only if not already defined by HIP)
// ============================================================================

#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotReady hipErrorNotReady
#endif

#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#endif

#ifndef cudaMemoryTypeHost
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeManaged hipMemoryTypeUnified
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#endif

#ifndef cudaHostAllocDefault
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#endif

#ifndef cudaStreamDefault
#define cudaStreamDefault hipStreamDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif

#ifndef cudaEventDefault
#define cudaEventDefault hipEventDefault
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDisableTiming hipEventDisableTiming
#endif

// Virtual memory management constants
#ifndef cudaMemAllocationTypePinned
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#endif
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif
#ifndef cudaMemAllocationGranularityMinimum
#define cudaMemAllocationGranularityMinimum hipMemAllocationGranularityMinimum
#endif
#ifndef cudaMemAccessFlagsProtReadWrite
#define cudaMemAccessFlagsProtReadWrite hipMemAccessFlagsProtReadWrite
#endif

// ============================================================================
// Stream Capture Mode (for CUDA Graphs / PyTorch compatibility)
// ============================================================================

// Stream capture mode - use HIP enum directly
#ifndef cudaStreamCaptureMode
using cudaStreamCaptureMode = hipStreamCaptureMode;
// Also define the enum values as macros for compatibility
#ifndef cudaStreamCaptureModeGlobal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#endif
#endif

// Stream capture status - use HIP enum directly
#ifndef cudaStreamCaptureStatus
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
// Also define the enum values as macros for compatibility
#ifndef cudaStreamCaptureStatusNone
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#endif
#endif

// Thread-local stream capture mode exchange
#ifndef cudaThreadExchangeStreamCaptureMode
inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    // HIP equivalent: hipThreadExchangeStreamCaptureMode
    // Since cudaStreamCaptureMode is now an alias for hipStreamCaptureMode,
    // we can pass it directly
#if HIP_VERSION >= 50500000
    return hipThreadExchangeStreamCaptureMode(mode);
#else
    // No-op for older ROCm versions that don't support this
    (void)mode;
    return cudaSuccess;
#endif
}
#endif

// Stream capture begin/end
#ifndef cudaStreamBeginCapture
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#endif

// Graph types
#ifndef cudaGraph_t
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaGraphNode_t = hipGraphNode_t;
#endif

// Graph operations
#ifndef cudaGraphCreate
#define cudaGraphCreate hipGraphCreate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy
#endif

// ============================================================================
// Function Mappings (CUDA -> HIP)
// ============================================================================

// Error handling
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaDriverGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#endif

// Memory allocation
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAlloc hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaHostFree hipHostFree
#endif

// Memory operations
#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpyFromSymbolAsync hipMemcpyFromSymbolAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpy3D hipMemcpy3D
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpy2DToArray hipMemcpy2DToArray
#define cudaMemcpy2DFromArray hipMemcpy2DFromArray
#endif

// Async memory allocation (CUDA 11.2+)
#ifndef cudaMallocAsync
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemPoolCreate hipMemPoolCreate
#define cudaMemPoolDestroy hipMemPoolDestroy
#endif
#ifndef cudaMemAddressReserve
#define cudaMemAddressReserve hipMemAddressReserve
#endif
#ifndef cudaMemAddressFree
#define cudaMemAddressFree hipMemAddressFree
#endif
#ifndef cudaMemCreate
#define cudaMemCreate hipMemCreate
#endif
#ifndef cudaMemRelease
#define cudaMemRelease hipMemRelease
#endif
#ifndef cudaMemMap
#define cudaMemMap hipMemMap
#endif
#ifndef cudaMemUnmap
#define cudaMemUnmap hipMemUnmap
#endif
#ifndef cudaMemSetAccess
#define cudaMemSetAccess hipMemSetAccess
#endif
#ifndef cudaMemGetAllocationGranularity
#define cudaMemGetAllocationGranularity hipMemGetAllocationGranularity
#endif

// Device management
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#endif

// Stream operations
#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamQuery hipStreamQuery
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamGetFlags hipStreamGetFlags
#endif

// Device stream priority
#ifndef cudaDeviceGetStreamPriorityRange
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif

// Event operations
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#endif

// Pointer attributes
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif

// Peer access
#ifndef cudaMemcpyPeer
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#endif

// Memory prefetch and advise
#ifndef cudaMemPrefetchAsync
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemAdvise hipMemAdvise
#endif

// Launch configuration
#ifndef cudaOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#endif

// Function attributes
#ifndef cudaFuncSetAttribute
namespace lfs::cuda_compat {
    template <typename FuncT>
    inline hipError_t func_set_attribute(FuncT func, hipFuncAttribute attr, int value) {
        return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
    }
} // namespace lfs::cuda_compat
#define cudaFuncSetAttribute(...) ::lfs::cuda_compat::func_set_attribute(__VA_ARGS__)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#endif

// Host function launch (callback in stream)
#ifndef cudaLaunchHostFunc
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaHostFn_t hipHostFn_t
#endif

// ============================================================================
// CUDA Driver VMM Compatibility (for cuMem*/CUmem* usage in existing code)
// ============================================================================

using CUresult = cudaError_t;
using CUdeviceptr = char*;
using CUcontext = hipCtx_t;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS cudaSuccess
#endif

#ifndef CU_MEM_ALLOCATION_TYPE_PINNED
#define CU_MEM_ALLOCATION_TYPE_PINNED cudaMemAllocationTypePinned
#endif
#ifndef CU_MEM_LOCATION_TYPE_DEVICE
#define CU_MEM_LOCATION_TYPE_DEVICE cudaMemLocationTypeDevice
#endif
#ifndef CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM cudaMemAllocationGranularityMinimum
#endif
#ifndef CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE cudaMemAccessFlagsProtReadWrite
#endif

inline CUresult lfsCuMemAddressReserve(
    CUdeviceptr* ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags) {
    void* requested = static_cast<void*>(addr);
    void* reserved = requested;
    const auto status = cudaMemAddressReserve(&reserved, size, alignment, requested, flags);
    if (status == cudaSuccess) {
        *ptr = static_cast<CUdeviceptr>(reserved);
    }
    return status;
}

#define cuMemAddressReserve lfsCuMemAddressReserve
#define cuMemAddressFree cudaMemAddressFree
#define cuMemCreate cudaMemCreate
#define cuMemRelease cudaMemRelease
#define cuMemMap cudaMemMap
#define cuMemUnmap cudaMemUnmap
#define cuMemSetAccess cudaMemSetAccess
#define cuMemGetAllocationGranularity cudaMemGetAllocationGranularity
#define cuCtxGetCurrent hipCtxGetCurrent
#define cuCtxSetCurrent hipCtxSetCurrent

// ============================================================================
// Graphics Interop (OpenGL) - may not be available on all platforms
// ============================================================================

#if defined(__linux__) || defined(__APPLE__)
// Graphics interop available on Linux/macOS
#ifndef cudaGraphicsResource_t
using cudaGraphicsResource_t = hipGraphicsResource_t;
#define cudaGraphicsMapResources hipGraphicsMapResources
#define cudaGraphicsUnmapResources hipGraphicsUnmapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsGLRegisterBuffer hipGraphicsGLRegisterBuffer
#define cudaGraphicsGLRegisterImage hipGraphicsGLRegisterImage
#define cudaGraphicsSubResourceGetMappedArray hipGraphicsSubResourceGetMappedArray
#define cudaGraphicsUnregisterResource hipGraphicsUnregisterResource
#define cudaGraphicsRegisterFlagsNone hipGraphicsRegisterFlagsNone
#define cudaGraphicsRegisterFlagsReadOnly hipGraphicsRegisterFlagsReadOnly
#define cudaGraphicsRegisterFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard
#define cudaGraphicsRegisterFlagsSurfaceLoadStore hipGraphicsRegisterFlagsSurfaceLoadStore
#endif
#endif // Graphics interop available

#else // CUDA backend

#include <cuda_runtime.h>

#endif // LFS_USE_HIP

// ============================================================================
// Common Utilities
// ============================================================================

#include <cstdio>
#include <stdexcept>
#include <string>

// GPU error checking macro
#ifndef GPU_CHECK
#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "GPU error at %s:%d - %s: %s\n",                   \
                    __FILE__, __LINE__,                                        \
                    cudaGetErrorName(err),                                     \
                    cudaGetErrorString(err));                                  \
        }                                                                      \
    } while (0)
#endif

// GPU error checking macro with return
#ifndef GPU_CHECK_RETURN
#define GPU_CHECK_RETURN(call)                                                 \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "GPU error at %s:%d - %s: %s\n",                   \
                    __FILE__, __LINE__,                                        \
                    cudaGetErrorName(err),                                     \
                    cudaGetErrorString(err));                                  \
            return err;                                                        \
        }                                                                      \
    } while (0)
#endif

// GPU error checking macro with throw
#ifndef GPU_CHECK_THROW
#define GPU_CHECK_THROW(call)                                                  \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("GPU error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + " - " +                             \
                cudaGetErrorName(err) + ": " +                                 \
                cudaGetErrorString(err));                                      \
        }                                                                      \
    } while (0)
#endif

#endif // LFS_HIP_RUNTIME_COMPAT_H
