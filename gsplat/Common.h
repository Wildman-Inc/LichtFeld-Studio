#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

// Include HIP compatibility layer
#include "kernels/hip_compat.h"

// Device guard headers - must be outside of namespace
#if USE_HIP
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

// Stream compatibility macros for gsplat
#if USE_HIP
namespace at { namespace cuda {
    // Alias HIPStreamMasqueradingAsCUDA as CUDAStream for at::cuda namespace
    using CUDAStream = c10::hip::HIPStreamMasqueradingAsCUDA;

    // Forward getCurrentCUDAStream
    inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
        return c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
    }

    // Forward getStreamFromPool  
    inline CUDAStream getStreamFromPool(const bool isHighPriority = false, c10::DeviceIndex device = -1) {
        return c10::hip::getStreamFromPoolMasqueradingAsCUDA(isHighPriority, device);
    }
}} // namespace at::cuda
#endif

namespace gsplat {

//
// Some Macros.
//
#if USE_HIP
#define CHECK_CUDA(x) TORCH_CHECK(x.is_hip(), #x " must be a HIP/ROCm tensor")
#else
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// Device guard macros
#if USE_HIP
#define DEVICE_GUARD(_ten) \
    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(_ten));
#else
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));
#endif

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#if USE_HIP
// For HIP, use hipCUB wrapper with HIPCachingAllocator
#define CUB_WRAPPER(func, ...)                                               \
    do {                                                                     \
        size_t temp_storage_bytes = 0;                                       \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                      \
        auto& caching_allocator = *::c10::hip::HIPCachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);  \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);           \
    } while (false)
#else
#define CUB_WRAPPER(func, ...)                                               \
    do {                                                                     \
        size_t temp_storage_bytes = 0;                                       \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                      \
        auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get(); \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);  \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);           \
    } while (false)
#endif

    //
    // Convenience typedefs for CUDA types
    //
    using vec2 = glm::vec<2, float>;
    using vec3 = glm::vec<3, float>;
    using vec4 = glm::vec<4, float>;
    using mat2 = glm::mat<2, 2, float>;
    using mat3 = glm::mat<3, 3, float>;
    using mat4 = glm::mat<4, 4, float>;
    using mat3x2 = glm::mat<3, 2, float>;

    //
    // Legacy Camera Types
    //
    enum CameraModelType {
        PINHOLE = 0,
        ORTHO = 1,
        FISHEYE = 2,
        EQUIRECTANGULAR = 3
    };

#define N_THREADS_PACKED 256
#define ALPHA_THRESHOLD  (1.f / 255.f)

} // namespace gsplat