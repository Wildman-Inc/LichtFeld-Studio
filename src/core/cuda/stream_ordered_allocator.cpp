/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/stream_ordered_allocator.hpp"
#include "core/environment.hpp"

namespace lfs::core {

    bool stream_ordered_allocation_supported() noexcept {
#if LFS_HAS_STREAM_ORDERED_ALLOCATOR
        static const bool disabled = environment::flag("LFS_DISABLE_ASYNC_ALLOCATOR");
        if (disabled) {
            return false;
        }
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;
        }
        thread_local int cached_device = -1;
        thread_local bool cached_supported = false;
        if (device != cached_device) {
            int supported = 0;
            const auto previous_error = cudaPeekAtLastError();
            const auto status = cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
#if LFS_USE_HIP
            const bool unavailable = status == hipErrorNotSupported || status == hipErrorInvalidValue;
#else
            const bool unavailable = status == cudaErrorNotSupported || status == cudaErrorInvalidValue;
#endif
            if (unavailable) {
                // An older driver may not recognize this attribute. Do not
                // turn that expected fallback into a later launch failure.
                if (previous_error == cudaSuccess && cudaPeekAtLastError() == status) {
                    (void)cudaGetLastError();
                }
            } else if (status != cudaSuccess) {
                return false;
            }
            cached_device = device;
            cached_supported = status == cudaSuccess && supported != 0;
        }
        return cached_supported;
#else
        return false;
#endif
    }

} // namespace lfs::core
