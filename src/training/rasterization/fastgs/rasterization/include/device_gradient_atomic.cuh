/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/hip_runtime_compat.h"

namespace fast_lfs::rasterization::kernels {

    // The caller must prove that every destination belongs to its own
    // device allocation before selecting Native=true. On gfx11
    // native float atomics require coarse-grained, cacheable device storage;
    // managed, mapped host, external and fine-grained allocations are excluded.
    // Native addition may flush subnormals, just as CUDA float atomicAdd does.
    template <bool Native>
    __device__ __forceinline__ void atomic_add_device_gradient(float* destination, float value) {
#if defined(LFS_USE_HIP) && LFS_USE_HIP &&                                   \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
     defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__))
        if constexpr (Native) {
            unsafeAtomicAdd(destination, value);
            return;
        }
#endif
        atomicAdd(destination, value);
    }

} // namespace fast_lfs::rasterization::kernels
