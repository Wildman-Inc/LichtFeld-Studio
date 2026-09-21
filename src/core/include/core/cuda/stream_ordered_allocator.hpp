/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/hip_runtime_compat.h"
#include "core/export.hpp"

// HIP and CUDA version numbers are unrelated. Header availability alone also
// does not imply that the selected device/driver implements memory pools.
#if LFS_USE_HIP || CUDART_VERSION >= 11020
#define LFS_HAS_STREAM_ORDERED_ALLOCATOR 1
#else
#define LFS_HAS_STREAM_ORDERED_ALLOCATOR 0
#endif

namespace lfs::core {

    // Queries the calling thread's device, caching successful capability queries
    // per thread/device. LFS_DISABLE_ASYNC_ALLOCATOR is a process-start override
    // for driver diagnosis and exercising the synchronous fallback.
    [[nodiscard]] LFS_CORE_API bool stream_ordered_allocation_supported() noexcept;

} // namespace lfs::core
