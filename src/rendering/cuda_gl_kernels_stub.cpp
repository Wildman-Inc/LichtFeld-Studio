/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.h"
#include "core/logger.hpp"
#include "core/cuda/hip_runtime_compat.h"

#include <atomic>

namespace lfs {

    void launchAdjustSaturation(
        float*,
        const float*,
        float,
        float,
        float,
        float,
        int,
        cudaStream_t) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LOG_WARN("launchAdjustSaturation is disabled for HIP on Windows (stub fallback)");
        }
    }

#ifdef CUDA_GL_INTEROP_ENABLED
    void launchWriteInterleavedPosColor(
        const float*,
        const float*,
        float*,
        int,
        cudaStream_t) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LOG_WARN("launchWriteInterleavedPosColor is disabled for HIP on Windows (stub fallback)");
        }
    }
#endif

} // namespace lfs
