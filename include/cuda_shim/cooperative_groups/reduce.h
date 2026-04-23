/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#if defined(USE_HIP) && USE_HIP

#include <cooperative_groups.h>

namespace cooperative_groups {

    template <typename T>
    struct plus {
        __device__ T operator()(const T& a, const T& b) const { return a + b; }
    };

    template <typename T>
    struct greater {
        __device__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
    };

    template <typename GroupT, typename T, typename OpT>
    __device__ T reduce(const GroupT& group, T value, OpT op) {
        for (unsigned int offset = group.size() / 2; offset > 0; offset >>= 1) {
            const T other = group.shfl_down(value, offset);
            if (group.thread_rank() + offset < group.size()) {
                value = op(value, other);
            }
        }
        return group.shfl(value, 0);
    }

} // namespace cooperative_groups

#endif
