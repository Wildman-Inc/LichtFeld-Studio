/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file cooperative_groups.h
 * @brief HIP-compatible CUDA cooperative groups shim header.
 */

#ifndef _LFS_COOPERATIVE_GROUPS_SHIM_H_
#define _LFS_COOPERATIVE_GROUPS_SHIM_H_

#if defined(USE_HIP) && USE_HIP
    // Include HIP cooperative groups
    #if defined(__HIPCC__)
        #include <hip/hip_cooperative_groups.h>
        namespace cg = cooperative_groups;
    #endif
#else
    // Include CUDA cooperative groups
    #include <cooperative_groups.h>
    namespace cg = cooperative_groups;
#endif

#endif // _LFS_COOPERATIVE_GROUPS_SHIM_H_
