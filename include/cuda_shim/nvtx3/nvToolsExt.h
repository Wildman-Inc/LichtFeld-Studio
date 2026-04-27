/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#if defined(USE_HIP) && USE_HIP

#include <cstdint>
#include <cwchar>

extern "C" {
    static inline int nvtxRangePushA(const char*) { return 0; }
    static inline int nvtxRangePushW(const wchar_t*) { return 0; }
    static inline int nvtxRangePushEx(const void*) { return 0; }
    static inline int nvtxRangePop(void) { return 0; }
    static inline void nvtxMarkA(const char*) {}
    static inline void nvtxMarkW(const wchar_t*) {}
}

#ifndef nvtxRangePush
#define nvtxRangePush nvtxRangePushA
#endif
#ifndef nvtxMark
#define nvtxMark nvtxMarkA
#endif

#else

#if defined(__has_include_next)
#if __has_include_next(<nvtx3/nvToolsExt.h>)
#include_next <nvtx3/nvToolsExt.h>
#elif __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#else
#error "nvtx3/nvToolsExt.h not found"
#endif
#else
#include <nvtx3/nvToolsExt.h>
#endif

#endif
