/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#if defined(USE_HIP) && USE_HIP
inline void nvtxNameCudaStreamA(cudaStream_t, const char*) {}
inline void nvtxNameCudaStreamW(cudaStream_t, const wchar_t*) {}
#endif
