/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <initializer_list>
#include <optional>

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core {

    // Thread-local current CUDA stream (PyTorch-style).
    // Exported from lfs_core so the singleton is shared across DSO boundaries.
    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

    // Makes execution_stream wait (GPU-side) for work currently enqueued on
    // dependency_stream. Uses pooled events; falls back to a host sync on failure.
    LFS_CORE_API void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream);

    LFS_CORE_API cudaStream_t prepare_inputs_for_stream(
        std::initializer_list<const Tensor*> inputs,
        std::optional<cudaStream_t> execution_stream = std::nullopt);

    // Host<->device copy ordered against `stream`, the tensor's home stream.
    // Downloads complete on the host after waiting for that stream's writes.
    // Non-default-stream uploads stage an owned copy in cached pinned memory,
    // then enqueue the DMA on home, after allocation and previous use. The
    // source can expire on return; staging is retained until DMA completion.
    LFS_CORE_API cudaError_t memcpy_ordered(void* dst, const void* src, size_t bytes,
                                            cudaMemcpyKind kind, cudaStream_t stream);

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(getCurrentCUDAStream()) {
            setCurrentCUDAStream(stream);
        }

        ~CUDAStreamGuard() {
            setCurrentCUDAStream(prev_stream_);
        }

        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

} // namespace lfs::core
