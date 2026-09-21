/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/stream_ordered_allocator.hpp"
#include "core/cuda_allocation.hpp"
#include "core/environment.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::core;

namespace {
    struct Stream {
        cudaStream_t value = nullptr;
        Stream() { LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
        ~Stream() {
            CudaMemoryPool::instance().release_stream(value);
            (void)cudaStreamDestroy(value);
        }
    };

    struct ConsumerGate {
        std::atomic<bool> entered{false};
        std::atomic<bool> release{false};
        std::atomic<bool> timed_out{false};

        static void wait(void* data) {
            auto& gate = *static_cast<ConsumerGate*>(data);
            gate.entered.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!gate.release.load()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    gate.timed_out.store(true);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ~ConsumerGate() {
            release.store(true);
            (void)cudaDeviceSynchronize();
        }
    };
} // namespace

TEST(StreamOrderedAllocator, CapabilityOrExplicitFallback) {
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int supported = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device), cudaSuccess);
    EXPECT_EQ(stream_ordered_allocation_supported(),
              supported != 0 && !environment::flag("LFS_DISABLE_ASYNC_ALLOCATOR"));
}

TEST(StreamOrderedAllocator, MovedScratchRetainsAllocationMethodAndPendingWrites) {
    Stream stream;
    unsigned char* host = nullptr;
    ASSERT_EQ(cudaMallocHost(&host, 256), cudaSuccess);
    for (int i = 1; i <= 256; ++i) {
        UniqueCudaAllocation<StreamOrderedCudaAllocator> scratch(1024 * 1024, stream.value, "test.scratch");
        ASSERT_EQ(cudaMemsetAsync(scratch.get(), i & 255, 1024 * 1024, stream.value), cudaSuccess);
        auto moved = std::move(scratch);
        EXPECT_FALSE(scratch);
        ASSERT_EQ(cudaMemcpyAsync(host, moved.get(), 256, cudaMemcpyDeviceToHost, stream.value), cudaSuccess);
        moved.reset();
        ASSERT_EQ(cudaStreamSynchronize(stream.value), cudaSuccess);
        for (int j = 0; j < 256; ++j) {
            ASSERT_EQ(host[j], i & 255);
        }
    }
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
}

TEST(StreamOrderedAllocator, ScratchFreeDoesNotWaitForUnrelatedStream) {
    if (!stream_ordered_allocation_supported()) {
        GTEST_SKIP() << "Requires asynchronous allocation support";
    }
    Stream home;
    Stream unrelated;
    UniqueCudaAllocation<StreamOrderedCudaAllocator> scratch(1024 * 1024, home.value, "test.scratch");
    ASSERT_EQ(cudaStreamSynchronize(home.value), cudaSuccess);
    ConsumerGate gate;
    ASSERT_EQ(cudaLaunchHostFunc(unrelated.value, ConsumerGate::wait, &gate), cudaSuccess);
    scratch.reset();
    EXPECT_FALSE(gate.timed_out.load());
    EXPECT_EQ(cudaStreamQuery(unrelated.value), cudaErrorNotReady);
    gate.release.store(true);
    ASSERT_EQ(cudaStreamSynchronize(unrelated.value), cudaSuccess);
}

class StreamOrderedReuse : public ::testing::TestWithParam<bool> {};

TEST_P(StreamOrderedReuse, PendingConsumerFinishesBeforeBucketReuse) {
    if (!stream_ordered_allocation_supported()) {
        GTEST_SKIP() << "Requires asynchronous allocation support";
    }
    auto& pool = CudaMemoryPool::instance();
    pool.trim_cached_memory();
    Stream home;
    Stream other;
    const cudaStream_t consumer = GetParam() ? nullptr : other.value;
    unsigned char* host = nullptr;
    ASSERT_EQ(cudaMallocHost(&host, 256), cudaSuccess);
    void* ptr = pool.allocate(1024 * 1024, home.value);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(cudaMemsetAsync(ptr, 0x35, 1024 * 1024, home.value), cudaSuccess);
    bridgeStreams(home.value, consumer);
    ConsumerGate gate;
    ASSERT_EQ(cudaLaunchHostFunc(consumer, ConsumerGate::wait, &gate), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host, ptr, 256, cudaMemcpyDeviceToHost, consumer), cudaSuccess);
    pool.record_stream(ptr, consumer);
    pool.deallocate(ptr);
    void* reused = pool.allocate(1024 * 1024, home.value);
    EXPECT_EQ(reused, ptr);
    ASSERT_EQ(cudaMemsetAsync(reused, 0x7e, 1024 * 1024, home.value), cudaSuccess);

    // A consumer deliberately held on the host must also hold the later GPU
    // overwrite, including when the consumer is the legacy default stream.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!gate.entered.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(gate.entered.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(cudaStreamQuery(home.value), cudaErrorNotReady);
    gate.release.store(true);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(home.value), cudaSuccess);
    EXPECT_FALSE(gate.timed_out.load());
    for (int j = 0; j < 256; ++j) {
        EXPECT_EQ(host[j], 0x35);
    }
    pool.deallocate(reused);
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(ConsumerStreams, StreamOrderedReuse, ::testing::Bool());

TEST(StreamOrderedAllocator, ExactAllocationOutlivesRetiredHomeStream) {
    auto& pool = CudaMemoryPool::instance();
    void* ptr = nullptr;
    {
        Stream home;
        ptr = pool.try_allocate_exact_async(1024 * 1024 + 17, home.value);
        ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(cudaMemsetAsync(ptr, 0x49, 1024 * 1024 + 17, home.value), cudaSuccess);
    }
    std::array<unsigned char, 16> values{};
    ASSERT_EQ(cudaMemcpy(values.data(), ptr, values.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (auto value : values) {
        EXPECT_EQ(value, 0x49);
    }
    pool.deallocate(ptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(StreamOrderedAllocator, UntrackedClassicAllocationCanBeReleased) {
    void* ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024 * 1024), cudaSuccess);
    CudaMemoryPool::instance().deallocate(ptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(StreamOrderedAllocator, EphemeralUploadRejectsGraphCapture) {
    Stream home;
    void* ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024), cudaSuccess);
    const auto capture_status = cudaStreamBeginCapture(home.value, cudaStreamCaptureModeGlobal);
    if (capture_status != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaFree(ptr);
        GTEST_SKIP() << "Stream capture is unavailable";
    }
    const int value = 42;
    EXPECT_EQ(memcpy_ordered(ptr, &value, sizeof(value), cudaMemcpyHostToDevice, home.value),
              cudaErrorStreamCaptureUnsupported);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(home.value, &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaFree(ptr), cudaSuccess);
}

TEST(StreamOrderedAllocator, PageableUploadFollowsPendingUseAndOwnsHostBytes) {
    auto& pool = CudaMemoryPool::instance();
    auto& pinned = PinnedMemoryAllocator::instance();
    Stream home;
    constexpr size_t bytes = 1024 * 1024;
    void* ptr = pool.allocate(bytes, home.value);
    ASSERT_NE(ptr, nullptr);
    // Warm staging memory before deliberately blocking the destination stream.
    void* staging = pinned.allocate(bytes);
    ASSERT_NE(staging, nullptr);
    pinned.deallocate(staging);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    unsigned char* host = nullptr;
    ASSERT_EQ(cudaMallocHost(&host, 256), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(ptr, 0x35, bytes, home.value), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(home.value), cudaSuccess);
    ConsumerGate gate;
    ASSERT_EQ(cudaLaunchHostFunc(home.value, ConsumerGate::wait, &gate), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host, ptr, 256, cudaMemcpyDeviceToHost, home.value), cudaSuccess);
    {
        std::vector<unsigned char> source(bytes, 0x7e);
        ASSERT_EQ(memcpy_ordered(ptr, source.data(), bytes, cudaMemcpyHostToDevice, home.value), cudaSuccess);
        std::fill(source.begin(), source.end(), 0xff);
    }
    EXPECT_FALSE(gate.timed_out.load());
    gate.release.store(true);
    ASSERT_EQ(cudaStreamSynchronize(home.value), cudaSuccess);
    for (int j = 0; j < 256; ++j) {
        EXPECT_EQ(host[j], 0x35) << "Upload overwrote a pending reader";
    }
    ASSERT_EQ(cudaMemcpy(host, ptr, 256, cudaMemcpyDeviceToHost), cudaSuccess);
    for (int j = 0; j < 256; ++j) {
        EXPECT_EQ(host[j], 0x7e) << "Upload used expired/replaced host bytes";
    }
    pool.deallocate(ptr);
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
}
