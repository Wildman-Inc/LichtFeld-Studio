/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"

#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

cudaError_t launch_native_gradient_atomic_probe(float* sums, int count, float value,
                                                bool alternate_sign, cudaStream_t stream);

namespace {
    using lfs::core::RasterizerMemoryArena;

    class ArenaFrame {
    public:
        explicit ArenaFrame(RasterizerMemoryArena& arena)
            : arena_(arena), id_(arena.begin_frame()) {}
        ~ArenaFrame() { arena_.end_frame(id_); }
        auto allocator() { return arena_.get_allocator(id_); }

    private:
        RasterizerMemoryArena& arena_;
        uint64_t id_;
    };

    class HipNativeAtomics : public ::testing::Test {
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
                GTEST_SKIP() << "GPU unavailable";
            }
        }
    };

    TEST_F(HipNativeAtomics, OwnedArenaRangeIncludesOnlyCommittedDeviceStorage) {
        for (const bool vmm : {false, true}) {
            SCOPED_TRACE(vmm ? "VMM when supported" : "malloc");
            RasterizerMemoryArena::Config config;
            config.enable_vmm = vmm;
            RasterizerMemoryArena arena(config);
            ArenaFrame frame(arena);
            auto allocate = frame.allocator();
            auto* memory = allocate(4096);
            ASSERT_NE(memory, nullptr);
            EXPECT_TRUE(arena.owns_device_allocation(memory, 4096));
            EXPECT_TRUE(arena.owns_device_allocation(memory + 128, sizeof(float)));
            EXPECT_FALSE(arena.owns_device_allocation(nullptr, sizeof(float)));
            EXPECT_FALSE(arena.owns_device_allocation(memory, 0));
            EXPECT_FALSE(arena.owns_device_allocation(memory, std::numeric_limits<size_t>::max()));
            EXPECT_FALSE(arena.owns_device_allocation(memory + arena.get_statistics().capacity, 1));
            float host_memory = 0.0f;
            EXPECT_FALSE(arena.owns_device_allocation(&host_memory, sizeof(float)));
        }
    }

    TEST_F(HipNativeAtomics, ExternalDeviceBackingIsNotOwned) {
        void* memory = nullptr;
        ASSERT_EQ(cudaMalloc(&memory, 4096), cudaSuccess);
        const std::shared_ptr<void> owner(memory, [](void* ptr) { cudaFree(ptr); });
        int device = -1;
        ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
        RasterizerMemoryArena arena;
        EXPECT_FALSE(arena.owns_device_allocation(memory, sizeof(float)));
        RasterizerMemoryArena::ExternalBacking backing{
            .device_ptr = memory,
            .size = 4096,
            .device = device,
            .owner = owner,
            .label = "native-atomic-test.external"};
        ASSERT_TRUE(arena.install_external_backing(std::move(backing)));
        ArenaFrame frame(arena);
        auto allocate = frame.allocator();
        auto* imported = allocate(256);
        ASSERT_NE(imported, nullptr);
        EXPECT_FALSE(arena.owns_device_allocation(imported, sizeof(float)));
    }

    TEST_F(HipNativeAtomics, ContendedSignedAndSmallNormalSumsMatchSafePath) {
        constexpr int count = 8192;
        for (const bool vmm : {false, true}) {
            RasterizerMemoryArena::Config config;
            config.enable_vmm = vmm;
            RasterizerMemoryArena arena(config);
            ArenaFrame frame(arena);
            auto allocate = frame.allocator();
            auto* sums = reinterpret_cast<float*>(allocate(2 * sizeof(float)));
            ASSERT_NE(sums, nullptr);
            ASSERT_TRUE(arena.owns_device_allocation(sums, 2 * sizeof(float)));
            for (const float value : {0.125f, -0.125f, 1.0e-30f, -1.0e-30f}) {
                for (const bool alternate : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "vmm=" << vmm << " value=" << value
                                                      << " alternate=" << alternate);
                    ASSERT_EQ(cudaMemset(sums, 0, 2 * sizeof(float)), cudaSuccess);
                    ASSERT_EQ(launch_native_gradient_atomic_probe(sums, count, value, alternate, nullptr),
                              cudaSuccess);
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    std::array<float, 2> actual{};
                    ASSERT_EQ(cudaMemcpy(actual.data(), sums, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
                    const float expected = alternate ? 0.0f : count * value;
                    const float tolerance = std::fabs(count * value) * 3.0e-4f;
                    EXPECT_TRUE(std::isfinite(actual[0]));
                    EXPECT_TRUE(std::isfinite(actual[1]));
                    EXPECT_NEAR(actual[0], expected, tolerance);
                    EXPECT_NEAR(actual[1], expected, tolerance);
                    EXPECT_NEAR(actual[1], actual[0], tolerance);
                }
            }
        }
    }
} // namespace
