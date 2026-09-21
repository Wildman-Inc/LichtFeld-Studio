/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/image_io.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    using namespace lfs::core;

    struct ResizeCase {
        int width, height, divisor, max_width, out_width, out_height;
    };

    class ImageLanczosTest : public ::testing::TestWithParam<ResizeCase> {
    protected:
        void SetUp() override {
            int devices = 0;
            ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
            ASSERT_GT(devices, 0);
            static std::atomic_uint64_t sequence{0};
            path_ = std::filesystem::temp_directory_path() /
                    ("lfs_lanczos_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                     "_" + std::to_string(sequence.fetch_add(1)) + ".png");
        }

        void TearDown() override {
            std::error_code error;
            std::filesystem::remove(path_, error);
            EXPECT_FALSE(error) << error.message();
        }

        template <typename T, int Channels>
        void compare_with_gpu() {
            const auto test = GetParam();
            constexpr int maximum = std::numeric_limits<T>::max();
            const size_t input_pixels = static_cast<size_t>(test.width) * test.height;
            std::vector<T> pixels(input_pixels * Channels);
            for (int y = 0; y < test.height; ++y) {
                for (int x = 0; x < test.width; ++x) {
                    const size_t i = (static_cast<size_t>(y) * test.width + x) * Channels;
                    // Checkerboard, sharp edge, impulse/noise, and constant alpha
                    // expose aliasing, ringing, normalization and border mistakes.
                    pixels[i] = (x + y) % 2 ? maximum : 0;
                    pixels[i + 1] = x < test.width / 2 ? 0 : maximum;
                    pixels[i + 2] = (x == 0 || y == 0) ? maximum : (x * 997 + y * 167) % (maximum + 1);
                    if constexpr (Channels == 4)
                        pixels[i + 3] = 173;
                }
            }
            ASSERT_TRUE(save_png(path_, pixels.data(), test.width, test.height, Channels, sizeof(T) * 8, 1));
            auto [resized, w, h, channels] = [&] {
                if constexpr (std::is_same_v<T, uint16_t>)
                    return load_image_u16(path_, test.divisor, test.max_width);
                else if constexpr (Channels == 4)
                    return load_image_with_alpha(path_, test.divisor, test.max_width);
                else
                    return load_image(path_, test.divisor, test.max_width);
            }();
            const std::unique_ptr<T, decltype(&free_image)> result(resized, free_image);
            ASSERT_NE(resized, nullptr);
            ASSERT_EQ(w, test.out_width);
            ASSERT_EQ(h, test.out_height);
            ASSERT_EQ(channels, Channels);

            // Identical lossless source pixels isolate resizing from JPEG IDCT
            // and chroma reconstruction differences between the two decoders.
            auto input = Tensor::empty({static_cast<size_t>(test.height), static_cast<size_t>(test.width), 3},
                                       Device::CPU, std::is_same_v<T, uint8_t> ? DataType::UInt8 : DataType::Float32);
            for (size_t i = 0; i < input_pixels; ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    if constexpr (std::is_same_v<T, uint8_t>)
                        input.ptr<uint8_t>()[3 * i + c] = pixels[Channels * i + c];
                    else
                        input.ptr<float>()[3 * i + c] = pixels[Channels * i + c] / static_cast<float>(maximum);
                }
            }
            const auto gpu = lanczos_resize(input.to(Device::CUDA), h, w, 2).to_vector();
            const size_t output_pixels = static_cast<size_t>(w) * h;
            ASSERT_EQ(gpu.size(), output_pixels * 3);
            int maximum_error = 0;
            float maximum_unrounded_error = 0;
            double total_error = 0;
            for (size_t i = 0; i < output_pixels; ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    const float value = gpu[c * output_pixels + i];
                    ASSERT_TRUE(std::isfinite(value));
                    const float scaled = std::clamp(value, 0.0f, 1.0f) * maximum;
                    const int expected = static_cast<int>(std::lround(scaled));
                    const int error = std::abs(int(resized[Channels * i + c]) - expected);
                    maximum_error = std::max(maximum_error, error);
                    maximum_unrounded_error = std::max(maximum_unrounded_error, std::abs(resized[Channels * i + c] - scaled));
                    total_error += error;
                }
                if constexpr (Channels == 4)
                    EXPECT_EQ(resized[Channels * i + 3], 173);
            }
            // Separable CPU and 2D GPU accumulation may straddle a rounding boundary.
            // A checkerboard deliberately produces many half-integer outputs, so
            // compare against the unrounded signal as well as quantized pixels.
            EXPECT_LE(maximum_error, 1);
            EXPECT_LE(maximum_unrounded_error, 0.5f + maximum * 2e-6f);
            RecordProperty("max_pixel_error", maximum_error);
            RecordProperty("max_unrounded_error", maximum_unrounded_error);
            RecordProperty("mean_pixel_error", total_error / (output_pixels * 3));
        }

        std::filesystem::path path_;
    };

    TEST_P(ImageLanczosTest, Rgb8MatchesGpu) {
        compare_with_gpu<uint8_t, 3>();
    }

    TEST_P(ImageLanczosTest, Rgb16MatchesGpu) {
        compare_with_gpu<uint16_t, 3>();
    }

    TEST_P(ImageLanczosTest, RgbaMatchesGpuAndPreservesAlpha) {
        compare_with_gpu<uint8_t, 4>();
    }

    INSTANTIATE_TEST_SUITE_P(Downscale, ImageLanczosTest, ::testing::Values(ResizeCase{97, 65, 2, 0, 48, 32}, ResizeCase{97, 65, 4, 0, 24, 16}, ResizeCase{97, 65, 8, 0, 12, 8}, ResizeCase{97, 65, 2, 17, 17, 11}, ResizeCase{65, 97, 1, 23, 15, 23}, ResizeCase{17, 1, 2, 0, 8, 1}, ResizeCase{1, 17, 8, 0, 1, 2}, ResizeCase{13, 9, 1, 1, 1, 1}));
} // namespace
