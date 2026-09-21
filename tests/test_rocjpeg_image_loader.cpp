/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/pipelined_image_loader.hpp"
#include "io/rocjpeg_image_loader.hpp"

#include <gtest/gtest.h>
#include <jpeglib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <process.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::io;

    void write_jpeg_variant(const std::filesystem::path& path, const Tensor& image, bool progressive, bool grayscale = false) {
        jpeg_compress_struct encoder{};
        jpeg_error_mgr error{};
        encoder.err = jpeg_std_error(&error);
        jpeg_create_compress(&encoder);
        unsigned char* encoded = nullptr;
        unsigned long size = 0;
        jpeg_mem_dest(&encoder, &encoded, &size);
        encoder.image_width = static_cast<JDIMENSION>(image.shape()[1]);
        encoder.image_height = static_cast<JDIMENSION>(image.shape()[0]);
        encoder.input_components = 3;
        encoder.in_color_space = JCS_RGB;
        jpeg_set_defaults(&encoder);
        jpeg_set_quality(&encoder, 95, TRUE);
        if (grayscale) {
            jpeg_set_colorspace(&encoder, JCS_GRAYSCALE);
        } else if (progressive) {
            jpeg_simple_progression(&encoder);
        } else {
            for (int i = 0; i < 3; ++i)
                encoder.comp_info[i].h_samp_factor = encoder.comp_info[i].v_samp_factor = 1;
        }
        jpeg_start_compress(&encoder, TRUE);
        while (encoder.next_scanline < encoder.image_height) {
            auto* row = const_cast<JSAMPLE*>(image.ptr<uint8_t>() +
                                             encoder.next_scanline * encoder.image_width * 3);
            jpeg_write_scanlines(&encoder, &row, 1);
        }
        jpeg_finish_compress(&encoder);
        jpeg_destroy_compress(&encoder);
        const std::unique_ptr<unsigned char, decltype(&std::free)> bytes(encoded, std::free);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(encoded), size);
        if (!file)
            throw std::runtime_error("Cannot write JPEG test fixture");
    }

    class DisableRocJpeg {
    public:
        explicit DisableRocJpeg(const bool disabled) {
            if (const auto* value = std::getenv("LFS_DISABLE_ROCJPEG"))
                previous_ = value;
            if (_putenv_s("LFS_DISABLE_ROCJPEG", disabled ? "1" : "0") != 0)
                throw std::runtime_error("Cannot configure rocJPEG test environment");
        }
        ~DisableRocJpeg() {
            _putenv_s("LFS_DISABLE_ROCJPEG", previous_ ? previous_->c_str() : "");
        }

    private:
        std::optional<std::string> previous_;
    };

    class RocJpegLoaderTest : public ::testing::Test {
    protected:
        static constexpr size_t width = 513;
        static constexpr size_t height = 257;

        void SetUp() override {
            int devices = 0;
            ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
            ASSERT_GT(devices, 0) << "This suite requires a HIP device";
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_rocjpeg_test_" + std::to_string(_getpid()) + "_" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            ASSERT_TRUE(std::filesystem::create_directory(directory_));
            jpeg_path_ = directory_ / "gradient.jpg";
            png_path_ = directory_ / "gradient.png";
            jpeg444_path_ = directory_ / "gradient444.jpg";
            progressive_path_ = directory_ / "progressive.jpg";
            grayscale_path_ = directory_ / "grayscale.jpg";

            auto image = Tensor::empty({height, width, size_t{3}}, Device::CPU, DataType::UInt8);
            auto* pixels = image.ptr<uint8_t>();
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    const size_t offset = 3 * (y * width + x);
                    pixels[offset] = static_cast<uint8_t>(32 + 160 * x / (width - 1));
                    pixels[offset + 1] = static_cast<uint8_t>(48 + 128 * y / (height - 1));
                    pixels[offset + 2] = static_cast<uint8_t>(80 + 24 * (x + y) / (width + height - 2));
                }
            }
            save_image_u8(jpeg_path_, image, 95);
            save_image_u8(png_path_, image);
            write_jpeg_variant(jpeg444_path_, image, false);
            write_jpeg_variant(progressive_path_, image, true);
            write_jpeg_variant(grayscale_path_, image, false, true);
            std::ifstream file(jpeg_path_, std::ios::binary);
            jpeg_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            ASSERT_GT(jpeg_.size(), 3u);
        }

        void TearDown() override {
            if (!directory_.empty()) {
                std::error_code error;
                std::filesystem::remove_all(directory_, error);
                EXPECT_FALSE(error) << error.message();
            }
        }

        static PipelinedLoaderConfig config() {
            PipelinedLoaderConfig result;
            result.io_threads = 1;
            result.cold_process_threads = 1;
            result.decoder_pool_size = 1;
            result.prefetch_count = 2;
            result.output_queue_size = 2;
            result.max_cache_bytes = 16 * 1024 * 1024;
            return result;
        }

        static void expect_image(const Tensor& image, size_t h, size_t w, DataType dtype) {
            ASSERT_TRUE(image.is_valid());
            EXPECT_EQ(image.device(), Device::CUDA);
            EXPECT_EQ(image.dtype(), dtype);
            EXPECT_EQ(image.shape(), TensorShape({3, h, w}));
        }

        std::filesystem::path directory_, jpeg_path_, png_path_, jpeg444_path_, progressive_path_, grayscale_path_;
        std::vector<uint8_t> jpeg_;
    };

    TEST_F(RocJpegLoaderTest, HardwareDecodeMatchesCpuRgbForOddDimensions) {
        DisableRocJpeg enabled(false);
        RocJpegImageLoader loader;
        ASSERT_TRUE(loader.available()) << "D3D11 MJPEG hardware is required for this opt-in suite";
        const auto decoded = loader.decode(jpeg_, 1, 0, true);
        ASSERT_NO_FATAL_FAILURE(expect_image(decoded, height, width, DataType::UInt8));
        const auto actual = decoded.to_vector_uint8();
        auto [pixels, cpu_width, cpu_height, channels] = load_image(jpeg_path_, 1, 0);
        const std::unique_ptr<unsigned char, decltype(&free_image)> reference(pixels, free_image);
        ASSERT_NE(pixels, nullptr);
        ASSERT_EQ(cpu_width, width);
        ASSERT_EQ(cpu_height, height);
        ASSERT_EQ(channels, 3);
        int maximum_error = 0;
        double total_error = 0;
        for (size_t i = 0; i < width * height; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                const int error = std::abs(int(actual[c * width * height + i]) - int(pixels[3 * i + c]));
                maximum_error = std::max(maximum_error, error);
                total_error += error;
            }
        }
        // Hardware and libjpeg use different chroma upsampling and IDCT rounding.
        EXPECT_LE(maximum_error, 5);
        EXPECT_LE(total_error / actual.size(), 1.5);
    }

    TEST_F(RocJpegLoaderTest, ResizeAndNormalizedFloatAgreeWithUint8Output) {
        DisableRocJpeg enabled(false);
        RocJpegImageLoader loader;
        ASSERT_TRUE(loader.available());
        struct ResizeCase {
            int factor, maximum;
            size_t h, w;
        };
        for (const auto& test : std::array<ResizeCase, 3>{{{1, 0, height, width}, {2, 0, 128, 256}, {2, 96, 48, 96}}}) {
            SCOPED_TRACE(test.maximum);
            const auto bytes = loader.decode(jpeg_, test.factor, test.maximum, true);
            const auto floats = loader.decode(jpeg_, test.factor, test.maximum, false);
            ASSERT_NO_FATAL_FAILURE(expect_image(bytes, test.h, test.w, DataType::UInt8));
            ASSERT_NO_FATAL_FAILURE(expect_image(floats, test.h, test.w, DataType::Float32));
            const auto byte_values = bytes.to_vector_uint8();
            const auto float_values = floats.to_vector();
            ASSERT_EQ(byte_values.size(), float_values.size());
            float maximum_error = 0;
            for (size_t i = 0; i < byte_values.size(); ++i) {
                ASSERT_TRUE(std::isfinite(float_values[i]));
                maximum_error = std::max(maximum_error, std::abs(float_values[i] - byte_values[i] / 255.0f));
            }
            EXPECT_LE(maximum_error, 0.501f / 255.0f);
        }
    }

    TEST_F(RocJpegLoaderTest, ResizedTensorSurvivesDecoderAndPoolReuse) {
        DisableRocJpeg enabled(false);
        for (int iteration = 0; iteration < 3; ++iteration) {
            Tensor survivor;
            std::vector<float> expected;
            {
                RocJpegImageLoader loader;
                ASSERT_TRUE(loader.available());
                survivor = loader.decode(jpeg_, 2, 96, false);
                ASSERT_NO_FATAL_FAILURE(expect_image(survivor, 48, 96, DataType::Float32));
                expected = survivor.to_vector();
            }
            // Both the returned allocation and cached intermediates must stop
            // referring to the decoder's destroyed private stream.
            EXPECT_EQ(survivor.stream(), nullptr);
            EXPECT_EQ(survivor.clone().to_vector(), expected);
            const auto shape = survivor.shape();
            survivor = {};
            auto reused = Tensor::empty(shape, Device::CUDA, DataType::Float32);
            ASSERT_EQ(cudaMemset(reused.ptr<float>(), 0, reused.numel() * sizeof(float)), cudaSuccess);
            const auto zeros = reused.to_vector();
            EXPECT_TRUE(std::all_of(zeros.begin(), zeros.end(), [](float value) { return value == 0; }));
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
    }

    TEST_F(RocJpegLoaderTest, ReusesExclusiveOutputAcrossDecodeAndResizeChanges) {
        DisableRocJpeg enabled(false);
        RocJpegImageLoader loader;
        ASSERT_TRUE(loader.available());
        Tensor storage;
        for (const bool uint8 : {true, false, true}) {
            for (const int factor : {1, 2, 4, 1}) {
                const auto reference = loader.decode(jpeg_, factor, 0, uint8).to_vector();
                const auto first = loader.decode(jpeg_, factor, 0, uint8, &storage);
                const auto pointer = first.data_ptr();
                for (int repeat = 0; repeat < 5; ++repeat) {
                    const auto decoded = loader.decode(jpeg_, factor, 0, uint8, &storage);
                    EXPECT_EQ(decoded.data_ptr(), pointer);
                    EXPECT_EQ(decoded.to_vector(), reference);
                }
            }
        }
    }

    TEST_F(RocJpegLoaderTest, PrefetchKeepsLeasedOutputAliveUntilConsumerRelease) {
        DisableRocJpeg enabled(false);
        auto settings = config();
        settings.decode_frame_ring_capacity = 4;
        PipelinedImageLoader loader(settings);
        for (const int factor : {1, 2}) {
            LoadParams params;
            params.output_uint8 = true;
            params.resize_factor = factor;
            loader.prefetch(0, jpeg_path_, params);
            auto held = loader.try_get_for(std::chrono::seconds(10));
            ASSERT_TRUE(held);
            ASSERT_TRUE(held->error.empty()) << held->error;
            ASSERT_FALSE(held->decoded_frame_leases.empty());
            const auto expected = held->tensor.to_vector_uint8();
            for (size_t sequence = 1; sequence <= 12; ++sequence) {
                loader.prefetch(sequence, jpeg_path_, params);
                auto ready = loader.try_get_for(std::chrono::seconds(10));
                ASSERT_TRUE(ready);
                ASSERT_TRUE(ready->error.empty()) << ready->error;
                EXPECT_NE(held->tensor.data_ptr(), ready->tensor.data_ptr());
                EXPECT_EQ(held->tensor.to_vector_uint8(), expected);
                EXPECT_EQ(ready->tensor.to_vector_uint8(), expected);
            }
        }
    }

    TEST_F(RocJpegLoaderTest, ReuseWaitsForGpuConsumerAfterHostLeaseExpires) {
        DisableRocJpeg enabled(false);
        auto settings = config();
        settings.decode_frame_ring_capacity = 1;
        PipelinedImageLoader loader(settings);
        LoadParams params;
        params.output_uint8 = true;
        auto dark = Tensor::empty({height, width, 3}, Device::CPU, DataType::UInt8);
        std::fill_n(dark.ptr<uint8_t>(), dark.numel(), uint8_t{16});
        const auto dark_path = directory_ / "dark.jpg";
        save_image_u8(dark_path, dark, 95);
        auto snapshot = Tensor::empty({3, height, width}, Device::CUDA, DataType::UInt8);
        struct Consumer {
            cudaStream_t stream = nullptr;
            std::atomic<bool> entered{false}, proceed{false};
            ~Consumer() {
                proceed.store(true);
                if (stream) {
                    (void)cudaStreamSynchronize(stream);
                    CudaMemoryPool::instance().release_stream(stream);
                    (void)cudaStreamDestroy(stream);
                }
            }
        } consumer;
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumer.stream, cudaStreamNonBlocking), cudaSuccess);
        loader.prefetch(0, jpeg_path_, params);
        auto first = loader.try_get_for(std::chrono::seconds(10));
        ASSERT_TRUE(first);
        ASSERT_TRUE(first->error.empty()) << first->error;
        const auto expected = first->tensor.to_vector_uint8();
        const auto pointer = first->tensor.data_ptr();
        first->tensor.set_stream(consumer.stream);
        ASSERT_EQ(cudaLaunchHostFunc(consumer.stream, [](void* data) {
            auto& gate = *static_cast<Consumer*>(data);
            gate.entered.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!gate.proceed.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield(); }, &consumer), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(snapshot.data_ptr(), pointer, snapshot.bytes(), cudaMemcpyDeviceToDevice,
                                  consumer.stream),
                  cudaSuccess);
        first.reset();
        loader.prefetch(1, dark_path, params);
        auto early = loader.try_get_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(early) << "Producer reused a slot before its GPU consumer finished";
        consumer.proceed.store(true);
        ASSERT_EQ(cudaStreamSynchronize(consumer.stream), cudaSuccess);
        EXPECT_TRUE(consumer.entered.load());
        EXPECT_EQ(snapshot.to_vector_uint8(), expected);
        auto second = early ? std::move(early) : loader.try_get_for(std::chrono::seconds(10));
        ASSERT_TRUE(second);
        ASSERT_TRUE(second->error.empty()) << second->error;
        EXPECT_EQ(second->tensor.data_ptr(), pointer);
        EXPECT_NE(second->tensor.to_vector_uint8(), expected);
    }

    TEST_F(RocJpegLoaderTest, RejectsUnsupportedAndMalformedInputsWithoutPoisoningDecoder) {
        DisableRocJpeg enabled(false);
        RocJpegImageLoader loader;
        ASSERT_TRUE(loader.available());
        const std::array<uint8_t, 8> png_header{0x89, 'P', 'N', 'G', 13, 10, 26, 10};
        const std::array<uint8_t, 4> truncated_jpeg{0xff, 0xd8, 0xff, 0xe0};
        EXPECT_FALSE(loader.decode({}, 1, 0, true).is_valid());
        EXPECT_FALSE(loader.decode(png_header, 1, 0, true).is_valid());
        EXPECT_FALSE(loader.decode(truncated_jpeg, 1, 0, true).is_valid());
        EXPECT_TRUE(loader.decode(jpeg_, 1, 0, true).is_valid());
    }

    TEST_F(RocJpegLoaderTest, ImmediateThenPrefetchCompletesWithHardwareOrCpuFallback) {
        for (const bool disabled : {false, true}) {
            SCOPED_TRACE(disabled);
            DisableRocJpeg setting(disabled);
            for (const bool uint8 : {false, true}) {
                SCOPED_TRACE(uint8);
                PipelinedImageLoader loader(config());
                LoadParams params;
                params.output_uint8 = uint8;
                const auto immediate = loader.load_image_immediate(jpeg_path_, params);
                ASSERT_NO_FATAL_FAILURE(expect_image(immediate, height, width, uint8 ? DataType::UInt8 : DataType::Float32));
                const auto expected = immediate.to_vector();
                for (size_t sequence = 0; sequence < 2; ++sequence) {
                    loader.prefetch(sequence, jpeg_path_, params);
                    const auto ready = loader.try_get_for(std::chrono::seconds(10));
                    ASSERT_TRUE(ready) << "JPEG entered a hot queue without a decoder consumer";
                    ASSERT_TRUE(ready->error.empty()) << ready->error;
                    EXPECT_EQ(ready->sequence_id, sequence);
                    ASSERT_NO_FATAL_FAILURE(expect_image(ready->tensor, height, width, uint8 ? DataType::UInt8 : DataType::Float32));
                    EXPECT_EQ(ready->tensor.to_vector(), expected);
                }
                const auto stats = loader.get_stats();
                EXPECT_EQ(stats.hot_path_hits, 0u);
                EXPECT_EQ(stats.jpeg_cache_entries, 0u);
                EXPECT_EQ(stats.cold_path_misses, 2u);
                EXPECT_EQ(stats.rocjpeg_decode_calls, disabled ? 0u : 3u);
                EXPECT_EQ(stats.cpu_decode_calls, disabled ? 3u : 0u);
            }
        }
    }

    TEST_F(RocJpegLoaderTest, PngUsesCpuFallbackWithRequestedShapeAndType) {
        DisableRocJpeg enabled(false);
        for (const bool uint8 : {false, true}) {
            PipelinedImageLoader loader(config());
            LoadParams params;
            params.resize_factor = 2;
            params.max_width = 96;
            params.output_uint8 = uint8;
            loader.prefetch(4, png_path_, params);
            const auto ready = loader.try_get_for(std::chrono::seconds(10));
            ASSERT_TRUE(ready);
            ASSERT_TRUE(ready->error.empty()) << ready->error;
            ASSERT_NO_FATAL_FAILURE(expect_image(ready->tensor, 48, 96, uint8 ? DataType::UInt8 : DataType::Float32));
            const auto stats = loader.get_stats();
            EXPECT_EQ(stats.cpu_decode_calls, 1u);
            EXPECT_EQ(stats.rocjpeg_decode_calls, 0u);
        }
    }

    TEST_F(RocJpegLoaderTest, UnsupportedJpegVariantsUseCpuFallback) {
        DisableRocJpeg enabled(false);
        RocJpegImageLoader hardware;
        ASSERT_TRUE(hardware.available());
        for (const auto& path : {jpeg444_path_, progressive_path_, grayscale_path_}) {
            SCOPED_TRACE(path.filename().string());
            std::ifstream file(path, std::ios::binary);
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                             std::istreambuf_iterator<char>());
            EXPECT_FALSE(hardware.decode(bytes, 1, 0, true).is_valid());
            for (const bool uint8 : {false, true}) {
                PipelinedImageLoader loader(config());
                LoadParams params;
                params.resize_factor = 2;
                params.max_width = 96;
                params.output_uint8 = uint8;
                const auto immediate = loader.load_image_immediate(path, params);
                ASSERT_NO_FATAL_FAILURE(expect_image(immediate, 48, 96, uint8 ? DataType::UInt8 : DataType::Float32));
                loader.prefetch(8, path, params);
                const auto ready = loader.try_get_for(std::chrono::seconds(10));
                ASSERT_TRUE(ready);
                ASSERT_TRUE(ready->error.empty()) << ready->error;
                ASSERT_NO_FATAL_FAILURE(expect_image(ready->tensor, 48, 96, uint8 ? DataType::UInt8 : DataType::Float32));
                EXPECT_EQ(ready->tensor.to_vector(), immediate.to_vector());
                EXPECT_EQ(loader.get_stats().cpu_decode_calls, 2u);
                EXPECT_EQ(loader.get_stats().rocjpeg_decode_calls, 0u);
            }
        }
    }
} // namespace
