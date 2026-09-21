/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;

    Tensor upload(const std::vector<float>& values, const std::vector<size_t>& shape) {
        return Tensor::from_vector(values, TensorShape(shape), Device::CUDA).to(DataType::Float16);
    }

    std::vector<float> download(const Tensor& value) {
        return value.to(DataType::Float32).to(Device::CPU).contiguous().to_vector();
    }

    // Exactly representable FP16 operands isolate accumulation/epilogue error
    // from CPU-vs-GPU input conversion, with deterministic signs and magnitudes.
    std::vector<float> values(size_t count, unsigned seed) {
        std::mt19937 random(seed);
        std::uniform_int_distribution<int> distribution(-128, 128);
        std::vector<float> result(count);
        for (auto& value : result) {
            value = static_cast<float>(distribution(random)) / 256.0f;
        }
        return result;
    }

    float activate(float value, nn::Activation activation) {
        switch (activation) {
        case nn::Activation::Relu:
            return std::max(value, 0.0f);
        case nn::Activation::GeluTanh:
            return 0.5f * value * (1.0f + std::tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
        case nn::Activation::GeluErf:
            return 0.5f * value * (1.0f + std::erf(value * 0.7071067811865476f));
        case nn::Activation::Silu:
            return value / (1.0f + std::exp(-value));
        default:
            return value;
        }
    }

    struct Stream {
        cudaStream_t value = nullptr;
        ~Stream() {
            if (value) {
                cudaStreamSynchronize(value);
                cudaStreamDestroy(value);
            }
        }
    };
} // namespace

TEST(NnGemm, TailsBatchesLayoutsAndFusedEpilogues) {
    Stream stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), cudaSuccess);
    CUDAStreamGuard guard(stream.value);
    // Cover all three tile sizes, K tails, and the small scalar fallback.
    for (const auto shape : {std::array{7, 13, 9}, std::array{17, 19, 31},
                             std::array{63, 67, 65}, std::array{129, 133, 97},
                             std::array{33, 35, 3072}}) {
        const auto [m, n, k] = shape;
        for (bool trans_b : {false, true}) {
            for (bool broadcast_b : {false, true}) {
                const int batch = 3;
                auto a = values(batch * m * k, 19);
                auto b = values((broadcast_b ? 1 : batch) * n * k, 23);
                auto bias = values(n, 29);
                auto scale = values(n, 31);
                auto residual = values(batch * m * n, 37);
                auto A = upload(a, {batch, size_t(m), size_t(k)});
                auto B = upload(b, {size_t(broadcast_b ? 1 : batch), size_t(trans_b ? n : k), size_t(trans_b ? k : n)});
                auto Bias = upload(bias, {size_t(n)});
                auto Scale = upload(scale, {size_t(n)});
                auto Residual = upload(residual, {batch, size_t(m), size_t(n)});
                for (auto activation : {nn::Activation::None, nn::Activation::Relu,
                                        nn::Activation::GeluTanh, nn::Activation::GeluErf, nn::Activation::Silu}) {
                    SCOPED_TRACE(::testing::Message() << m << ',' << n << ',' << k << " NT=" << trans_b
                                                      << " broadcast=" << broadcast_b << " activation=" << int(activation));
                    auto output = nn::gemm(A, B, false, trans_b, &Bias, activation, &Residual, &Scale);
                    ASSERT_EQ(output.stream(), stream.value);
                    const auto actual = download(output);
                    float worst_excess = 0.0f;
                    for (int z = 0; z < batch; ++z) {
                        for (int row = 0; row < m; ++row) {
                            for (int col = 0; col < n; ++col) {
                                double sum = 0.0;
                                for (int t = 0; t < k; ++t) {
                                    sum += double(a[(z * m + row) * k + t]) *
                                           b[(broadcast_b ? 0 : z * n * k) + (trans_b ? col * k + t : t * n + col)];
                                }
                                const int index = (z * m + row) * n + col;
                                const float expected = activate(float(sum) + bias[col], activation) * scale[col] + residual[index];
                                ASSERT_TRUE(std::isfinite(actual[index]));
                                // FP16 output rounding is at most 0.049% for normals.
                                const float tolerance = 0.00003f + 0.0006f * std::abs(expected);
                                worst_excess = std::max(worst_excess, std::abs(actual[index] - expected) - tolerance);
                            }
                        }
                    }
                    EXPECT_EQ(worst_excess, 0.0f);
                }
            }
        }
    }
}

TEST(NnGemm, TransposedInputAndNchwOutput) {
    // Conv1x1 passes trans_a=true/trans_c=true directly to the GEMM kernel.
    // Check multiple images and partial tiles using an independent CPU product.
    constexpr int batch = 2, cin = 35, cout = 67, height = 7, width = 19;
    const auto x = values(batch * cin * height * width, 41);
    const auto w = values(cout * cin, 43);
    const auto bias = values(cout, 47);
    auto X = upload(x, {batch, cin, height, width});
    auto W = upload(w, {cout, cin, 1, 1});
    auto B = upload(bias, {cout});
    const auto actual = download(nn::conv2d(X, W, &B, nn::Conv2dParams{}));
    for (int image = 0; image < batch; ++image) {
        for (int out = 0; out < cout; ++out) {
            for (int pixel = 0; pixel < height * width; ++pixel) {
                double expected = bias[out];
                for (int in = 0; in < cin; ++in) {
                    expected += double(x[(image * cin + in) * height * width + pixel]) * w[out * cin + in];
                }
                ASSERT_NEAR(actual[(image * cout + out) * height * width + pixel], expected,
                            0.00003 + 0.0006 * std::abs(expected));
            }
        }
    }
}

TEST(NnGemm, WaitsForEpilogueOperandsOnAnotherStream) {
    Stream producer, consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer.value, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer.value, cudaStreamNonBlocking), cudaSuccess);
    CUDAStreamGuard guard(consumer.value);
    for (bool linear : {false, true}) {
        for (int role = 0; role < 3; ++role) {
            if (linear && role == 1)
                continue; // linear has no scale parameter
            SCOPED_TRACE(::testing::Message() << "linear=" << linear << " role=" << role);
            auto A = Tensor::ones({32, 32}, Device::CUDA, DataType::Float16);
            auto B = Tensor::ones({32, 32}, Device::CUDA, DataType::Float16);
            auto Bias = Tensor::ones({32}, Device::CUDA, DataType::Float16);
            auto Scale = Tensor::ones({32}, Device::CUDA, DataType::Float16);
            auto Residual = Tensor::ones({32, 32}, Device::CUDA, DataType::Float16);
            Tensor& delayed = role == 0 ? Bias : (role == 1 ? Scale : Residual);
            delayed = Tensor::zeros(delayed.shape(), Device::CUDA, DataType::Float16);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            uint16_t* host_data = nullptr;
            ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&host_data), delayed.bytes()), cudaSuccess);
            const std::unique_ptr<uint16_t, decltype(&cudaFreeHost)> host(host_data, cudaFreeHost);
            std::fill_n(host_data, delayed.numel(), uint16_t{0x3c00}); // FP16 1.0
            delayed.set_stream(producer.value);
            ASSERT_EQ(cudaLaunchHostFunc(producer.value, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }, nullptr), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(delayed.data_ptr(), host_data, delayed.bytes(),
                                      cudaMemcpyHostToDevice, producer.value),
                      cudaSuccess);
            auto output = linear ? nn::linear(A, B, &Bias, nn::Activation::None, &Residual)
                                 : nn::gemm(A, B, false, true, &Bias, nn::Activation::None, &Residual, &Scale);
            const auto actual = download(output);
            EXPECT_TRUE(std::all_of(actual.begin(), actual.end(), [](float v) { return v == 34.0f; }))
                << "first value=" << actual.front();
            ASSERT_EQ(cudaStreamSynchronize(producer.value), cudaSuccess);
        }
    }
}

TEST(NnGemm, ConvolutionWaitsForBiasOnAnotherStream) {
    Stream producer, consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer.value, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer.value, cudaStreamNonBlocking), cudaSuccess);
    CUDAStreamGuard guard(consumer.value);
    for (int mode = 0; mode < 4; ++mode) {
        SCOPED_TRACE(mode);
        const bool transpose = mode >= 2;
        const int kernel = mode == 0 ? 1 : (mode == 3 ? 3 : 2);
        nn::Conv2dParams params;
        params.groups = mode == 1 ? 2 : 1;
        params.stride_h = params.stride_w = mode == 2 ? 2 : 1;
        auto X = Tensor::ones({1, 32, 9, 17}, Device::CUDA, DataType::Float16);
        auto W = Tensor::ones({32, size_t(32 / params.groups), size_t(kernel), size_t(kernel)}, Device::CUDA, DataType::Float16);
        auto Bias = Tensor::ones({32}, Device::CUDA, DataType::Float16);
        auto run = [&] { return transpose ? nn::conv_transpose2d(X, W, &Bias, params)
                                          : nn::conv2d(X, W, &Bias, params); };
        const auto expected = download(run());
        Bias = Tensor::zeros({32}, Device::CUDA, DataType::Float16);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        uint16_t* host_data = nullptr;
        ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&host_data), Bias.bytes()), cudaSuccess);
        const std::unique_ptr<uint16_t, decltype(&cudaFreeHost)> host(host_data, cudaFreeHost);
        std::fill_n(host_data, Bias.numel(), uint16_t{0x3c00});
        Bias.set_stream(producer.value);
        ASSERT_EQ(cudaLaunchHostFunc(producer.value, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }, nullptr), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(Bias.data_ptr(), host_data, Bias.bytes(), cudaMemcpyHostToDevice,
                                  producer.value),
                  cudaSuccess);
        EXPECT_EQ(download(run()), expected);
        ASSERT_EQ(cudaStreamSynchronize(producer.value), cudaSuccess);
    }
}

// Opt-in GPU-event benchmark of the public operator (including output
// allocation and its fused epilogue). Run separate processes with/without
// LFS_DISABLE_HIP_WMMA=1, alternating their order; each process warms every shape.
TEST(NnGemm, DISABLED_ProductionShapesBenchmark) {
    const char* path = std::getenv("LFS_GEMM_BENCH_OUTPUT");
    ASSERT_NE(path, nullptr) << "Set LFS_GEMM_BENCH_OUTPUT to an output JSON file";
    struct Shape {
        const char* name;
        int m, n, k;
        bool trans_a, trans_b;
    };
    const Shape shapes[] = {
        {"moge_qkv", 1370, 2304, 768, false, true},
        {"moge_projection", 1370, 768, 768, false, true},
        {"moge_mlp_up", 1370, 3072, 768, false, true},
        {"moge_mlp_down", 1370, 768, 3072, false, true},
        {"decoder_1x1", 4096, 256, 256, true, true},
        {"nn_layout", 1370, 768, 768, false, false},
        {"small", 33, 65, 127, false, true},
    };
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    nlohmann::json results = {{"device", properties.name}, {"warmup", 5}, {"iterations", 20}, {"wmma_disabled", std::getenv("LFS_DISABLE_HIP_WMMA") != nullptr}, {"cases", nlohmann::json::array()}};
    cudaEvent_t start = nullptr, stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
    for (const auto& s : shapes) {
        auto A = upload(values(size_t(s.m) * s.k, 53), {size_t(s.trans_a ? s.k : s.m), size_t(s.trans_a ? s.m : s.k)});
        auto B = upload(values(size_t(s.n) * s.k, 59), {size_t(s.trans_b ? s.n : s.k), size_t(s.trans_b ? s.k : s.n)});
        auto Bias = upload(values(s.n, 61), {size_t(s.n)});
        Tensor output;
        auto run = [&] { output = nn::gemm(A, B, s.trans_a, s.trans_b, &Bias, nn::Activation::GeluTanh); };
        for (int i = 0; i < 5; ++i)
            run();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
        for (int i = 0; i < 20; ++i)
            run();
        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
        float elapsed = 0.0f;
        ASSERT_EQ(cudaEventElapsedTime(&elapsed, start, stop), cudaSuccess);
        const float ms = elapsed / 20;
        const double tflops = 2.0 * s.m * s.n * s.k / (ms * 1e9);
        std::cout << s.name << " " << ms << " ms " << tflops << " TFLOP/s\n";
        results["cases"].push_back({{"name", s.name}, {"m", s.m}, {"n", s.n}, {"k", s.k}, {"trans_a", s.trans_a}, {"trans_b", s.trans_b}, {"ms", ms}, {"tflops", tflops}});
        // Read back to surface asynchronous errors and prevent dead work.
        const auto host = download(output);
        ASSERT_TRUE(std::all_of(host.begin(), host.end(), [](float v) { return std::isfinite(v); }));
        // Check spatially separated outputs (including both extreme corners)
        // against a double-precision CPU reference at production K lengths.
        const auto av = values(size_t(s.m) * s.k, 53);
        const auto bv = values(size_t(s.n) * s.k, 59);
        const auto bias = values(s.n, 61);
        for (size_t sample = 0; sample < 64; ++sample) {
            const size_t index = sample * (host.size() - 1) / 63;
            const size_t row = index / s.n, col = index % s.n;
            double sum = 0.0;
            for (size_t t = 0; t < size_t(s.k); ++t) {
                sum += double(av[s.trans_a ? t * s.m + row : row * s.k + t]) *
                       bv[s.trans_b ? col * s.k + t : t * s.n + col];
            }
            const float expected = activate(float(sum) + bias[col], nn::Activation::GeluTanh);
            ASSERT_NEAR(host[index], expected, 0.00003f + 0.0006f * std::abs(expected))
                << s.name << " at " << row << ',' << col;
        }
    }
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    std::ofstream file(path);
    ASSERT_TRUE(file.good());
    file << results.dump(2) << '\n';
}
