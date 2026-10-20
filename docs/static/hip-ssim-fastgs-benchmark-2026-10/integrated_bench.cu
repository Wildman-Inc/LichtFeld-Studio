#include "core/cuda/memory_arena.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/rasterization/fastgs/rasterization/include/kernels_backward.cuh"

#include <cuda_runtime.h>
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using lfs::core::RasterizerMemoryArena;
using lfs::core::Tensor;
using lfs::core::TensorShape;
using lfs::core::Device;
using lfs::training::kernels::FusedL1SSIMWorkspace;
using lfs::training::kernels::fused_l1_ssim_forward;
using lfs::training::kernels::fused_l1_ssim_backward;
using lfs::core::getCurrentCUDAStream;
using fast_lfs::rasterization::config::block_size_blend_backward;
using fast_lfs::rasterization::kernels::backward::blend_backward_cu;

unsigned kWidth = 489;
unsigned kHeight = 273;
constexpr unsigned kTileWidth = 16;
constexpr unsigned kTileHeight = 16;
unsigned kInstancesPerTile = 48;
unsigned kTilesX = (kWidth + kTileWidth - 1) / kTileWidth;
unsigned kTilesY = (kHeight + kTileHeight - 1) / kTileHeight;
unsigned kTileCount = kTilesX * kTilesY;
unsigned kClustersX = (kTilesX + 1) / 2;
unsigned kClustersY = (kTilesY + 1) / 2;
unsigned kPrimitives = kClustersX * kClustersY * kInstancesPerTile;
unsigned kInstances = kTileCount * kInstancesPerTile;
size_t kPixels = static_cast<size_t>(kWidth) * kHeight;
size_t kTilePixelStates = static_cast<size_t>(kTileCount) * 256;
float g_transmittance_min = 1.0f;
float g_transmittance_max = 0.0f;
unsigned g_contributions_min = UINT32_MAX;
unsigned g_contributions_max = 0;
constexpr std::array<double, 11> kSsimWeights{
    0.001028380123898387, 0.0075987582094967365, 0.036000773310661316,
    0.10936068743467331, 0.21300552785396576, 0.26601171493530273,
    0.21300552785396576, 0.10936068743467331, 0.036000773310661316,
    0.0075987582094967365, 0.001028380123898387};

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

template <typename T>
T* arena_array(const std::function<char*(size_t)>& allocate, size_t count) {
    return reinterpret_cast<T*>(allocate(count * sizeof(T)));
}

template <typename T>
void upload(T* dst, const std::vector<T>& src, const char* what) {
    check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
}

template <bool Native>
void launch_variant(
    const uint2* ranges, const uint* instance_ids, const float2* means,
    const float4* conic_opacity, const float3* colors, const float* grad_image,
    const float* grad_alpha, const float* image, const float* alpha,
    const uint* contributions, const float* transmittance, float2* grad_mean,
    float* grad_conic, float* grad_opacity, float3* grad_color, unsigned width,
    unsigned height, unsigned grid_width, cudaStream_t stream) {
#if LFS_BENCH_HAS_NATIVE_ATOMICS
    blend_backward_cu<::DensificationType::None, Native>
        <<<kTileCount, block_size_blend_backward, 0, stream>>>(
            ranges, instance_ids, means, conic_opacity, colors, grad_image,
            grad_alpha, image, alpha, contributions, transmittance, grad_mean,
            grad_conic, grad_opacity, grad_color, nullptr, nullptr, nullptr,
            kInstances, kPrimitives, width, height, grid_width);
#else
    static_assert(!Native, "Native atomics do not exist in the base revision");
    blend_backward_cu<::DensificationType::None>
        <<<kTileCount, block_size_blend_backward, 0, stream>>>(
            ranges, instance_ids, means, conic_opacity, colors, grad_image,
            grad_alpha, image, alpha, contributions, transmittance, grad_mean,
            grad_conic, grad_opacity, grad_color, nullptr, nullptr, nullptr,
            kInstances, kPrimitives, width, height, grid_width);
#endif
}

struct OutputSet { float2* mean; float* conic; float* opacity; float3* color; };

void zero_outputs(const OutputSet& out, cudaStream_t stream) {
    check(cudaMemsetAsync(out.mean, 0, kPrimitives * sizeof(float2), stream), "zero mean grads");
    check(cudaMemsetAsync(out.conic, 0, 3 * kPrimitives * sizeof(float), stream), "zero conic grads");
    check(cudaMemsetAsync(out.opacity, 0, kPrimitives * sizeof(float), stream), "zero opacity grads");
    check(cudaMemsetAsync(out.color, 0, kPrimitives * sizeof(float3), stream), "zero color grads");
}

void invoke(bool native, const uint2* ranges, const uint* ids, const float2* means,
            const float4* conic, const float3* colors, const float* grad_image,
            const float* grad_alpha, const float* image, const float* alpha,
            const uint* contributions, const float* transmittance,
            const OutputSet& out, cudaStream_t stream) {
#if LFS_BENCH_HAS_NATIVE_ATOMICS
    if (native) {
        launch_variant<true>(ranges, ids, means, conic, colors, grad_image, grad_alpha,
            image, alpha, contributions, transmittance, out.mean, out.conic,
            out.opacity, out.color, kWidth, kHeight, kTilesX, stream);
        return;
    }
#else
    (void)native;
#endif
    launch_variant<false>(ranges, ids, means, conic, colors, grad_image, grad_alpha,
        image, alpha, contributions, transmittance, out.mean, out.conic,
        out.opacity, out.color, kWidth, kHeight, kTilesX, stream);
}

struct DeviceInputs {
    uint2* ranges{}; uint* ids{}; float2* means{}; float4* conic{}; float3* colors{};
    float* grad_image{}; float* grad_alpha{}; float* image{}; float* alpha{};
    uint* contributions{}; float* transmittance{};
};

DeviceInputs prepare_inputs(const std::function<char*(size_t)>& allocate, std::vector<float>& rendered_image) {
    DeviceInputs d;
    d.ranges = arena_array<uint2>(allocate, kTileCount);
    d.ids = arena_array<uint>(allocate, kInstances);
    d.means = arena_array<float2>(allocate, kPrimitives);
    d.conic = arena_array<float4>(allocate, kPrimitives);
    d.colors = arena_array<float3>(allocate, kPrimitives);
    d.grad_image = arena_array<float>(allocate, kPixels * 3);
    d.grad_alpha = arena_array<float>(allocate, kPixels);
    d.image = arena_array<float>(allocate, kPixels * 3);
    d.alpha = arena_array<float>(allocate, kPixels);
    d.contributions = arena_array<uint>(allocate, kPixels);
    d.transmittance = arena_array<float>(allocate, kTilePixelStates);

    std::vector<uint2> ranges(kTileCount);
    std::vector<uint> ids(kInstances);
    std::vector<float2> means(kPrimitives);
    std::vector<float4> conic(kPrimitives);
    std::vector<float3> colors(kPrimitives);
    for (unsigned cy = 0; cy < kClustersY; ++cy) {
        for (unsigned cx = 0; cx < kClustersX; ++cx) {
            const unsigned cluster = cy * kClustersX + cx;
            const unsigned base = cluster * kInstancesPerTile;
            const float center_x = static_cast<float>(cx * 2 * kTileWidth + kTileWidth);
            const float center_y = static_cast<float>(cy * 2 * kTileHeight + kTileHeight);
            for (unsigned j = 0; j < kInstancesPerTile; ++j) {
                const unsigned p = base + j;
                means[p] = make_float2(center_x + static_cast<float>(static_cast<int>(j % 5) - 2) * 0.35f,
                                       center_y + static_cast<float>(static_cast<int>(j % 7) - 3) * 0.30f);
                conic[p] = make_float4(0.0025f + 0.00002f * (j % 3), 0.0f,
                                       0.0025f + 0.00002f * (j % 5), 0.010f + 0.00001f * (j % 7));
                colors[p] = make_float3(0.25f + 0.01f * (j % 13),
                                        0.35f + 0.01f * (j % 11),
                                        0.45f + 0.01f * (j % 9));
            }
        }
    }
    for (unsigned tile = 0; tile < kTileCount; ++tile) {
        ranges[tile] = make_uint2(tile * kInstancesPerTile, (tile + 1) * kInstancesPerTile);
        const unsigned cluster = (tile / kTilesX / 2) * kClustersX + (tile % kTilesX / 2);
        const unsigned base = cluster * kInstancesPerTile;
        for (unsigned j = 0; j < kInstancesPerTile; ++j)
            ids[tile * kInstancesPerTile + j] = base + j;
    }
    std::vector<float> grad_image(kPixels * 3, 0.0f), grad_alpha(kPixels, 0.0f), image(kPixels * 3), alpha(kPixels);
    std::vector<uint> contributions(kPixels, 0);
    std::vector<float> trans(kTilePixelStates, 1.0f);

    // Mirror forward compositing on CPU so every pixel's final transmittance,
    // alpha map, accumulated RGB, and last-contributor count are coherent.
    for (unsigned tile_y = 0; tile_y < kTilesY; ++tile_y) {
        for (unsigned tile_x = 0; tile_x < kTilesX; ++tile_x) {
            const unsigned tile = tile_y * kTilesX + tile_x;
            for (unsigned rank = 0; rank < 256; ++rank) {
                const unsigned x = tile_x * kTileWidth + rank % kTileWidth;
                const unsigned y = tile_y * kTileHeight + rank / kTileWidth;
                const size_t tile_state = static_cast<size_t>(tile) * 256 + rank;
                if (x >= kWidth || y >= kHeight) { trans[tile_state] = 1.0f; continue; }
                float t = 1.0f, rgb[3]{0.0f, 0.0f, 0.0f};
                unsigned possible = 0, last = 0;
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                for (unsigned j = 0; j < kInstancesPerTile; ++j) {
                    ++possible;
                    const unsigned primitive = ids[tile * kInstancesPerTile + j];
                    const float dx = means[primitive].x - px, dy = means[primitive].y - py;
                    const float sigma = 0.5f * (conic[primitive].x * dx * dx + conic[primitive].z * dy * dy);
                    if (sigma < 0.0f) continue;
                    const float gaussian = std::exp(-sigma);
                    const float a = std::min(conic[primitive].w * gaussian, 0.999f);
                    if (a < (1.0f / 255.0f)) continue;
                    for (unsigned c = 0; c < 3; ++c) {
                        const float color = c == 0 ? colors[primitive].x : (c == 1 ? colors[primitive].y : colors[primitive].z);
                        rgb[c] += t * a * std::clamp(color, 0.0f, 4.0f);
                    }
                    t *= (1.0f - a);
                    last = possible;
                    if (t < 1.0e-4f) break;
                }
                const size_t pixel = static_cast<size_t>(y) * kWidth + x;
                contributions[pixel] = last;
                trans[tile_state] = t;
                alpha[pixel] = 1.0f - t;
                image[pixel] = rgb[0]; image[kPixels + pixel] = rgb[1]; image[2 * kPixels + pixel] = rgb[2];
                g_transmittance_min = std::min(g_transmittance_min, t);
                g_transmittance_max = std::max(g_transmittance_max, t);
                g_contributions_min = std::min(g_contributions_min, last);
                g_contributions_max = std::max(g_contributions_max, last);
            }
        }
    }
    rendered_image = image;
    upload(d.ranges, ranges, "upload tile ranges"); upload(d.ids, ids, "upload ids");
    upload(d.means, means, "upload means"); upload(d.conic, conic, "upload conic");
    upload(d.colors, colors, "upload colors"); upload(d.grad_image, grad_image, "upload image grads");
    upload(d.grad_alpha, grad_alpha, "upload alpha grads"); upload(d.image, image, "upload image");
    upload(d.alpha, alpha, "upload alpha"); upload(d.contributions, contributions, "upload contributions");
    upload(d.transmittance, trans, "upload transmittance");
    return d;
}

OutputSet allocate_outputs(const std::function<char*(size_t)>& allocate,
                           const RasterizerMemoryArena& arena) {
    OutputSet out{arena_array<float2>(allocate, kPrimitives),
                  arena_array<float>(allocate, 3 * kPrimitives),
                  arena_array<float>(allocate, kPrimitives),
                  arena_array<float3>(allocate, kPrimitives)};
#if LFS_BENCH_HAS_NATIVE_ATOMICS
    if (!arena.owns_device_allocation(out.mean, kPrimitives * sizeof(float2)) ||
        !arena.owns_device_allocation(out.conic, 3 * kPrimitives * sizeof(float)) ||
        !arena.owns_device_allocation(out.opacity, kPrimitives * sizeof(float)) ||
        !arena.owns_device_allocation(out.color, kPrimitives * sizeof(float3))) {
        std::fprintf(stderr, "Gradient outputs are not arena-owned\n"); std::exit(2);
    }
#else
    (void)arena;
#endif
    return out;
}

std::vector<float> download(const OutputSet& out) {
    std::vector<float2> mean(kPrimitives); std::vector<float> conic(3 * kPrimitives), opacity(kPrimitives);
    std::vector<float3> color(kPrimitives);
    check(cudaMemcpy(mean.data(), out.mean, mean.size() * sizeof(float2), cudaMemcpyDeviceToHost), "read means");
    check(cudaMemcpy(conic.data(), out.conic, conic.size() * sizeof(float), cudaMemcpyDeviceToHost), "read conic");
    check(cudaMemcpy(opacity.data(), out.opacity, opacity.size() * sizeof(float), cudaMemcpyDeviceToHost), "read opacity");
    check(cudaMemcpy(color.data(), out.color, color.size() * sizeof(float3), cudaMemcpyDeviceToHost), "read color");
    std::vector<float> all; all.reserve(2 * mean.size() + conic.size() + opacity.size() + 3 * color.size());
    for (auto v : mean) { all.push_back(v.x); all.push_back(v.y); }
    all.insert(all.end(), conic.begin(), conic.end()); all.insert(all.end(), opacity.begin(), opacity.end());
    for (auto v : color) { all.push_back(v.x); all.push_back(v.y); all.push_back(v.z); }
    return all;
}

void save_snapshot(const std::string& path, const std::vector<float>& values) {
    if (path.empty()) return;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("could not open snapshot output: " + path);
    constexpr char magic[8] = {'L','F','S','B','G','R','1','\0'};
    const std::array<std::uint32_t, 6> metadata{
        kWidth, kHeight, kTileCount, kInstances, kPrimitives, kInstancesPerTile};
    const std::uint64_t count = values.size();
    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char*>(metadata.data()), sizeof(metadata));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("failed writing snapshot output: " + path);
}

std::vector<double> cpu_ssim_backward(const lfs::training::kernels::FusedL1SSIMContext& ctx) {
    const auto pred = ctx.img1.cpu().to_vector();
    const auto target = ctx.img2.to(lfs::core::DataType::Float32).cpu().to_vector();
    const std::array<std::vector<float>, 3> partials{
        ctx.dm_dmu1.to(lfs::core::DataType::Float32).cpu().to_vector(),
        ctx.dm_dsigma1_sq.to(lfs::core::DataType::Float32).cpu().to_vector(),
        ctx.dm_dsigma12.to(lfs::core::DataType::Float32).cpu().to_vector()};
    const int height = ctx.H, width = ctx.W;
    const size_t plane_size = static_cast<size_t>(height) * width;
    const size_t planes = pred.size() / plane_size;
    const bool crop = ctx.apply_valid_padding && height > 10 && width > 10;
    const double norm = 1.0 / static_cast<double>(planes * (crop ? height - 10 : height) *
                                                  (crop ? width - 10 : width));
    const auto inside = [&](int y, int x) {
        return y >= 0 && y < height && x >= 0 && x < width &&
               (!crop || (y >= 5 && y < height - 5 && x >= 5 && x < width - 5));
    };
    std::vector<double> result(pred.size());
    for (size_t plane = 0; plane < planes; ++plane) {
        const size_t base = plane * plane_size;
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            std::array<double, 3> conv{};
            for (int dy = -5; dy <= 5; ++dy) for (int dx = -5; dx <= 5; ++dx) {
                if (!inside(y + dy, x + dx)) continue;
                const size_t source = base + static_cast<size_t>(y + dy) * width + (x + dx);
                const double w = kSsimWeights[dy + 5] * kSsimWeights[dx + 5] *
                                 (-ctx.ssim_weight) * norm;
                for (size_t c = 0; c < conv.size(); ++c) conv[c] += w * partials[c][source];
            }
            const size_t index = base + static_cast<size_t>(y) * width + x;
            const double delta = static_cast<double>(pred[index]) - target[index];
            const double sign = (delta > 0.0) - (delta < 0.0);
            const double l1 = inside(y, x) ? (1.0 - ctx.ssim_weight) * sign * norm : 0.0;
            result[index] = conv[0] + 2.0 * pred[index] * conv[1] + target[index] * conv[2] + l1;
        }
    }
    return result;
}

void check_ssim_oracle(const std::vector<float>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("SSIM oracle size mismatch");
    double error_sq = 0.0, reference_sq = 0.0, max_error = 0.0, max_reference = 0.0;
    size_t nonzero = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite SSIM image gradient");
        const double delta = static_cast<double>(actual[i]) - expected[i];
        error_sq += delta * delta; reference_sq += expected[i] * expected[i];
        max_error = std::max(max_error, std::fabs(delta));
        max_reference = std::max(max_reference, std::fabs(expected[i]));
        nonzero += actual[i] != 0.0f;
    }
    const double rel_l2 = std::sqrt(error_sq / std::max(reference_sq, 1.0e-300));
    if (!nonzero || max_reference == 0.0 || rel_l2 > 0.005)
        throw std::runtime_error("integrated SSIM CPU oracle gate failed");
    std::printf("ssim_oracle count=%zu nonzero=%zu max_abs=%.12g max_reference_abs=%.12g relative_l2=%.12g limit=0.005\n",
                actual.size(), nonzero, max_error, max_reference, rel_l2);
}

void save_ssim_gradient(const std::string& path, const std::vector<float>& values) {
    if (path.empty()) return;
    std::ofstream file(path + ".ssimgrad", std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("could not open SSIM gradient snapshot");
    constexpr char magic[8] = {'L','F','S','I','G','1','\0','\0'};
    const std::array<std::uint32_t, 3> metadata{kWidth, kHeight, 3};
    const std::uint64_t count = values.size();
    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char*>(metadata.data()), sizeof(metadata));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("failed writing SSIM gradient snapshot");
}
void report_group_errors(const std::vector<float>& reference, const std::vector<float>& actual) {
    const size_t n = kPrimitives;
    const std::array<std::pair<size_t, size_t>, 4> groups{{
        {0, 2 * n}, {2 * n, 5 * n}, {5 * n, 6 * n}, {6 * n, 9 * n}}};
    const std::array<const char*, 4> names{{"mean", "conic", "opacity", "color"}};
    bool passed = true;
    for (size_t g = 0; g < groups.size(); ++g) {
        float abs_error = 0.0f, scale = 0.0f;
        double error_sq = 0.0, reference_sq = 0.0;
        for (size_t i = groups[g].first; i < groups[g].second; ++i) {
            const double delta = static_cast<double>(actual[i]) - reference[i];
            abs_error = std::max(abs_error, static_cast<float>(std::fabs(delta)));
            scale = std::max(scale, std::fabs(reference[i]));
            error_sq += delta * delta;
            reference_sq += static_cast<double>(reference[i]) * reference[i];
        }
        const float relative = abs_error / std::max(scale, 1.0e-6f);
        const double relative_l2 = std::sqrt(error_sq) / std::max(std::sqrt(reference_sq), 1.0e-12);
        std::printf("safe_native_group=%s max_abs_diff=%.9g max_relative_diff=%.9g relative_l2=%.9g\n", names[g], abs_error, relative, relative_l2);
        if (!(relative <= 1.0e-4f) || !(relative_l2 <= 1.0e-4)) passed = false;
    }
    if (!passed) { std::fprintf(stderr, "safe/native per-group relative error exceeds tolerance 1e-4\n"); std::exit(3); }
}

std::string loaded_module_path(const wchar_t* name, const char* label) {
    HMODULE module = GetModuleHandleW(name);
    if (!module) return std::string("<") + label + "-not-loaded>";
    wchar_t wide_path[32768]{};
    constexpr DWORD path_capacity = static_cast<DWORD>(sizeof(wide_path) / sizeof(wide_path[0]));
    const DWORD length = GetModuleFileNameW(module, wide_path, path_capacity);
    if (length == 0 || length >= path_capacity) return std::string("<") + label + "-path-unavailable>";
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide_path, static_cast<int>(length), nullptr, 0, nullptr, nullptr);
    std::string path(static_cast<size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide_path, static_cast<int>(length), path.data(), bytes, nullptr, nullptr);
    return path;
}
bool validate_output(const std::vector<float>& values) {
    const size_t mean_end = 2 * static_cast<size_t>(kPrimitives);
    const size_t conic_end = mean_end + 3 * static_cast<size_t>(kPrimitives);
    const size_t opacity_end = conic_end + kPrimitives;
    const std::array<std::pair<size_t, size_t>, 4> groups{{
        {0, mean_end}, {mean_end, conic_end}, {conic_end, opacity_end}, {opacity_end, values.size()}}};
    const std::array<const char*, 4> names{{"mean", "conic", "opacity", "color"}};
    bool valid = true;
    for (size_t g = 0; g < groups.size(); ++g) {
        float maximum = 0.0f;
        for (size_t i = groups[g].first; i < groups[g].second; ++i) {
            if (!std::isfinite(values[i])) valid = false;
            maximum = std::max(maximum, std::fabs(values[i]));
        }
        std::printf("gradient_group=%s max_abs=%.9g\n", names[g], maximum);
        if (!(maximum > 1.0e-10f)) valid = false;
    }
    return valid;
}

size_t count_active_primitives(const std::vector<float>& values) {
    size_t active = 0;
    const size_t n = kPrimitives;
    for (size_t i = 0; i < n; ++i) {
        const size_t mean = 2 * i;
        const size_t conic = 2 * n + 3 * i;
        const size_t opacity = 5 * n + i;
        const size_t color = 6 * n + 3 * i;
        if (values[mean] != 0.0f || values[mean + 1] != 0.0f ||
            values[conic] != 0.0f || values[conic + 1] != 0.0f || values[conic + 2] != 0.0f ||
            values[opacity] != 0.0f || values[color] != 0.0f ||
            values[color + 1] != 0.0f || values[color + 2] != 0.0f) ++active;
    }
    return active;
}

struct TimingSamples { std::vector<double> wall_us; std::vector<double> event_us; };

template<class ComputeGradient>
TimingSamples measure_integrated(bool native, DeviceInputs& in, const OutputSet& out,
                                 cudaStream_t stream, int warmups, int samples,
                                 int batch_calls, ComputeGradient&& compute_gradient) {
    for (int i = 0; i < warmups; ++i) {
        zero_outputs(out, stream);
        Tensor grad = compute_gradient();
        in.grad_image = grad.ptr<float>();
        invoke(native, in.ranges, in.ids, in.means, in.conic, in.colors, in.grad_image,
               in.grad_alpha, in.image, in.alpha, in.contributions, in.transmittance, out, stream);
        check(cudaStreamSynchronize(stream), "integrated warmup sync");
    }
    TimingSamples timings;
    timings.wall_us.reserve(samples);
    timings.event_us.reserve(samples);
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create integrated start event");
    check(cudaEventCreate(&stop), "create integrated stop event");
    for (int sample = 0; sample < samples; ++sample) {
        check(cudaDeviceSynchronize(), "integrated pre-sample device sync");
        std::vector<Tensor> gradients;
        gradients.reserve(batch_calls);
        const auto wall_start = std::chrono::steady_clock::now();
        check(cudaEventRecord(start, stream), "record integrated start");
        for (int call = 0; call < batch_calls; ++call) {
            zero_outputs(out, stream);
            gradients.push_back(compute_gradient());
            in.grad_image = gradients.back().ptr<float>();
            invoke(native, in.ranges, in.ids, in.means, in.conic, in.colors, in.grad_image,
                   in.grad_alpha, in.image, in.alpha, in.contributions, in.transmittance, out, stream);
            check(cudaGetLastError(), "integrated blend backward launch");
        }
        check(cudaEventRecord(stop, stream), "record integrated stop");
        check(cudaEventSynchronize(stop), "sync integrated stop");
        check(cudaDeviceSynchronize(), "integrated batch device sync");
        float elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&elapsed_ms, start, stop), "integrated elapsed time");
        gradients.clear();
        const auto wall_stop = std::chrono::steady_clock::now();
        const double wall_batch_us = std::chrono::duration<double, std::micro>(wall_stop - wall_start).count();
        const double event_batch_us = static_cast<double>(elapsed_ms) * 1000.0;
        const double wall_per_call_us = wall_batch_us / batch_calls;
        const double event_per_call_us = event_batch_us / batch_calls;
        if (!(wall_per_call_us > 0.0) || !std::isfinite(wall_per_call_us))
            throw std::runtime_error("integrated synchronized wall-clock sample is not finite and positive");
        if (!(event_per_call_us > 0.0) || !std::isfinite(event_per_call_us))
            throw std::runtime_error("integrated HIP event batch sample is not finite and positive");
        timings.wall_us.push_back(wall_per_call_us);
        timings.event_us.push_back(event_per_call_us);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return timings;
}
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "safe";
    const int warmups = argc > 2 ? std::max(0, std::atoi(argv[2])) : 4;
    const int samples = argc > 3 ? std::max(1, std::atoi(argv[3])) : 9;
    const std::string scene = argc > 4 ? argv[4] : "half";
    const std::string snapshot_path = argc > 5 ? argv[5] : "";
    const std::string source_sha = argc > 6 ? argv[6] : "unspecified";
    if ((mode != "safe" && mode != "native") || (scene != "half" && scene != "full")) {
        std::fprintf(stderr, "usage: integrated_bench safe|native [warmups samples] [half|full] [snapshot-path] [source-sha]\n"); return 2;
    }
    if (scene == "full") { kWidth = 979; kHeight = 546; kInstancesPerTile = 192; }
    kTilesX = (kWidth + kTileWidth - 1) / kTileWidth;
    kTilesY = (kHeight + kTileHeight - 1) / kTileHeight;
    kTileCount = kTilesX * kTilesY;
    kClustersX = (kTilesX + 1) / 2;
    kClustersY = (kTilesY + 1) / 2;
    kPrimitives = kClustersX * kClustersY * kInstancesPerTile;
    kInstances = kTileCount * kInstancesPerTile;
    kPixels = static_cast<size_t>(kWidth) * kHeight;
    kTilePixelStates = static_cast<size_t>(kTileCount) * 256;
#if !LFS_BENCH_HAS_NATIVE_ATOMICS
    if (mode == "native") { std::fprintf(stderr, "native variant unavailable in this source revision\n"); return 2; }
#endif
    int count = 0; check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 1) { std::fprintf(stderr, "No HIP device found\n"); return 2; }
    int runtime_version = 0, driver_version = 0;
    check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    check(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    RasterizerMemoryArena arena; const auto frame = arena.begin_frame(); auto allocate = arena.get_allocator(frame);
    std::vector<float> rendered_image;
    DeviceInputs in = prepare_inputs(allocate, rendered_image);
    const OutputSet out = allocate_outputs(allocate, arena);
    const cudaStream_t stream = getCurrentCUDAStream();
    
    std::vector<float> target_values(rendered_image.size());
    for (size_t i = 0; i < rendered_image.size(); ++i) {
        const float perturbation = static_cast<float>((i * 19 + 13) % 31) / 400.0f;
        target_values[i] = std::clamp(0.72f * rendered_image[i] + 0.08f + perturbation, 0.0f, 1.0f);
    }
    Tensor prediction = Tensor::from_vector(rendered_image,
        TensorShape({1, 3, static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)}), Device::CUDA);
    Tensor target = Tensor::from_vector(target_values,
        TensorShape({1, 3, static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)}), Device::CUDA);
    FusedL1SSIMWorkspace ssim_workspace;
    auto compute_gradient = [&]() {
        auto [loss, context] = fused_l1_ssim_forward(prediction, target, 0.2f, ssim_workspace, true);
        (void)loss;
        return fused_l1_ssim_backward(context, ssim_workspace);
    };

    auto [loss, ssim_context] = fused_l1_ssim_forward(prediction, target, 0.2f, ssim_workspace, true);
    (void)loss;
    Tensor image_gradient = fused_l1_ssim_backward(ssim_context, ssim_workspace);
    const auto image_gradient_host = image_gradient.cpu().to_vector();
    check_ssim_oracle(image_gradient_host, cpu_ssim_backward(ssim_context));
    save_ssim_gradient(snapshot_path, image_gradient_host);
    in.grad_image = image_gradient.ptr<float>();
    
    const bool native = mode == "native";
    std::vector<float> result;
#if LFS_BENCH_HAS_NATIVE_ATOMICS
    if (native) {
        OutputSet safe_out = allocate_outputs(allocate, arena);
        zero_outputs(safe_out, stream);
        invoke(false, in.ranges, in.ids, in.means, in.conic, in.colors, in.grad_image,
               in.grad_alpha, in.image, in.alpha, in.contributions, in.transmittance, safe_out, stream);
        check(cudaStreamSynchronize(stream), "integrated safe correctness sync");
        const auto reference = download(safe_out);
        if (!validate_output(reference)) throw std::runtime_error("safe integrated reference gradients invalid");
        zero_outputs(out, stream);
        invoke(true, in.ranges, in.ids, in.means, in.conic, in.colors, in.grad_image,
               in.grad_alpha, in.image, in.alpha, in.contributions, in.transmittance, out, stream);
        check(cudaStreamSynchronize(stream), "integrated native correctness sync");
        result = download(out);
        report_group_errors(reference, result);
        if (!validate_output(result)) throw std::runtime_error("native integrated gradients invalid");
    }
#endif
    if (!native) {
        zero_outputs(out, stream);
        invoke(false, in.ranges, in.ids, in.means, in.conic, in.colors, in.grad_image,
               in.grad_alpha, in.image, in.alpha, in.contributions, in.transmittance, out, stream);
        check(cudaStreamSynchronize(stream), "integrated safe correctness sync");
        result = download(out);
    }
    if (!validate_output(result)) throw std::runtime_error("integrated gradients must be finite and nonzero in all groups");
    save_snapshot(snapshot_path, result);
    if (std::getenv("LFS_BENCH_SNAPSHOT_ONLY") != nullptr) {
        std::printf("SNAPSHOT_ONLY variant=%s scene=%s ssim_oracle=backward_combination_from_context_partials timing=disabled\n", mode.c_str(), scene.c_str());
        arena.end_frame(frame);
        return 0;
    }
    constexpr int batch_calls = 64;
    const auto timings = measure_integrated(native, in, out, stream, warmups, samples,
                                            batch_calls, compute_gradient);
    auto sorted_us = timings.wall_us; std::sort(sorted_us.begin(), sorted_us.end());
    const double median_us = sorted_us.size() % 2
        ? sorted_us[sorted_us.size() / 2]
        : 0.5 * (sorted_us[sorted_us.size() / 2 - 1] + sorted_us[sorted_us.size() / 2]);
    const auto final_result = download(out);
    const double checksum = std::accumulate(final_result.begin(), final_result.end(), 0.0);
    const std::string runtime = loaded_module_path(L"amdhip64_7.dll", "amdhip64_7.dll");
    std::printf("samples_wall_us=");
    for (size_t i = 0; i < timings.wall_us.size(); ++i) std::printf("%s%.6f", i ? "," : "", timings.wall_us[i]);
    std::printf("\n");
    std::printf("samples_event_us=");
    for (size_t i = 0; i < timings.event_us.size(); ++i) std::printf("%s%.6f", i ? "," : "", timings.event_us[i]);
    std::printf("\n");
    std::printf("INTEGRATED variant=%s source_sha=%s scene=%s device=\"%s\" total_global_mem=%zu runtime_version=%d driver_version=%d runtime_module=\"%s\" core_module=\"%s\" width=%u height=%u pixels=%zu tiles=%u instances=%u primitives=%u instances_per_tile=%u grad_alpha=zero prediction=CPU_mirrored_forward target=deterministic_bounded standard_sigma_partials=on timing_scope=SSIM_forward_backward_plus_FastGS_blend_backward_plus_output_zeroing timer=steady_clock_batch_synced_primary HIP_events_secondary batch_calls=%d warmups=%d samples=%d median_wall_us=%.6f min_wall_us=%.6f max_wall_us=%.6f checksum=%.12g\n",
        mode.c_str(), source_sha.c_str(), scene.c_str(), prop.name, prop.totalGlobalMem, runtime_version, driver_version,
        runtime.c_str(), loaded_module_path(L"lfs_core.dll", "lfs_core.dll").c_str(),
        kWidth, kHeight, kPixels, kTileCount, kInstances, kPrimitives, kInstancesPerTile,
        batch_calls, warmups, samples, median_us, sorted_us.front(), sorted_us.back(), checksum);
    arena.end_frame(frame);
    return 0;
}
