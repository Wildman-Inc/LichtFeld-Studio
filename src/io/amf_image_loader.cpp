/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/amf_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "cuda/image_format_kernels.cuh"

#include <cuda_runtime.h>
#include <stb_image.h>

#include "public/common/AMFFactory.h"
#include "public/common/AMFSTL.h"
#include "public/include/components/VideoDecoderUVD.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace lfs::io {

    namespace {
        constexpr int LANCZOS_KERNEL_SIZE = 2;
        constexpr int QUERY_ATTEMPTS = 200;
        constexpr auto QUERY_SLEEP = std::chrono::milliseconds(1);

        std::mutex& amf_factory_mutex() {
            static std::mutex mutex;
            return mutex;
        }

        const char* amf_result_to_string(const AMF_RESULT result) {
            switch (result) {
            case AMF_OK: return "AMF_OK";
            case AMF_FAIL: return "AMF_FAIL";
            case AMF_UNEXPECTED: return "AMF_UNEXPECTED";
            case AMF_ACCESS_DENIED: return "AMF_ACCESS_DENIED";
            case AMF_INVALID_ARG: return "AMF_INVALID_ARG";
            case AMF_OUT_OF_RANGE: return "AMF_OUT_OF_RANGE";
            case AMF_OUT_OF_MEMORY: return "AMF_OUT_OF_MEMORY";
            case AMF_INVALID_POINTER: return "AMF_INVALID_POINTER";
            case AMF_NO_INTERFACE: return "AMF_NO_INTERFACE";
            case AMF_NOT_IMPLEMENTED: return "AMF_NOT_IMPLEMENTED";
            case AMF_NOT_SUPPORTED: return "AMF_NOT_SUPPORTED";
            case AMF_NOT_FOUND: return "AMF_NOT_FOUND";
            case AMF_ALREADY_INITIALIZED: return "AMF_ALREADY_INITIALIZED";
            case AMF_NOT_INITIALIZED: return "AMF_NOT_INITIALIZED";
            case AMF_INVALID_FORMAT: return "AMF_INVALID_FORMAT";
            case AMF_WRONG_STATE: return "AMF_WRONG_STATE";
            case AMF_FILE_NOT_OPEN: return "AMF_FILE_NOT_OPEN";
            case AMF_NO_DEVICE: return "AMF_NO_DEVICE";
            case AMF_DIRECTX_FAILED: return "AMF_DIRECTX_FAILED";
            case AMF_OPENCL_FAILED: return "AMF_OPENCL_FAILED";
            case AMF_GLX_FAILED: return "AMF_GLX_FAILED";
            case AMF_ALSA_FAILED: return "AMF_ALSA_FAILED";
            case AMF_EOF: return "AMF_EOF";
            case AMF_REPEAT: return "AMF_REPEAT";
            case AMF_INPUT_FULL: return "AMF_INPUT_FULL";
            case AMF_RESOLUTION_CHANGED: return "AMF_RESOLUTION_CHANGED";
            case AMF_RESOLUTION_UPDATED: return "AMF_RESOLUTION_UPDATED";
            case AMF_INVALID_DATA_TYPE: return "AMF_INVALID_DATA_TYPE";
            case AMF_INVALID_RESOLUTION: return "AMF_INVALID_RESOLUTION";
            case AMF_CODEC_NOT_SUPPORTED: return "AMF_CODEC_NOT_SUPPORTED";
            case AMF_SURFACE_FORMAT_NOT_SUPPORTED: return "AMF_SURFACE_FORMAT_NOT_SUPPORTED";
            case AMF_SURFACE_MUST_BE_SHARED: return "AMF_SURFACE_MUST_BE_SHARED";
            case AMF_DECODER_NOT_PRESENT: return "AMF_DECODER_NOT_PRESENT";
            case AMF_DECODER_SURFACE_ALLOCATION_FAILED: return "AMF_DECODER_SURFACE_ALLOCATION_FAILED";
            case AMF_DECODER_NO_FREE_SURFACES: return "AMF_DECODER_NO_FREE_SURFACES";
            case AMF_NEED_MORE_INPUT: return "AMF_NEED_MORE_INPUT";
            default: return "AMF_UNKNOWN";
            }
        }

        void throw_if_failed(const AMF_RESULT result, const char* operation) {
            if (result == AMF_OK)
                return;
            throw std::runtime_error(std::string(operation) + " failed: " + amf_result_to_string(result));
        }

        [[nodiscard]] bool is_jpeg_data(const std::vector<uint8_t>& data) {
            return data.size() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
        }

        [[nodiscard]] std::pair<int, int> get_jpeg_dimensions(const std::vector<uint8_t>& data) {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!stbi_info_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, &channels) ||
                width <= 0 || height <= 0) {
                throw std::runtime_error("Failed to read JPEG dimensions");
            }
            return {width, height};
        }

        [[nodiscard]] std::pair<int, int> compute_target_size(
            const int src_width,
            const int src_height,
            const int resize_factor,
            const int max_width) {
            int target_width = src_width;
            int target_height = src_height;
            if (resize_factor > 1) {
                target_width = std::max(1, target_width / resize_factor);
                target_height = std::max(1, target_height / resize_factor);
            }
            if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
                if (target_width > target_height) {
                    target_height = std::max(1, max_width * target_height / target_width);
                    target_width = max_width;
                } else {
                    target_width = std::max(1, max_width * target_width / target_height);
                    target_height = max_width;
                }
            }
            return {target_width, target_height};
        }

        cuda::PackedPixelFormat packed_pixel_format_from_amf_surface(const amf::AMF_SURFACE_FORMAT surface_format) {
            switch (surface_format) {
            case amf::AMF_SURFACE_BGRA:
                return cuda::PackedPixelFormat::BGRA;
            case amf::AMF_SURFACE_RGBA:
                return cuda::PackedPixelFormat::RGBA;
            case amf::AMF_SURFACE_ARGB:
                return cuda::PackedPixelFormat::ARGB;
            default:
                throw std::runtime_error("Unsupported AMF output surface format");
            }
        }

        lfs::core::Tensor upload_surface_to_tensor(
            amf::AMFSurface* const surface,
            const int target_width,
            const int target_height,
            const DecodeFormat format,
            void* const cuda_stream) {
            if (!surface)
                throw std::runtime_error("AMF decode produced no surface");

            const AMF_RESULT convert_result = surface->Convert(amf::AMF_MEMORY_HOST);
            throw_if_failed(convert_result, "AMF surface host conversion");

            amf::AMFPlane* const plane = surface->GetPlaneAt(0);
            if (!plane || !plane->GetNative())
                throw std::runtime_error("AMF decode produced no host plane");

            const int width = plane->GetWidth();
            const int height = plane->GetHeight();
            const int pitch = plane->GetHPitch();
            if (width <= 0 || height <= 0 || pitch < width * 4)
                throw std::runtime_error("AMF decode produced invalid dimensions");

            const auto packed_format = packed_pixel_format_from_amf_surface(surface->GetFormat());
            const auto stream = static_cast<cudaStream_t>(cuda_stream);
            const auto pitch_bytes = static_cast<size_t>(pitch);
            const auto width_bytes = static_cast<size_t>(width) * 4;
            auto packed_gpu = lfs::core::Tensor::empty(
                lfs::core::TensorShape({static_cast<size_t>(height) * pitch_bytes}),
                lfs::core::Device::CUDA,
                lfs::core::DataType::UInt8);

            const cudaError_t copy_result = cudaMemcpy2DAsync(
                packed_gpu.ptr<uint8_t>(),
                pitch_bytes,
                plane->GetNative(),
                pitch_bytes,
                width_bytes,
                static_cast<size_t>(height),
                cudaMemcpyHostToDevice,
                stream);
            if (copy_result != cudaSuccess) {
                throw std::runtime_error(std::string("AMF surface upload failed: ") + cudaGetErrorString(copy_result));
            }

            lfs::core::Tensor output;
            const bool needs_resize = target_width != width || target_height != height;
            if (needs_resize) {
                if (format == DecodeFormat::Grayscale) {
                    auto gpu_uint8 = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({static_cast<size_t>(height), static_cast<size_t>(width)}),
                        lfs::core::Device::CUDA,
                        lfs::core::DataType::UInt8);
                    cuda::launch_packed4_to_uint8_hw(
                        packed_gpu.ptr<uint8_t>(),
                        gpu_uint8.ptr<uint8_t>(),
                        static_cast<size_t>(height),
                        static_cast<size_t>(width),
                        pitch_bytes,
                        packed_format,
                        stream);
                    output = lfs::core::lanczos_resize_grayscale(
                        gpu_uint8, target_height, target_width, LANCZOS_KERNEL_SIZE, stream);
                } else {
                    auto gpu_uint8 = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), 3}),
                        lfs::core::Device::CUDA,
                        lfs::core::DataType::UInt8);
                    cuda::launch_packed4_to_uint8_hwc(
                        packed_gpu.ptr<uint8_t>(),
                        gpu_uint8.ptr<uint8_t>(),
                        static_cast<size_t>(height),
                        static_cast<size_t>(width),
                        pitch_bytes,
                        packed_format,
                        stream);
                    output = lfs::core::lanczos_resize(
                        gpu_uint8, target_height, target_width, LANCZOS_KERNEL_SIZE, stream);
                }
            } else if (format == DecodeFormat::Grayscale) {
                output = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({static_cast<size_t>(height), static_cast<size_t>(width)}),
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                cuda::launch_packed4_to_float32_hw(
                    packed_gpu.ptr<uint8_t>(),
                    output.ptr<float>(),
                    static_cast<size_t>(height),
                    static_cast<size_t>(width),
                    pitch_bytes,
                    packed_format,
                    stream);
            } else {
                output = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({3, static_cast<size_t>(height), static_cast<size_t>(width)}),
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                cuda::launch_packed4_to_float32_chw(
                    packed_gpu.ptr<uint8_t>(),
                    output.ptr<float>(),
                    static_cast<size_t>(height),
                    static_cast<size_t>(width),
                    pitch_bytes,
                    packed_format,
                    stream);
            }

            const cudaError_t sync_result = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            if (sync_result != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(sync_result));
            }
            return output;
        }

        bool check_amf_availability() {
#if defined(_WIN32)
            std::lock_guard<std::mutex> lock(amf_factory_mutex());
            if (g_AMFFactory.Init() != AMF_OK)
                return false;

            bool available = false;
            amf::AMFContextPtr context;
            amf::AMFComponentPtr decoder;
            if (g_AMFFactory.GetFactory() &&
                g_AMFFactory.GetFactory()->CreateContext(&context) == AMF_OK &&
                context != nullptr &&
                context->InitDX11(nullptr) == AMF_OK &&
                g_AMFFactory.GetFactory()->CreateComponent(context, AMFVideoDecoderUVD_MJPEG, &decoder) == AMF_OK &&
                decoder != nullptr) {
                available = true;
                decoder->Terminate();
            }
            decoder = nullptr;

            if (context != nullptr) {
                context->Terminate();
                context = nullptr;
            }
            g_AMFFactory.Terminate();
            return available;
#else
            return false;
#endif
        }

    } // namespace

    struct AmfImageLoader::Impl {
        amf::AMFContextPtr context;
        int device_id = 0;
        std::mutex mutex;

        void initialize() {
            std::lock_guard<std::mutex> factory_lock(amf_factory_mutex());
            throw_if_failed(g_AMFFactory.Init(), "AMF runtime initialization");
            throw_if_failed(g_AMFFactory.GetFactory()->CreateContext(&context), "AMF context creation");
            throw_if_failed(context->InitDX11(nullptr), "AMF DX11 initialization");
        }

        ~Impl() {
            if (context != nullptr) {
                context->Terminate();
                context = nullptr;
            }
            std::lock_guard<std::mutex> factory_lock(amf_factory_mutex());
            g_AMFFactory.Terminate();
        }
    };

    AmfImageLoader::AmfImageLoader(const Options& options)
        : impl_(std::make_unique<Impl>()) {
        impl_->device_id = options.device_id;
        LOG_INFO("[AmfImageLoader] Initializing: device={}, pool={}",
                 options.device_id, options.decoder_pool_size);
        impl_->initialize();
    }

    AmfImageLoader::~AmfImageLoader() = default;

    bool AmfImageLoader::is_available() {
        static std::once_flag once;
        static bool available = false;
        std::call_once(once, [] { available = check_amf_availability(); });
        return available;
    }

    std::vector<uint8_t> AmfImageLoader::read_file(const std::filesystem::path& path) const {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file)) {
            throw std::runtime_error("Failed to open file: " + lfs::core::path_to_utf8(path));
        }

        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(static_cast<size_t>(size));
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + lfs::core::path_to_utf8(path));
        }
        return buffer;
    }

    lfs::core::Tensor AmfImageLoader::load_image_gpu(
        const std::filesystem::path& path,
        const int resize_factor,
        const int max_width,
        void* const cuda_stream,
        const DecodeFormat format) {
        return load_image_from_memory_gpu(read_file(path), resize_factor, max_width, cuda_stream, format);
    }

    lfs::core::Tensor AmfImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>& jpeg_data,
        const int resize_factor,
        const int max_width,
        void* const cuda_stream,
        const DecodeFormat format) {
        if (!is_jpeg_data(jpeg_data))
            throw std::runtime_error("AMF decoder only supports JPEG data");

        const auto [src_width, src_height] = get_jpeg_dimensions(jpeg_data);
        const auto [target_width, target_height] =
            compute_target_size(src_width, src_height, resize_factor, max_width);

        lfs::core::Tensor output;

        {
            std::lock_guard<std::mutex> lock(impl_->mutex);

            amf::AMFComponentPtr decoder;
            throw_if_failed(
                g_AMFFactory.GetFactory()->CreateComponent(impl_->context, AMFVideoDecoderUVD_MJPEG, &decoder),
                "AMF MJPEG decoder creation");

            decoder->SetProperty(AMF_TIMESTAMP_MODE, amf_int64(AMF_TS_DECODE));
            decoder->SetProperty(AMF_VIDEO_DECODER_LOW_LATENCY, true);
            decoder->SetProperty(AMF_VIDEO_DECODER_SURFACE_CPU, true);
            decoder->SetProperty(AMF_VIDEO_DECODER_SURFACE_POOL_SIZE, amf_int64(2));
            decoder->SetProperty(AMF_VIDEO_DECODER_OUTPUT_FORMAT, amf_int64(amf::AMF_SURFACE_BGRA));

            AMF_RESULT result = decoder->Init(amf::AMF_SURFACE_BGRA, src_width, src_height);
            if (result != AMF_OK) {
                decoder->Terminate();
                throw_if_failed(result, "AMF MJPEG decoder initialization");
            }

            amf::AMFBufferPtr input;
            result = impl_->context->AllocBuffer(amf::AMF_MEMORY_HOST, jpeg_data.size(), &input);
            if (result != AMF_OK) {
                decoder->Terminate();
                throw_if_failed(result, "AMF input buffer allocation");
            }
            std::memcpy(input->GetNative(), jpeg_data.data(), jpeg_data.size());
            input->SetPts(0);
            input->SetDuration(1);

            result = decoder->SubmitInput(input);
            if (result != AMF_OK) {
                decoder->Terminate();
                throw_if_failed(result, "AMF MJPEG input submission");
            }

            amf::AMFDataPtr data;
            for (int attempt = 0; attempt < QUERY_ATTEMPTS; ++attempt) {
                result = decoder->QueryOutput(&data);
                if (result == AMF_OK && data != nullptr)
                    break;
                if (result == AMF_REPEAT || result == AMF_NEED_MORE_INPUT || result == AMF_INPUT_FULL ||
                    result == AMF_DECODER_NO_FREE_SURFACES) {
                    std::this_thread::sleep_for(QUERY_SLEEP);
                    continue;
                }
                if (result != AMF_OK) {
                    decoder->Terminate();
                    throw_if_failed(result, "AMF MJPEG output query");
                }
                std::this_thread::sleep_for(QUERY_SLEEP);
            }

            if (data == nullptr) {
                decoder->Drain();
                for (int attempt = 0; attempt < QUERY_ATTEMPTS && data == nullptr; ++attempt) {
                    result = decoder->QueryOutput(&data);
                    if (result == AMF_OK && data != nullptr)
                        break;
                    std::this_thread::sleep_for(QUERY_SLEEP);
                }
            }

            if (data == nullptr) {
                decoder->Terminate();
                throw std::runtime_error("AMF MJPEG decoder produced no output");
            }

            amf::AMFSurfacePtr surface(data);
            output = upload_surface_to_tensor(surface, target_width, target_height, format, cuda_stream);
            decoder->Terminate();
        }

        return output;
    }

} // namespace lfs::io
