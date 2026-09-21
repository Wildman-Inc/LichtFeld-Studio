/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/rocjpeg_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "cuda/image_format_kernels.cuh"

#include <rocjpeg/rocjpeg.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfs::io {
    namespace {
        struct StreamHandle {
            RocJpegStreamHandle value = nullptr;
            ~StreamHandle() {
                if (value)
                    rocJpegStreamDestroy(value);
            }
        };

        void check_cuda(const cudaError_t result, const char* operation) {
            if (result != cudaSuccess)
                throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
        }

        class DeviceScope {
        public:
            explicit DeviceScope(const int device) {
                check_cuda(cudaGetDevice(&previous_), "Get current JPEG device");
                if (previous_ != device) {
                    check_cuda(cudaSetDevice(device), "Select JPEG decode device");
                    changed_ = true;
                }
            }
            ~DeviceScope() {
                if (changed_)
                    (void)cudaSetDevice(previous_);
            }
            DeviceScope(const DeviceScope&) = delete;
            DeviceScope& operator=(const DeviceScope&) = delete;

        private:
            int previous_ = 0;
            bool changed_ = false;
        };
    } // namespace

    struct RocJpegImageLoader::Impl {
        RocJpegHandle decoder = nullptr;
        cudaStream_t stream = nullptr;
        int device = 0;
        std::mutex mutex;
        StreamHandle input;
        lfs::core::Tensor image;
        lfs::core::LanczosResizeWorkspace resize_workspace;

        ~Impl() {
            if (!stream && !decoder)
                return;
            int previous = 0;
            const bool got_device = cudaGetDevice(&previous) == cudaSuccess;
            if (!got_device || previous != device)
                (void)cudaSetDevice(device);
            if (stream) {
                (void)cudaStreamSynchronize(stream);
                image = {};
                resize_workspace = {};
                lfs::core::CudaMemoryPool::instance().release_stream(stream);
                (void)cudaStreamDestroy(stream);
            }
            if (decoder)
                rocJpegDestroy(decoder);
            if (got_device && previous != device)
                (void)cudaSetDevice(previous);
        }
    };

    RocJpegImageLoader::RocJpegImageLoader() : impl_(std::make_unique<Impl>()) {
        if (const char* disabled = std::getenv("LFS_DISABLE_ROCJPEG");
            disabled && (std::string_view(disabled) == "1" || std::string_view(disabled) == "true"))
            return;

        check_cuda(cudaGetDevice(&impl_->device), "Get JPEG decode device");
        const auto status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, impl_->device, &impl_->decoder);
        if (status != ROCJPEG_STATUS_SUCCESS) {
            LOG_INFO("[RocJpegImageLoader] Hardware JPEG unavailable ({}); using CPU decode",
                     rocJpegGetErrorName(status));
            return;
        }
        check_cuda(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking),
                   "Create JPEG post-processing stream");
        LOG_INFO("[RocJpegImageLoader] Windows D3D11/VCN hardware JPEG enabled on GPU {}", impl_->device);
    }

    RocJpegImageLoader::~RocJpegImageLoader() = default;

    bool RocJpegImageLoader::available() const {
        return impl_->decoder != nullptr;
    }

    lfs::core::Tensor RocJpegImageLoader::decode(
        const std::span<const uint8_t> jpeg, const int resize_factor,
        const int max_width, const bool output_uint8, lfs::core::Tensor* reusable_output) {
        using namespace lfs::core;
        if (!available() || jpeg.size() < 3 || jpeg[0] != 0xff || jpeg[1] != 0xd8 || jpeg[2] != 0xff)
            return {};

        std::lock_guard lock(impl_->mutex);
        const DeviceScope device_scope(impl_->device);
        const CUDAStreamGuard stream_guard(impl_->stream);
        auto& input = impl_->input;
        if (!input.value) {
            const auto status = rocJpegStreamCreate(&input.value);
            if (status != ROCJPEG_STATUS_SUCCESS)
                throw std::runtime_error(std::string("Create JPEG stream: ") + rocJpegGetErrorName(status));
        }
        if (rocJpegStreamParse(jpeg.data(), jpeg.size(), input.value) != ROCJPEG_STATUS_SUCCESS)
            return {};

        uint8_t components = 0;
        RocJpegChromaSubsampling subsampling{};
        uint32_t widths[ROCJPEG_MAX_COMPONENT]{};
        uint32_t heights[ROCJPEG_MAX_COMPONENT]{};
        auto status = rocJpegGetImageInfo(impl_->decoder, input.value, &components,
                                          &subsampling, widths, heights);
        if (status != ROCJPEG_STATUS_SUCCESS || widths[0] == 0 || heights[0] == 0)
            return {};
        const size_t width = widths[0], height = heights[0];
        // Bound allocations before calling the hardware backend. Larger images
        // and non-RGB JPEG variants remain supported by the CPU decoder.
        if (width > 8192 || height > 8192 || (components != 1 && components != 3))
            return {};

        int target_width = std::max(1, static_cast<int>(width) / std::max(1, resize_factor));
        int target_height = std::max(1, static_cast<int>(height) / std::max(1, resize_factor));
        if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
            if (target_width > target_height) {
                target_height = std::max(1, max_width * target_height / target_width);
                target_width = max_width;
            } else {
                target_width = std::max(1, max_width * target_width / target_height);
                target_height = max_width;
            }
        }

        const TensorShape shape({size_t{3}, static_cast<size_t>(target_height), static_cast<size_t>(target_width)});
        const auto dtype = output_uint8 ? DataType::UInt8 : DataType::Float32;
        const bool reuse_output = reusable_output && reusable_output->is_valid() &&
                                  reusable_output->shape() == shape && reusable_output->device() == Device::CUDA &&
                                  reusable_output->dtype() == dtype && reusable_output->is_contiguous();
        Tensor output = reuse_output ? *reusable_output : Tensor::empty(shape, Device::CUDA, dtype);
        output.set_stream(impl_->stream);
        const bool resize = target_width != static_cast<int>(width) || target_height != static_cast<int>(height);
        const bool direct_planar = !resize && output_uint8;
        auto& image = impl_->image;
        const bool allocate_image = !direct_planar && (!image.is_valid() || image.shape() != TensorShape({height, width, size_t{3}}));
        if (allocate_image)
            image = Tensor::empty({height, width, size_t{3}}, Device::CUDA, DataType::UInt8);
        // A fresh pool allocation can have a reuse dependency on our stream.
        // Retained HWC and leased CHW destinations are already complete.
        if (allocate_image || (direct_planar && !reuse_output))
            check_cuda(cudaStreamSynchronize(impl_->stream), "Prepare JPEG destination");
        RocJpegImage destination{};
        RocJpegDecodeParams params{};
        if (direct_planar) {
            params.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
            for (size_t c = 0; c < 3; ++c) {
                destination.channel[c] = output.ptr<uint8_t>() + c * width * height;
                destination.pitch[c] = static_cast<uint32_t>(width);
            }
        } else {
            params.output_format = ROCJPEG_OUTPUT_RGB;
            destination.channel[0] = image.ptr<uint8_t>();
            destination.pitch[0] = static_cast<uint32_t>(width * 3);
        }
        status = rocJpegDecode(impl_->decoder, input.value, &params, &destination);
        if (status != ROCJPEG_STATUS_SUCCESS) {
            LOG_DEBUG("[RocJpegImageLoader] JPEG falls back to CPU: {}", rocJpegGetErrorName(status));
            return {};
        }
        try {
            if (resize)
                lanczos_resize_into(image, output, impl_->resize_workspace, 2, impl_->stream);
            else if (!direct_planar) {
                cuda::launch_uint8_hwc_to_float32_chw(image.ptr<uint8_t>(), output.ptr<float>(),
                                                      height, width, 3, impl_->stream);
                check_cuda(cudaStreamSynchronize(impl_->stream), "Finish JPEG post-processing");
            }
        } catch (...) {
            // A failed launch/check may leave earlier work queued against the
            // retained HWC buffer or leased output. Drain before CPU fallback.
            (void)cudaStreamSynchronize(impl_->stream);
            throw;
        }
        output.set_stream(nullptr);
        if (reusable_output)
            *reusable_output = output;
        return output;
    }

} // namespace lfs::io
