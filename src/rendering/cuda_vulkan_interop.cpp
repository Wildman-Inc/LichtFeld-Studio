/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_vulkan_interop.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"
#include "image_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::rendering {
    namespace {
        std::mutex g_device_identity_mutex;
        std::optional<std::array<std::uint8_t, 16>> g_expected_vk_uuid;
        std::array<std::uint8_t, 8> g_expected_vk_luid{};
        bool g_expected_vk_luid_valid = false;
        bool g_device_match_resolved = false;
        bool g_device_match_ok = false;
        std::string g_device_match_error;

#if defined(USE_HIP) && USE_HIP
        constexpr const char* kComputeApiName = "HIP";
        constexpr const char* kVisibleDevicesVariable = "HIP_VISIBLE_DEVICES";
#else
        constexpr const char* kComputeApiName = "CUDA";
        constexpr const char* kVisibleDevicesVariable = "CUDA_VISIBLE_DEVICES";
#endif

        std::string formatUuid(const std::array<std::uint8_t, 16>& uuid) {
            std::string s;
            s.reserve(36);
            for (std::size_t i = 0; i < uuid.size(); ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) {
                    s.push_back('-');
                }
                std::array<char, 3> buf{};
                std::snprintf(buf.data(), buf.size(), "%02x", uuid[i]);
                s.append(buf.data(), 2);
            }
            return s;
        }

        template <std::size_t N>
        bool hasNonZeroIdentityByte(const std::array<std::uint8_t, N>& identity) {
            return std::any_of(identity.begin(), identity.end(), [](const std::uint8_t byte) {
                return byte != 0;
            });
        }

#if defined(_WIN32) && defined(USE_HIP) && USE_HIP
        std::string formatLuid(const std::array<std::uint8_t, 8>& luid) {
            constexpr char kHexDigits[] = "0123456789abcdef";
            std::string s;
            s.reserve(luid.size() * 2);
            for (const std::uint8_t byte : luid) {
                s.push_back(kHexDigits[byte >> 4]);
                s.push_back(kHexDigits[byte & 0x0f]);
            }
            return s;
        }
#endif
    } // namespace

    void setExpectedVulkanDeviceIdentity(const std::array<std::uint8_t, 16>& uuid,
                                         const std::array<std::uint8_t, 8>& luid,
                                         const bool luid_valid) {
        std::lock_guard lk(g_device_identity_mutex);
        g_expected_vk_uuid = uuid;
        g_expected_vk_luid = luid;
        g_expected_vk_luid_valid = luid_valid;
        g_device_match_resolved = false;
        g_device_match_ok = false;
        g_device_match_error.clear();
    }

    std::optional<std::string> verifyCudaMatchesVulkanDevice() {
        std::lock_guard lk(g_device_identity_mutex);
        if (g_device_match_resolved) {
            if (g_device_match_ok) {
                return std::nullopt;
            }
            return g_device_match_error;
        }
        g_device_match_resolved = true;
        if (!g_expected_vk_uuid) {
            g_device_match_error =
                "Vulkan device identity was not registered before CUDA/Vulkan interop init; "
                "call setExpectedVulkanDeviceIdentity() once at startup";
            return g_device_match_error;
        }
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDevice failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }
        cudaDeviceProp props{};
        status = cudaGetDeviceProperties(&props, cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDeviceProperties failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }

#if defined(_WIN32) && defined(USE_HIP) && USE_HIP
        if (g_expected_vk_luid_valid) {
            std::array<std::uint8_t, 8> hip_luid_bytes{};
            static_assert(sizeof(props.luid) == hip_luid_bytes.size());
            std::memcpy(hip_luid_bytes.data(), props.luid, hip_luid_bytes.size());
            if (!hasNonZeroIdentityByte(g_expected_vk_luid) ||
                !hasNonZeroIdentityByte(hip_luid_bytes)) {
                g_device_match_error =
                    "HIP/Vulkan LUID identity is all zero and cannot safely authorize external-memory interop";
                return g_device_match_error;
            }
            if (hip_luid_bytes != g_expected_vk_luid) {
                g_device_match_error = std::format(
                    "{} device {} (LUID {}) does not match the selected Vulkan physical device (LUID {}). "
                    "Set {} to expose the same GPU to both APIs.",
                    kComputeApiName,
                    cuda_device,
                    formatLuid(hip_luid_bytes),
                    formatLuid(g_expected_vk_luid),
                    kVisibleDevicesVariable);
                return g_device_match_error;
            }
            g_device_match_ok = true;
            return std::nullopt;
        }
#endif

        std::array<std::uint8_t, 16> cuda_uuid_bytes{};
        std::memcpy(cuda_uuid_bytes.data(), props.uuid.bytes, 16);
        if (!hasNonZeroIdentityByte(*g_expected_vk_uuid) ||
            !hasNonZeroIdentityByte(cuda_uuid_bytes)) {
            g_device_match_error = std::format(
                "{}/Vulkan UUID identity is all zero and cannot safely authorize external-memory interop",
                kComputeApiName);
            return g_device_match_error;
        }
        if (cuda_uuid_bytes != *g_expected_vk_uuid) {
            g_device_match_error = std::format(
                "{} device {} (UUID {}) does not match the selected Vulkan physical device (UUID {}). "
                "Set {} to expose the same GPU to both APIs.",
                kComputeApiName,
                cuda_device,
                formatUuid(cuda_uuid_bytes),
                formatUuid(*g_expected_vk_uuid),
                kVisibleDevicesVariable);
            return g_device_match_error;
        }
        g_device_match_ok = true;
        return std::nullopt;
    }

    namespace {
#ifdef _WIN32
        constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType =
            cudaExternalMemoryHandleTypeOpaqueWin32;
        constexpr cudaExternalSemaphoreHandleType kCudaExternalSemaphoreHandleType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
        constexpr cudaExternalMemoryHandleType kCudaExternalMemoryHandleType =
            cudaExternalMemoryHandleTypeOpaqueFd;
        constexpr cudaExternalSemaphoreHandleType kCudaExternalSemaphoreHandleType =
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif

        [[nodiscard]] bool nativeHandleValid(const CudaVulkanExternalHandle handle) {
#ifdef _WIN32
            return handle != nullptr;
#else
            return handle >= 0;
#endif
        }

        void closeNativeHandle(CudaVulkanExternalHandle& handle) {
            if (!nativeHandleValid(handle)) {
                return;
            }
#ifdef _WIN32
            CloseHandle(static_cast<HANDLE>(handle));
            handle = nullptr;
#else
            ::close(handle);
            handle = -1;
#endif
        }

        class NativeHandleOwner {
        public:
            explicit NativeHandleOwner(const CudaVulkanExternalHandle handle)
                : handle_(handle) {}

            ~NativeHandleOwner() {
                closeNativeHandle(handle_);
            }

            NativeHandleOwner(const NativeHandleOwner&) = delete;
            NativeHandleOwner& operator=(const NativeHandleOwner&) = delete;

            [[nodiscard]] CudaVulkanExternalHandle get() const { return handle_; }

            void release() {
                handle_ = kInvalidCudaVulkanExternalHandle;
            }

        private:
            CudaVulkanExternalHandle handle_ = kInvalidCudaVulkanExternalHandle;
        };

        [[nodiscard]] cudaChannelFormatDesc channelDescForFormat(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm:
                return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
            case CudaVulkanImageFormat::R32Sfloat:
                return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            }
            return {};
        }

        [[nodiscard]] const char* formatName(const CudaVulkanImageFormat format) {
            switch (format) {
            case CudaVulkanImageFormat::Rgba8Unorm:
                return "RGBA8_UNORM";
            case CudaVulkanImageFormat::R32Sfloat:
                return "R32_SFLOAT";
            }
            return "unknown";
        }

        [[nodiscard]] bool formatSupported(const CudaVulkanImageFormat format) {
            return format == CudaVulkanImageFormat::Rgba8Unorm ||
                   format == CudaVulkanImageFormat::R32Sfloat;
        }

        struct PreparedCudaImageTensor {
            lfs::core::Tensor tensor;
            int width = 0;
            int height = 0;
            int channels = 0;
            detail::CudaVulkanTensorLayout layout = detail::CudaVulkanTensorLayout::Hwc;
            detail::CudaVulkanTensorElementType element_type = detail::CudaVulkanTensorElementType::UInt8;
        };

        [[nodiscard]] bool prepareCudaImageTensor(const lfs::core::Tensor& tensor,
                                                  const CudaVulkanExtent2D extent,
                                                  const cudaStream_t stream,
                                                  PreparedCudaImageTensor& out,
                                                  std::string& error) {
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                error = "CUDA/Vulkan image copy requires a valid 3D tensor";
                return false;
            }

            const ImageLayout layout = detectImageLayout(tensor);
            if (layout == ImageLayout::Unknown) {
                error = "CUDA/Vulkan image copy received an unsupported tensor layout";
                return false;
            }

            const int width = imageWidth(tensor, layout);
            const int height = imageHeight(tensor, layout);
            const int channels = imageChannels(tensor, layout);
            if (width != static_cast<int>(extent.width) || height != static_cast<int>(extent.height)) {
                error = std::format("CUDA/Vulkan image copy size mismatch: tensor {}x{}, target {}x{}",
                                    width,
                                    height,
                                    extent.width,
                                    extent.height);
                return false;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                error = std::format("CUDA/Vulkan image copy requires 1, 3, or 4 channels, got {}",
                                    channels);
                return false;
            }

            lfs::core::Tensor prepared = tensor;
            if (prepared.dtype() != lfs::core::DataType::UInt8 &&
                prepared.dtype() != lfs::core::DataType::Float32) {
                prepared = prepared.to(lfs::core::DataType::Float32);
            }
            if (prepared.device() != lfs::core::Device::CUDA) {
                prepared = prepared.to(lfs::core::Device::CUDA, stream);
            }
            if (!prepared.is_contiguous()) {
                prepared = prepared.contiguous();
            }

            out.tensor = std::move(prepared);
            out.width = width;
            out.height = height;
            out.channels = channels;
            out.layout = layout == ImageLayout::HWC
                             ? detail::CudaVulkanTensorLayout::Hwc
                             : detail::CudaVulkanTensorLayout::Chw;
            out.element_type = out.tensor.dtype() == lfs::core::DataType::UInt8
                                   ? detail::CudaVulkanTensorElementType::UInt8
                                   : detail::CudaVulkanTensorElementType::Float32;
            error.clear();
            return true;
        }
    } // namespace

    namespace detail {
        [[nodiscard]] cudaError_t launchCudaVulkanCopyTensorToSurface(
            cudaSurfaceObject_t surface,
            const void* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            CudaVulkanTensorElementType element_type,
            bool flip_y,
            const cudaStream_t stream);
        [[nodiscard]] cudaError_t launchCudaVulkanCopyTensorToSurfaceR32f(
            cudaSurfaceObject_t surface,
            const float* source,
            std::uint32_t width,
            std::uint32_t height,
            int channels,
            CudaVulkanTensorLayout layout,
            bool flip_y,
            const cudaStream_t stream);
    } // namespace detail

    CudaVulkanInterop::CudaVulkanInterop(CudaVulkanExternalImageImport image,
                                         CudaVulkanExternalSemaphoreImport semaphore) {
        if (!init(std::move(image), std::move(semaphore))) {
            throw std::runtime_error(last_error_);
        }
    }

    CudaVulkanInterop::~CudaVulkanInterop() {
        reset();
    }

    CudaVulkanInterop::CudaVulkanInterop(CudaVulkanInterop&& other) noexcept {
        *this = std::move(other);
    }

    CudaVulkanInterop& CudaVulkanInterop::operator=(CudaVulkanInterop&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        reset();
        cuda_mem_ = std::exchange(other.cuda_mem_, nullptr);
        cuda_mip_ = std::exchange(other.cuda_mip_, nullptr);
        cuda_array_ = std::exchange(other.cuda_array_, nullptr);
        surface_ = std::exchange(other.surface_, cudaSurfaceObject_t{});
        cuda_timeline_ = std::exchange(other.cuda_timeline_, nullptr);
        surface_copy_event_ = std::exchange(other.surface_copy_event_, nullptr);
        surface_copy_status_ = std::exchange(other.surface_copy_status_, SurfaceCopyStatus::Idle);
        surface_copy_event_recorded_ = std::exchange(other.surface_copy_event_recorded_, false);
        last_signaled_ = std::exchange(other.last_signaled_, 0);
        extent_ = std::exchange(other.extent_, {});
        format_ = std::exchange(other.format_, CudaVulkanImageFormat::Rgba8Unorm);
        upload_source_ = std::move(other.upload_source_);
        tracked_streams_ = std::move(other.tracked_streams_);
        other.tracked_streams_.clear();
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanInterop::init(CudaVulkanExternalImageImport image,
                                 CudaVulkanExternalSemaphoreImport semaphore) {
        return initImpl(std::move(image), std::move(semaphore));
    }

    bool CudaVulkanInterop::init(CudaVulkanExternalImageImport image) {
        return initImpl(std::move(image), std::nullopt);
    }

    bool CudaVulkanInterop::initImpl(
        CudaVulkanExternalImageImport image,
        std::optional<CudaVulkanExternalSemaphoreImport> semaphore) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return fail(*err);
        }
        if (!nativeHandleValid(image.memory_handle)) {
            return fail("CUDA/Vulkan external image import requires a valid memory handle");
        }
        if (semaphore && !nativeHandleValid(semaphore->semaphore_handle)) {
            return fail("CUDA/Vulkan external semaphore import requires a valid semaphore handle");
        }
        if (image.allocation_size == 0 || image.extent.width == 0 || image.extent.height == 0) {
            return fail("CUDA/Vulkan external image import requires non-zero allocation and extent");
        }
        if (!formatSupported(image.format)) {
            return fail(std::format("CUDA/Vulkan external image format {} is unsupported",
                                    formatName(image.format)));
        }
        NativeHandleOwner memory_handle(image.memory_handle);
        NativeHandleOwner semaphore_handle(
            semaphore ? semaphore->semaphore_handle : kInvalidCudaVulkanExternalHandle);

        cudaError_t status = cudaSuccess;
        if (semaphore) {
            int cuda_device = 0;
            status = cudaGetDevice(&cuda_device);
            if (status != cudaSuccess) {
                return failCuda("cudaGetDevice", status);
            }
            int timeline_interop_supported = 0;
            status = cudaDeviceGetAttribute(&timeline_interop_supported,
                                            cudaDevAttrTimelineSemaphoreInteropSupported,
                                            cuda_device);
            if (status != cudaSuccess) {
                return failCuda("cudaDeviceGetAttribute(cudaDevAttrTimelineSemaphoreInteropSupported)", status);
            }
            if (timeline_interop_supported == 0) {
                return fail("CUDA device does not support external timeline semaphore interop");
            }
        }

        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = kCudaExternalMemoryHandleType;
        memory_desc.size = image.allocation_size;
        if (image.dedicated_allocation) {
            memory_desc.flags = cudaExternalMemoryDedicated;
        }
#ifdef _WIN32
        memory_desc.handle.win32.handle = memory_handle.get();
#else
        memory_desc.handle.fd = memory_handle.get();
#endif

        status = cudaImportExternalMemory(&cuda_mem_, &memory_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaImportExternalMemory", status);
        }
#ifndef _WIN32
        memory_handle.release();
#endif

        cudaExternalMemoryMipmappedArrayDesc array_desc{};
        array_desc.offset = 0;
        array_desc.formatDesc = channelDescForFormat(image.format);
        array_desc.extent = make_cudaExtent(image.extent.width, image.extent.height, 0);
        array_desc.flags = cudaArraySurfaceLoadStore;
        array_desc.numLevels = 1;

        status = cudaExternalMemoryGetMappedMipmappedArray(&cuda_mip_, cuda_mem_, &array_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaExternalMemoryGetMappedMipmappedArray", status);
        }

        status = cudaGetMipmappedArrayLevel(&cuda_array_, cuda_mip_, 0);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaGetMipmappedArrayLevel", status);
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = cuda_array_;
        status = cudaCreateSurfaceObject(&surface_, &resource_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaCreateSurfaceObject", status);
        }

        if (semaphore) {
            // cudaExternalSemaphoreHandleDesc has no initialValue field; the imported timeline starts at
            // whatever value Vulkan created it with. Our signal contract is monotonic from 1, which only
            // holds when the Vulkan-side initial value is 0.
            assert(semaphore->initial_value == 0 &&
                   "CUDA timeline import assumes Vulkan initialValue==0; first signal is value 1");

            cudaExternalSemaphoreHandleDesc semaphore_desc{};
            semaphore_desc.type = kCudaExternalSemaphoreHandleType;
#ifdef _WIN32
            semaphore_desc.handle.win32.handle = semaphore_handle.get();
#else
            semaphore_desc.handle.fd = semaphore_handle.get();
#endif

            status = cudaImportExternalSemaphore(&cuda_timeline_, &semaphore_desc);
            if (status != cudaSuccess) {
                reset();
                return failCuda("cudaImportExternalSemaphore", status);
            }
#ifndef _WIN32
            semaphore_handle.release();
#endif
            last_signaled_ = semaphore->initial_value;
        }

        extent_ = image.extent;
        format_ = image.format;
        return true;
    }

    void CudaVulkanInterop::reset() {
        if (surface_copy_status_ == SurfaceCopyStatus::Pending &&
            surface_copy_event_recorded_ &&
            surface_copy_event_ != nullptr) {
            if (cudaEventSynchronize(surface_copy_event_) != cudaSuccess) {
                (void)cudaGetLastError();
            }
        }
        synchronizeTrackedStreams();
        upload_source_ = {};
        if (surface_copy_event_ != nullptr) {
            (void)cudaEventDestroy(surface_copy_event_);
            surface_copy_event_ = nullptr;
        }
        surface_copy_status_ = SurfaceCopyStatus::Idle;
        surface_copy_event_recorded_ = false;
        if (surface_ != cudaSurfaceObject_t{}) {
            cudaDestroySurfaceObject(surface_);
            surface_ = cudaSurfaceObject_t{};
        }
        cuda_array_ = nullptr;
        if (cuda_mip_ != nullptr) {
            cudaFreeMipmappedArray(cuda_mip_);
            cuda_mip_ = nullptr;
        }
        if (cuda_mem_ != nullptr) {
            cudaDestroyExternalMemory(cuda_mem_);
            cuda_mem_ = nullptr;
        }
        if (cuda_timeline_ != nullptr) {
            cudaDestroyExternalSemaphore(cuda_timeline_);
            cuda_timeline_ = nullptr;
        }
        last_signaled_ = 0;
        extent_ = {};
        format_ = CudaVulkanImageFormat::Rgba8Unorm;
    }

    bool CudaVulkanInterop::valid() const {
        return cuda_mem_ != nullptr &&
               cuda_mip_ != nullptr &&
               cuda_array_ != nullptr &&
               surface_ != cudaSurfaceObject_t{} &&
               extent_.width > 0 &&
               extent_.height > 0;
    }

    bool CudaVulkanInterop::copyTensorToSurface(const lfs::core::Tensor& tensor,
                                                 const cudaStream_t stream,
                                                 const bool flip_y) const {
        last_error_.clear();
        if (surface_copy_status_ == SurfaceCopyStatus::Pending) {
            return fail("CUDA/Vulkan surface copy is already pending");
        }
        surface_copy_status_ = SurfaceCopyStatus::Idle;
        surface_copy_event_recorded_ = false;
        if (!valid()) {
            return fail("CUDA/Vulkan interop target is not initialized");
        }
        PreparedCudaImageTensor prepared{};
        if (!prepareCudaImageTensor(tensor, extent_, stream, prepared, last_error_)) {
            return false;
        }
        rememberStream(stream);
        upload_source_ = std::move(prepared.tensor);

        const void* data = upload_source_.data_ptr();
        if (data == nullptr) {
            return fail("CUDA/Vulkan interop copy received a tensor with null data");
        }

        lfs::core::waitForCUDAStream(stream, upload_source_.stream());
        cudaError_t status = cudaSuccess;
        if (format_ == CudaVulkanImageFormat::R32Sfloat) {
            if (prepared.element_type != detail::CudaVulkanTensorElementType::Float32) {
                return fail("CUDA/Vulkan R32_SFLOAT surface requires a float tensor");
            }
            status = detail::launchCudaVulkanCopyTensorToSurfaceR32f(
                surface_,
                static_cast<const float*>(data),
                extent_.width,
                extent_.height,
                prepared.channels,
                prepared.layout,
                flip_y,
                stream);
        } else {
            status = detail::launchCudaVulkanCopyTensorToSurface(
                surface_,
                data,
                extent_.width,
                extent_.height,
                prepared.channels,
                prepared.layout,
                prepared.element_type,
                flip_y,
                stream);
        }
        return failCuda("copy tensor to CUDA surface", status);
    }

    bool CudaVulkanInterop::enqueueTensorToSurface(const lfs::core::Tensor& tensor,
                                                    const cudaStream_t stream,
                                                    const bool flip_y) const {
        last_error_.clear();
        if (surface_copy_status_ == SurfaceCopyStatus::Pending) {
            return fail("CUDA/Vulkan surface copy is already pending");
        }
        if (surface_copy_event_ == nullptr) {
            cudaEvent_t event = nullptr;
            const cudaError_t status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (status != cudaSuccess) {
                return failCuda("cudaEventCreateWithFlags(surface copy)", status);
            }
            surface_copy_event_ = event;
        }
        if (!copyTensorToSurface(tensor, stream, flip_y)) {
            return false;
        }

        const cudaError_t record_status = cudaEventRecord(surface_copy_event_, stream);
        if (record_status != cudaSuccess) {
            (void)failCuda("cudaEventRecord(surface copy)", record_status);
            const std::string record_error = last_error_;
            surface_copy_status_ = SurfaceCopyStatus::Pending;
            surface_copy_event_recorded_ = false;

            const cudaError_t sync_status = cudaStreamSynchronize(stream);
            if (sync_status == cudaSuccess) {
                upload_source_ = {};
                surface_copy_status_ = SurfaceCopyStatus::Idle;
                last_error_ = record_error;
            } else {
                last_error_ = std::format(
                    "{}; cudaStreamSynchronize after event record failure also failed: {} ({})",
                    record_error,
                    cudaGetErrorName(sync_status),
                    cudaGetErrorString(sync_status));
                (void)cudaGetLastError();
            }
            return false;
        }

        surface_copy_event_recorded_ = true;
        surface_copy_status_ = SurfaceCopyStatus::Pending;
        return true;
    }

    bool CudaVulkanInterop::pollSurfaceCopy(SurfaceCopyStatus& status) const {
        last_error_.clear();
        status = surface_copy_status_;
        if (surface_copy_status_ != SurfaceCopyStatus::Pending) {
            return true;
        }
        if (surface_copy_event_ == nullptr || !surface_copy_event_recorded_) {
            return fail("CUDA/Vulkan surface copy completion event is unavailable; reset is required");
        }

        const cudaError_t query_status = cudaEventQuery(surface_copy_event_);
        if (query_status == cudaErrorNotReady) {
            return true;
        }
        if (query_status != cudaSuccess) {
            return failCuda("cudaEventQuery(surface copy)", query_status);
        }

        upload_source_ = {};
        surface_copy_status_ = SurfaceCopyStatus::Complete;
        status = SurfaceCopyStatus::Complete;
        return true;
    }

    bool CudaVulkanInterop::wait(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA/Vulkan timeline semaphore is not initialized");
        }

        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        rememberStream(stream);
        const cudaError_t status = cudaWaitExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        return failCuda("cudaWaitExternalSemaphoresAsync", status);
    }

    bool CudaVulkanInterop::signal(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA/Vulkan timeline semaphore is not initialized");
        }

        assert(value > last_signaled_ && "Vulkan timeline signal values must strictly increase");
        last_signaled_ = value;

        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        rememberStream(stream);
        const cudaError_t status = cudaSignalExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream);
        return failCuda("cudaSignalExternalSemaphoresAsync", status);
    }

    void CudaVulkanInterop::rememberStream(const cudaStream_t stream) const {
        if (std::find(tracked_streams_.begin(), tracked_streams_.end(), stream) == tracked_streams_.end()) {
            tracked_streams_.push_back(stream);
        }
    }

    void CudaVulkanInterop::synchronizeTrackedStreams() {
        for (const cudaStream_t stream : tracked_streams_) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                (void)cudaGetLastError();
            }
        }
        tracked_streams_.clear();
    }

    bool CudaVulkanInterop::fail(std::string message) const {
        last_error_ = std::move(message);
        return false;
    }

    bool CudaVulkanInterop::failCuda(const char* const operation, const cudaError_t status) const {
        if (status == cudaSuccess) {
            return true;
        }
        last_error_ = std::format("{} failed: {} ({})",
                                  operation,
                                  cudaGetErrorName(status),
                                  cudaGetErrorString(status));
        return false;
    }

    // ===== CudaTimelineSemaphore =================================================

    CudaTimelineSemaphore::~CudaTimelineSemaphore() {
        reset();
    }

    CudaTimelineSemaphore::CudaTimelineSemaphore(CudaTimelineSemaphore&& other) noexcept {
        *this = std::move(other);
    }

    CudaTimelineSemaphore& CudaTimelineSemaphore::operator=(CudaTimelineSemaphore&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        cuda_timeline_ = std::exchange(other.cuda_timeline_, nullptr);
        last_signaled_ = std::exchange(other.last_signaled_, 0);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaTimelineSemaphore::init(CudaVulkanExternalSemaphoreImport semaphore) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return fail(*err);
        }
        if (!nativeHandleValid(semaphore.semaphore_handle)) {
            return fail("CUDA timeline semaphore import requires a valid handle");
        }

        // cudaExternalSemaphoreHandleDesc has no initialValue field; our signal contract is monotonic
        // from 1, which only holds when the Vulkan-side initial value is 0.
        assert(semaphore.initial_value == 0 &&
               "CUDA timeline import assumes Vulkan initialValue==0; first signal is value 1");

        NativeHandleOwner semaphore_handle(semaphore.semaphore_handle);

        cudaExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = kCudaExternalSemaphoreHandleType;
#ifdef _WIN32
        semaphore_desc.handle.win32.handle = semaphore_handle.get();
#else
        semaphore_desc.handle.fd = semaphore_handle.get();
#endif

        cudaError_t status = cudaImportExternalSemaphore(&cuda_timeline_, &semaphore_desc);
        if (status != cudaSuccess) {
            return failCuda("cudaImportExternalSemaphore", status);
        }
#ifndef _WIN32
        semaphore_handle.release();
#endif
        last_signaled_ = semaphore.initial_value;
        return true;
    }

    void CudaTimelineSemaphore::reset() {
        if (cuda_timeline_ != nullptr) {
            cudaDestroyExternalSemaphore(cuda_timeline_);
            cuda_timeline_ = nullptr;
        }
        last_signaled_ = 0;
    }

    bool CudaTimelineSemaphore::cudaSignal(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA timeline semaphore is not initialized");
        }
        assert(value > last_signaled_ && "Vulkan timeline signal values must strictly increase");
        last_signaled_ = value;

        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        return failCuda("cudaSignalExternalSemaphoresAsync",
                        cudaSignalExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream));
    }

    bool CudaTimelineSemaphore::cudaWait(const std::uint64_t value, const cudaStream_t stream) const {
        last_error_.clear();
        if (cuda_timeline_ == nullptr) {
            return fail("CUDA timeline semaphore is not initialized");
        }
        cudaExternalSemaphoreWaitParams params{};
        params.params.fence.value = value;
        return failCuda("cudaWaitExternalSemaphoresAsync",
                        cudaWaitExternalSemaphoresAsync(&cuda_timeline_, &params, 1, stream));
    }

    bool CudaTimelineSemaphore::fail(std::string message) const {
        last_error_ = std::move(message);
        return false;
    }

    bool CudaTimelineSemaphore::failCuda(const char* const operation, const cudaError_t status) const {
        if (status == cudaSuccess) {
            return true;
        }
        last_error_ = std::format("{} failed: {} ({})",
                                  operation,
                                  cudaGetErrorName(status),
                                  cudaGetErrorString(status));
        return false;
    }

    // ===== CudaVulkanBufferInterop ===============================================

    CudaVulkanBufferInterop::CudaVulkanBufferInterop(CudaVulkanExternalBufferImport buffer) {
        if (!init(std::move(buffer))) {
            throw std::runtime_error(last_error_);
        }
    }

    CudaVulkanBufferInterop::~CudaVulkanBufferInterop() {
        reset();
    }

    CudaVulkanBufferInterop::CudaVulkanBufferInterop(CudaVulkanBufferInterop&& other) noexcept {
        *this = std::move(other);
    }

    CudaVulkanBufferInterop& CudaVulkanBufferInterop::operator=(CudaVulkanBufferInterop&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        cuda_mem_ = std::exchange(other.cuda_mem_, nullptr);
        device_ptr_ = std::exchange(other.device_ptr_, nullptr);
        allocation_size_ = std::exchange(other.allocation_size_, 0);
        size_ = std::exchange(other.size_, 0);
        upload_source_ = std::move(other.upload_source_);
        last_error_ = std::move(other.last_error_);
        return *this;
    }

    bool CudaVulkanBufferInterop::init(CudaVulkanExternalBufferImport buffer) {
        reset();
        last_error_.clear();

        if (auto err = verifyCudaMatchesVulkanDevice(); err) {
            return fail(*err);
        }
        if (!nativeHandleValid(buffer.memory_handle)) {
            return fail("CUDA/Vulkan external buffer import requires a valid memory handle");
        }
        if (buffer.allocation_size == 0 || buffer.size == 0 || buffer.size > buffer.allocation_size) {
            return fail("CUDA/Vulkan external buffer import requires a non-zero size within the allocation");
        }

        NativeHandleOwner memory_handle(buffer.memory_handle);

        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = kCudaExternalMemoryHandleType;
        memory_desc.size = buffer.allocation_size;
        if (buffer.dedicated_allocation) {
            memory_desc.flags = cudaExternalMemoryDedicated;
        }
#ifdef _WIN32
        memory_desc.handle.win32.handle = memory_handle.get();
#else
        memory_desc.handle.fd = memory_handle.get();
#endif

        cudaError_t status = cudaImportExternalMemory(&cuda_mem_, &memory_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaImportExternalMemory(buffer)", status);
        }
#ifndef _WIN32
        memory_handle.release();
#endif

        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size = buffer.size;
        status = cudaExternalMemoryGetMappedBuffer(&device_ptr_, cuda_mem_, &buffer_desc);
        if (status != cudaSuccess) {
            reset();
            return failCuda("cudaExternalMemoryGetMappedBuffer", status);
        }

        allocation_size_ = buffer.allocation_size;
        size_ = buffer.size;
        return true;
    }

    void CudaVulkanBufferInterop::reset() {
        upload_source_ = {};
        if (device_ptr_ != nullptr) {
            cudaFree(device_ptr_);
            device_ptr_ = nullptr;
        }
        if (cuda_mem_ != nullptr) {
            cudaDestroyExternalMemory(cuda_mem_);
            cuda_mem_ = nullptr;
        }
        allocation_size_ = 0;
        size_ = 0;
    }

    bool CudaVulkanBufferInterop::valid() const {
        return cuda_mem_ != nullptr && device_ptr_ != nullptr && size_ > 0;
    }

    bool CudaVulkanBufferInterop::copyFromTensor(const lfs::core::Tensor& tensor,
                                                 const std::size_t byte_count,
                                                 const cudaStream_t stream) const {
        return copyFromTensor(tensor, byte_count, 0, stream);
    }

    bool CudaVulkanBufferInterop::copyFromTensor(const lfs::core::Tensor& tensor,
                                                 const std::size_t byte_count,
                                                 const std::size_t dst_offset,
                                                 const cudaStream_t stream) const {
        last_error_.clear();
        if (!valid()) {
            return fail("CUDA/Vulkan external buffer is not initialized");
        }
        if (byte_count == 0 || dst_offset > size_ || byte_count > size_ - dst_offset) {
            return fail(std::format(
                "CUDA/Vulkan buffer copy [{}, {}+{}) exceeds target {}",
                dst_offset, dst_offset, byte_count, size_));
        }
        if (!tensor.is_valid() || tensor.data_ptr() == nullptr) {
            return fail("CUDA/Vulkan buffer copy received an invalid tensor");
        }

        upload_source_ = tensor;
        if (upload_source_.device() != lfs::core::Device::CUDA) {
            upload_source_ = upload_source_.to(lfs::core::Device::CUDA, stream);
        }
        if (!upload_source_.is_contiguous()) {
            upload_source_ = upload_source_.contiguous();
        }
        if (byte_count > upload_source_.bytes()) {
            return fail(std::format("CUDA/Vulkan buffer copy requested {} bytes from {} byte tensor",
                                    byte_count,
                                    upload_source_.bytes()));
        }

        lfs::core::waitForCUDAStream(stream, upload_source_.stream());
        auto* const dst = static_cast<std::uint8_t*>(device_ptr_) + dst_offset;
        const cudaError_t status = cudaMemcpyAsync(
            dst, upload_source_.data_ptr(), byte_count, cudaMemcpyDeviceToDevice, stream);
        return failCuda("cudaMemcpyAsync(CUDA tensor -> Vulkan buffer)", status);
    }

    bool CudaVulkanBufferInterop::fail(std::string message) const {
        last_error_ = std::move(message);
        return false;
    }

    bool CudaVulkanBufferInterop::failCuda(const char* const operation, const cudaError_t status) const {
        if (status == cudaSuccess) {
            return true;
        }
        last_error_ = std::format("{} failed: {} ({})",
                                  operation,
                                  cudaGetErrorName(status),
                                  cudaGetErrorString(status));
        return false;
    }

} // namespace lfs::rendering
