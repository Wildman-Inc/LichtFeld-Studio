/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_image_uploader.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/image_layout.hpp"
#include "vulkan_viewport_pass.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <string>
#include <vector>

namespace lfs::vis {
    namespace {
        constexpr VkDeviceSize kScenePixelBytes = 4;

        struct SceneUploadDescription {
            lfs::rendering::ImageLayout layout = lfs::rendering::ImageLayout::Unknown;
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::size_t width = 0;
            std::size_t height = 0;
            std::size_t channels = 0;
            VkDeviceSize row_bytes = 0;
            VkDeviceSize upload_bytes = 0;
        };

        [[nodiscard]] const char* formatName(const VkFormat format) {
            switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM:
                return "RGBA8_UNORM";
            case VK_FORMAT_R32_SFLOAT:
                return "R32_SFLOAT";
            default:
                return "unsupported";
            }
        }

        [[nodiscard]] bool describeSceneTensor(const lfs::core::Tensor& tensor,
                                               const glm::ivec2 expected_size,
                                               SceneUploadDescription& out,
                                               std::string& error) {
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                error = "scene image staging requires a valid 3D tensor";
                return false;
            }

            const auto layout = lfs::rendering::detectImageLayout(tensor);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                error = std::format("unsupported scene image layout [{}, {}, {}]",
                                    tensor.size(0), tensor.size(1), tensor.size(2));
                return false;
            }

            const std::size_t width = layout == lfs::rendering::ImageLayout::HWC
                                          ? tensor.size(1)
                                          : tensor.size(2);
            const std::size_t height = layout == lfs::rendering::ImageLayout::HWC
                                           ? tensor.size(0)
                                           : tensor.size(1);
            const std::size_t channels = layout == lfs::rendering::ImageLayout::HWC
                                             ? tensor.size(2)
                                             : tensor.size(0);
            if (width == 0 || height == 0 ||
                width > std::numeric_limits<std::uint32_t>::max() ||
                height > std::numeric_limits<std::uint32_t>::max()) {
                error = std::format("unsupported scene image dimensions {}x{}", width, height);
                return false;
            }
            if (expected_size.x <= 0 || expected_size.y <= 0 ||
                width != static_cast<std::size_t>(expected_size.x) ||
                height != static_cast<std::size_t>(expected_size.y)) {
                error = std::format("scene image size mismatch: tensor {}x{}, target {}x{}",
                                    width, height, expected_size.x, expected_size.y);
                return false;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                error = std::format("unsupported scene image channel count {}; expected 1, 3, or 4",
                                    channels);
                return false;
            }
            if (tensor.dtype() != lfs::core::DataType::UInt8 &&
                tensor.dtype() != lfs::core::DataType::Float32) {
                error = std::format("unsupported scene image dtype {}; expected uint8 or float32",
                                    lfs::core::dtype_name(tensor.dtype()));
                return false;
            }

            if (width > std::numeric_limits<std::size_t>::max() / height) {
                error = "scene image dimensions overflow host address space";
                return false;
            }
            const std::size_t pixels = width * height;
            if (pixels > std::numeric_limits<std::size_t>::max() / kScenePixelBytes ||
                pixels > std::numeric_limits<VkDeviceSize>::max() / kScenePixelBytes) {
                error = "scene image byte size overflows Vulkan buffer size";
                return false;
            }

            out.layout = layout;
            out.format = tensor.dtype() == lfs::core::DataType::Float32 && channels == 1
                             ? VK_FORMAT_R32_SFLOAT
                             : VK_FORMAT_R8G8B8A8_UNORM;
            out.width = width;
            out.height = height;
            out.channels = channels;
            out.row_bytes = static_cast<VkDeviceSize>(width) * kScenePixelBytes;
            out.upload_bytes = static_cast<VkDeviceSize>(pixels) * kScenePixelBytes;
            error.clear();
            return true;
        }

        [[nodiscard]] std::uint8_t floatToByte(const float value) {
            if (std::isnan(value) || value <= 0.0f) {
                return 0;
            }
            if (value >= 1.0f) {
                return 255;
            }
            return static_cast<std::uint8_t>(value * 255.0f + 0.5f);
        }

        [[nodiscard]] bool packSceneTensor(const lfs::core::Tensor& tensor,
                                           const SceneUploadDescription& description,
                                           const bool flip_y,
                                           void* const destination,
                                           std::string& error) {
            if (destination == nullptr) {
                error = "scene image staging allocation is not mapped";
                return false;
            }

            try {
                lfs::core::Tensor host = tensor.device() == lfs::core::Device::CPU
                                             ? tensor.contiguous()
                                             : tensor.to(lfs::core::Device::CPU).contiguous();
                if (!host.is_valid() || host.device() != lfs::core::Device::CPU ||
                    !host.is_contiguous() || host.dtype() != tensor.dtype()) {
                    error = "failed to materialize a contiguous CPU scene image tensor";
                    return false;
                }

                const std::size_t width = description.width;
                const std::size_t height = description.height;
                const std::size_t channels = description.channels;
                const auto sourceIndex = [&](const std::size_t x,
                                             const std::size_t y,
                                             const std::size_t channel) {
                    if (description.layout == lfs::rendering::ImageLayout::HWC) {
                        return (y * width + x) * channels + channel;
                    }
                    return (channel * height + y) * width + x;
                };

                if (description.format == VK_FORMAT_R32_SFLOAT) {
                    const float* const source = host.ptr<float>();
                    auto* const output = static_cast<float*>(destination);
                    if (source == nullptr) {
                        error = "CPU R32 scene image tensor has no data";
                        return false;
                    }
                    for (std::size_t source_y = 0; source_y < height; ++source_y) {
                        const std::size_t destination_y = flip_y ? height - 1 - source_y : source_y;
                        std::memcpy(output + destination_y * width,
                                    source + source_y * width,
                                    static_cast<std::size_t>(description.row_bytes));
                    }
                    error.clear();
                    return true;
                }

                auto* const output = static_cast<std::uint8_t*>(destination);
                if (host.dtype() == lfs::core::DataType::UInt8) {
                    const std::uint8_t* const source = host.ptr<std::uint8_t>();
                    if (source == nullptr) {
                        error = "CPU RGBA8 scene image tensor has no data";
                        return false;
                    }
                    if (description.layout == lfs::rendering::ImageLayout::HWC && channels == 4) {
                        for (std::size_t source_y = 0; source_y < height; ++source_y) {
                            const std::size_t destination_y = flip_y ? height - 1 - source_y : source_y;
                            std::memcpy(output + destination_y * description.row_bytes,
                                        source + source_y * description.row_bytes,
                                        static_cast<std::size_t>(description.row_bytes));
                        }
                        error.clear();
                        return true;
                    }

                    for (std::size_t source_y = 0; source_y < height; ++source_y) {
                        const std::size_t destination_y = flip_y ? height - 1 - source_y : source_y;
                        for (std::size_t x = 0; x < width; ++x) {
                            std::uint8_t* const pixel = output + (destination_y * width + x) * 4;
                            pixel[0] = source[sourceIndex(x, source_y, 0)];
                            pixel[1] = channels > 1 ? source[sourceIndex(x, source_y, 1)] : 0;
                            pixel[2] = channels > 2 ? source[sourceIndex(x, source_y, 2)] : 0;
                            pixel[3] = channels > 3 ? source[sourceIndex(x, source_y, 3)] : 255;
                        }
                    }
                    error.clear();
                    return true;
                }

                const float* const source = host.ptr<float>();
                if (source == nullptr) {
                    error = "CPU float scene image tensor has no data";
                    return false;
                }
                for (std::size_t source_y = 0; source_y < height; ++source_y) {
                    const std::size_t destination_y = flip_y ? height - 1 - source_y : source_y;
                    for (std::size_t x = 0; x < width; ++x) {
                        std::uint8_t* const pixel = output + (destination_y * width + x) * 4;
                        pixel[0] = floatToByte(source[sourceIndex(x, source_y, 0)]);
                        pixel[1] = channels > 1 ? floatToByte(source[sourceIndex(x, source_y, 1)]) : 0;
                        pixel[2] = channels > 2 ? floatToByte(source[sourceIndex(x, source_y, 2)]) : 0;
                        pixel[3] = channels > 3 ? floatToByte(source[sourceIndex(x, source_y, 3)]) : 255;
                    }
                }
                error.clear();
                return true;
            } catch (const std::exception& exception) {
                error = std::format("CPU scene image copy failed: {}", exception.what());
                return false;
            } catch (...) {
                error = "CPU scene image copy failed with an unknown exception";
                return false;
            }
        }
    } // namespace

    struct VulkanSceneImageUploader::Impl {
        struct UploadSlot {
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            void* staging_mapped = nullptr;
            VkDeviceSize staging_capacity = 0;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            bool in_flight = false;
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkSampler scene_sampler = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkCommandPool upload_command_pool = VK_NULL_HANDLE;
        std::vector<UploadSlot> upload_slots;

        VkImage scene_image = VK_NULL_HANDLE;
        VmaAllocation scene_image_allocation = VK_NULL_HANDLE;
        VkImageView scene_image_view = VK_NULL_HANDLE;
        VulkanImageBarrierTracker scene_image_barriers;
        glm::ivec2 scene_image_size{0, 0};
        VkFormat scene_image_format = VK_FORMAT_UNDEFINED;
        std::string scene_image_vram_label;
        bool scene_image_external = false;
        std::uint64_t scene_image_external_generation = 0;
        bool logged_staging_fallback = false;
        std::string last_upload_error;

        ~Impl() { shutdown(); }

        [[nodiscard]] bool init(VulkanContext& vulkan_context, const VkSampler sampler) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            context = &vulkan_context;
            device = vulkan_context.device();
            allocator = vulkan_context.allocator();
            scene_sampler = sampler;
            graphics_queue = vulkan_context.graphicsQueue();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE || scene_sampler == VK_NULL_HANDLE) {
                LOG_ERROR("Vulkan scene image uploader requires an initialized Vulkan context");
                return false;
            }
            return true;
        }

        void reportUploadError(std::string error) {
            if (error != last_upload_error) {
                LOG_ERROR("Vulkan viewport scene image staging upload failed: {}", error);
                last_upload_error = std::move(error);
            }
        }

        void clearUploadError() {
            last_upload_error.clear();
        }

        [[nodiscard]] bool waitForUploadSlot(UploadSlot& slot) {
            if (!slot.in_flight) {
                return true;
            }
            const VkResult result = vkWaitForFences(
                device, 1, &slot.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            if (result != VK_SUCCESS) {
                reportUploadError(std::format("waiting for staging fence failed ({})",
                                              static_cast<int>(result)));
                return false;
            }
            slot.in_flight = false;
            return true;
        }

        [[nodiscard]] bool waitForAllUploads() {
            bool success = true;
            for (auto& slot : upload_slots) {
                success = waitForUploadSlot(slot) && success;
            }
            return success;
        }

        void destroyUploadResources() {
            (void)waitForAllUploads();
            for (auto& slot : upload_slots) {
                if (slot.staging_buffer != VK_NULL_HANDLE) {
                    if (slot.staging_mapped != nullptr) {
                        vmaUnmapMemory(allocator, slot.staging_allocation);
                    }
                    vmaDestroyBuffer(allocator, slot.staging_buffer, slot.staging_allocation);
                }
                if (slot.fence != VK_NULL_HANDLE) {
                    vkDestroyFence(device, slot.fence, nullptr);
                }
                slot = {};
            }
            upload_slots.clear();
            if (upload_command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, upload_command_pool, nullptr);
                upload_command_pool = VK_NULL_HANDLE;
            }
        }

        void shutdown() {
            if (device != VK_NULL_HANDLE) {
                destroySceneImage();
                destroyUploadResources();
            }
            scene_sampler = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            context = nullptr;
        }

        void clearSceneImageBinding() {
            scene_image_barriers.forgetImage(scene_image);
            scene_image = VK_NULL_HANDLE;
            scene_image_allocation = VK_NULL_HANDLE;
            scene_image_view = VK_NULL_HANDLE;
            scene_image_size = {0, 0};
            scene_image_format = VK_FORMAT_UNDEFINED;
            scene_image_vram_label.clear();
            scene_image_external = false;
            scene_image_external_generation = 0;
        }

        void updateSceneDescriptor(const VkDescriptorSet scene_descriptor_set,
                                   const VkImageView image_view,
                                   const VkImageLayout image_layout) const {
            if (scene_descriptor_set == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorImageInfo descriptor_info{};
            descriptor_info.sampler = scene_sampler;
            descriptor_info.imageView = image_view;
            descriptor_info.imageLayout = image_layout;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = scene_descriptor_set;
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &descriptor_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void waitForOwnedSceneImage() {
            (void)waitForAllUploads();
            if (context != nullptr && !context->waitForSubmittedFrames()) {
                LOG_WARN("Vulkan scene image replacement could not wait for submitted frames: {}",
                         context->lastError());
            }
        }

        void destroySceneImage() {
            if (scene_image_external) {
                clearSceneImageBinding();
                return;
            }
            if (scene_image != VK_NULL_HANDLE) {
                waitForOwnedSceneImage();
            }
            if (scene_image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, scene_image_view, nullptr);
            }
            if (scene_image != VK_NULL_HANDLE) {
                if (!scene_image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.scene_image.image",
                        scene_image_vram_label,
                        0);
                }
                vmaDestroyImage(allocator, scene_image, scene_image_allocation);
            }
            clearSceneImageBinding();
        }

        [[nodiscard]] bool ensureSceneImage(const glm::ivec2 size, const VkFormat format) {
            if (scene_image != VK_NULL_HANDLE && scene_image_size == size && scene_image_format == format) {
                return true;
            }
            destroySceneImage();

            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = format;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo created_allocation_info{};
            if (vmaCreateImage(allocator,
                               &image_info,
                               &allocation_info,
                               &scene_image,
                               &scene_image_allocation,
                               &created_allocation_info) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }
            vmaSetAllocationName(allocator, scene_image_allocation, "Viewport scene image");
            scene_image_vram_label = std::format("{}:{}x{}", formatName(format), size.x, size.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_image.image",
                scene_image_vram_label,
                static_cast<std::size_t>(created_allocation_info.size));

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = scene_image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, nullptr, &scene_image_view) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }

            scene_image_size = size;
            scene_image_format = format;
            scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            return true;
        }

        [[nodiscard]] bool ensureUploadResources(std::string& error) {
            if (upload_command_pool != VK_NULL_HANDLE) {
                return true;
            }
            if (context == nullptr || graphics_queue == VK_NULL_HANDLE) {
                error = "Vulkan graphics queue is unavailable";
                return false;
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = context->graphicsQueueFamily();
            if (vkCreateCommandPool(device, &pool_info, nullptr, &upload_command_pool) != VK_SUCCESS) {
                error = "failed to create the scene image upload command pool";
                return false;
            }
            upload_slots.resize(std::max<std::size_t>(1, context->framesInFlight()));
            return true;
        }

        [[nodiscard]] bool ensureUploadSlot(UploadSlot& slot, std::string& error) {
            if (slot.command_buffer != VK_NULL_HANDLE && slot.fence != VK_NULL_HANDLE) {
                return true;
            }

            VkCommandBufferAllocateInfo command_info{};
            command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command_info.commandPool = upload_command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &command_info, &slot.command_buffer) != VK_SUCCESS) {
                error = "failed to allocate a scene image upload command buffer";
                return false;
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (vkCreateFence(device, &fence_info, nullptr, &slot.fence) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &slot.command_buffer);
                slot.command_buffer = VK_NULL_HANDLE;
                error = "failed to create a scene image upload fence";
                return false;
            }
            return true;
        }

        [[nodiscard]] bool ensureStagingBuffer(UploadSlot& slot,
                                               const VkDeviceSize bytes,
                                               std::string& error) {
            if (slot.staging_buffer != VK_NULL_HANDLE &&
                slot.staging_mapped != nullptr &&
                slot.staging_capacity >= bytes) {
                return true;
            }
            if (slot.staging_buffer != VK_NULL_HANDLE) {
                if (slot.staging_mapped != nullptr) {
                    vmaUnmapMemory(allocator, slot.staging_allocation);
                }
                vmaDestroyBuffer(allocator, slot.staging_buffer, slot.staging_allocation);
                slot.staging_buffer = VK_NULL_HANDLE;
                slot.staging_allocation = VK_NULL_HANDLE;
                slot.staging_mapped = nullptr;
                slot.staging_capacity = 0;
            }

            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = bytes;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo created_info{};
            if (vmaCreateBuffer(allocator,
                                &buffer_info,
                                &allocation_info,
                                &slot.staging_buffer,
                                &slot.staging_allocation,
                                &created_info) != VK_SUCCESS ||
                created_info.pMappedData == nullptr) {
                if (slot.staging_buffer != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(allocator, slot.staging_buffer, slot.staging_allocation);
                }
                slot.staging_buffer = VK_NULL_HANDLE;
                slot.staging_allocation = VK_NULL_HANDLE;
                error = "failed to allocate mapped scene image staging memory";
                return false;
            }
            vmaSetAllocationName(allocator, slot.staging_allocation, "Viewport scene image staging");
            slot.staging_mapped = created_info.pMappedData;
            slot.staging_capacity = bytes;
            return true;
        }

        [[nodiscard]] bool uploadStagedSceneImage(const VulkanViewportPassParams& params,
                                                  const SceneUploadDescription& description,
                                                  const VkDescriptorSet scene_descriptor_set,
                                                  std::string& error) {
            if (!ensureUploadResources(error) ||
                !ensureSceneImage(params.scene_image_size, description.format)) {
                if (error.empty()) {
                    error = std::format("failed to create the {} scene image",
                                        formatName(description.format));
                }
                return false;
            }

            UploadSlot& slot = upload_slots[params.frame_slot % upload_slots.size()];
            if (!ensureUploadSlot(slot, error) || !waitForUploadSlot(slot) ||
                !ensureStagingBuffer(slot, description.upload_bytes, error)) {
                if (error.empty()) {
                    error = "failed to prepare a scene image staging slot";
                }
                return false;
            }
            if (!packSceneTensor(*params.scene_image,
                                 description,
                                 params.scene_image_flip_y,
                                 slot.staging_mapped,
                                 error)) {
                return false;
            }
            const VkResult flush_result =
                vmaFlushAllocation(allocator, slot.staging_allocation, 0, description.upload_bytes);
            if (flush_result != VK_SUCCESS) {
                error = std::format("flushing scene image staging memory failed ({})",
                                    static_cast<int>(flush_result));
                return false;
            }

            if (vkResetFences(device, 1, &slot.fence) != VK_SUCCESS ||
                vkResetCommandBuffer(slot.command_buffer, 0) != VK_SUCCESS) {
                error = "failed to reset the scene image upload command state";
                return false;
            }
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(slot.command_buffer, &begin_info) != VK_SUCCESS) {
                error = "failed to begin the scene image upload command buffer";
                return false;
            }

            const VkImageLayout previous_layout = scene_image_barriers.imageLayout(scene_image);
            scene_image_barriers.transitionImage(slot.command_buffer,
                                                 scene_image,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.bufferOffset = 0;
            copy.bufferRowLength = 0;
            copy.bufferImageHeight = 0;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageOffset = {0, 0, 0};
            copy.imageExtent = {static_cast<std::uint32_t>(description.width),
                                static_cast<std::uint32_t>(description.height),
                                1};
            vkCmdCopyBufferToImage(slot.command_buffer,
                                   slot.staging_buffer,
                                   scene_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy);
            scene_image_barriers.transitionImage(slot.command_buffer,
                                                 scene_image,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS) {
                scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, previous_layout);
                error = "failed to end the scene image upload command buffer";
                return false;
            }
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &slot.command_buffer;
            const VkResult submit_result = vkQueueSubmit(graphics_queue, 1, &submit_info, slot.fence);
            if (submit_result != VK_SUCCESS) {
                scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, previous_layout);
                error = std::format("submitting the scene image upload failed ({})",
                                    static_cast<int>(submit_result));
                return false;
            }

            // The viewport frame is submitted later on this same queue, so queue order
            // makes the transfer and final shader-read transition visible without a
            // per-frame CPU wait. The slot fence is waited only when that slot is reused.
            slot.in_flight = true;
            updateSceneDescriptor(scene_descriptor_set,
                                  scene_image_view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            error.clear();
            return true;
        }

        [[nodiscard]] bool bindExternalSceneImage(const VulkanViewportPassParams& params,
                                                  const VkDescriptorSet scene_descriptor_set) {
            if (params.external_scene_image == VK_NULL_HANDLE ||
                params.external_scene_image_view == VK_NULL_HANDLE ||
                params.scene_image_size.x <= 0 ||
                params.scene_image_size.y <= 0) {
                return false;
            }
            if (scene_image_external &&
                scene_image == params.external_scene_image &&
                scene_image_view == params.external_scene_image_view &&
                scene_image_size == params.scene_image_size &&
                scene_image_barriers.imageLayout(scene_image, VK_IMAGE_LAYOUT_UNDEFINED) == params.external_scene_image_layout &&
                scene_image_external_generation == params.external_scene_image_generation) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }

            destroySceneImage();
            scene_image = params.external_scene_image;
            scene_image_view = params.external_scene_image_view;
            scene_image_size = params.scene_image_size;
            scene_image_external = true;
            scene_image_external_generation = params.external_scene_image_generation;
            scene_image_barriers.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, params.external_scene_image_layout);
            updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
        }

        void upload(const VulkanViewportPassParams& params, const VkDescriptorSet scene_descriptor_set) {
            const bool has_external_image =
                params.external_scene_image != VK_NULL_HANDLE &&
                params.external_scene_image_view != VK_NULL_HANDLE;
            if ((!params.scene_image && !has_external_image) ||
                params.scene_image_size.x <= 0 || params.scene_image_size.y <= 0) {
                // Release the binding — scene_image_view points into externally-owned
                // interop slots that the caller has already vkDestroyImage'd. Leaving
                // it set keeps hasImage()==true and the viewport pass would sample from
                // a freed image, faulting the device.
                destroySceneImage();
                return;
            }
            if (has_external_image) {
                if (!bindExternalSceneImage(params, scene_descriptor_set)) {
                    LOG_ERROR("Failed to bind external Vulkan viewport scene image");
                } else {
                    clearUploadError();
                }
                return;
            }
            if (scene_image_external) {
                destroySceneImage();
            }

            SceneUploadDescription description{};
            std::string error;
            if (!describeSceneTensor(*params.scene_image, params.scene_image_size, description, error)) {
                reportUploadError(std::move(error));
                destroySceneImage();
                return;
            }

            if (!logged_staging_fallback) {
                LOG_INFO("GPU/Vulkan scene interop image unavailable; using CPU/Vulkan staging fallback");
                logged_staging_fallback = true;
            }
            if (!uploadStagedSceneImage(params, description, scene_descriptor_set, error)) {
                reportUploadError(std::move(error));
                destroySceneImage();
                return;
            }
            clearUploadError();
        }

        [[nodiscard]] bool hasImage() const {
            return scene_image_view != VK_NULL_HANDLE;
        }
    };

    VulkanSceneImageUploader::VulkanSceneImageUploader()
        : impl_(std::make_unique<Impl>()) {}

    VulkanSceneImageUploader::~VulkanSceneImageUploader() {
        if (impl_) {
            impl_->shutdown();
        }
    }

    VulkanSceneImageUploader::VulkanSceneImageUploader(VulkanSceneImageUploader&&) noexcept = default;

    VulkanSceneImageUploader& VulkanSceneImageUploader::operator=(VulkanSceneImageUploader&&) noexcept = default;

    bool VulkanSceneImageUploader::init(VulkanContext& context, const VkSampler scene_sampler) {
        return impl_->init(context, scene_sampler);
    }

    void VulkanSceneImageUploader::shutdown() {
        impl_->shutdown();
    }

    void VulkanSceneImageUploader::upload(const VulkanViewportPassParams& params,
                                          const VkDescriptorSet scene_descriptor_set) {
        impl_->upload(params, scene_descriptor_set);
    }

    bool VulkanSceneImageUploader::hasImage() const {
        return impl_->hasImage();
    }
} // namespace lfs::vis
