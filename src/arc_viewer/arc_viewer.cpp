// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "arc_viewer.hpp"

#include "ply_loader.hpp"

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::arc {
    namespace {

        constexpr std::uint32_t kIntelVendorId = 0x8086;
        constexpr std::uint32_t kInitialWidth = 1280;
        constexpr std::uint32_t kInitialHeight = 800;

        struct PushConstants {
            glm::mat4 viewProjection{1.0F};
            float pointSize = 1.0F;
        };

        static_assert(sizeof(glm::mat4) == 64);
        static_assert(offsetof(PushConstants, pointSize) == 64);
        static_assert(sizeof(PushConstants) % 4 == 0);

        struct QueueFamilies {
            std::optional<std::uint32_t> graphics;
            std::optional<std::uint32_t> present;

            [[nodiscard]] bool complete() const {
                return graphics.has_value() && present.has_value();
            }
        };

        struct SwapchainSupport {
            VkSurfaceCapabilitiesKHR capabilities{};
            std::vector<VkSurfaceFormatKHR> formats;
            std::vector<VkPresentModeKHR> presentModes;
        };

        struct BufferResource {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
        };

        [[noreturn]] void fail(const std::string& message) {
            throw std::runtime_error(message);
        }

        void vkCheck(VkResult result, const char* operation) {
            if (result != VK_SUCCESS) {
                std::ostringstream message;
                message << operation << " failed with VkResult " << static_cast<int>(result);
                fail(message.str());
            }
        }

        std::filesystem::path executableDirectory() {
            const char* basePath = SDL_GetBasePath();
            if (!basePath || *basePath == '\0') {
                fail(std::string("SDL_GetBasePath failed: ") + SDL_GetError());
            }
            std::u8string utf8;
            const std::string_view bytes(basePath);
            utf8.reserve(bytes.size());
            for (const unsigned char byte : bytes) {
                utf8.push_back(static_cast<char8_t>(byte));
            }
            return std::filesystem::path(utf8);
        }

        std::filesystem::path pathFromUtf8(const char* text) {
            std::u8string utf8;
            const std::string_view bytes(text);
            utf8.reserve(bytes.size());
            for (const unsigned char byte : bytes) {
                utf8.push_back(static_cast<char8_t>(byte));
            }
            return std::filesystem::path(utf8);
        }

        std::string pathToUtf8(const std::filesystem::path& path) {
            const std::u8string utf8 = path.u8string();
            std::string result;
            result.reserve(utf8.size());
            for (const char8_t byte : utf8) {
                result.push_back(static_cast<char>(byte));
            }
            return result;
        }

        std::vector<std::uint32_t> readSpirv(const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input) {
                fail("Could not open SPIR-V shader beside the executable: " + pathToUtf8(path));
            }
            const std::streampos end = input.tellg();
            if (end <= 0 || (static_cast<std::uint64_t>(end) % sizeof(std::uint32_t)) != 0) {
                fail("SPIR-V shader has an invalid byte size: " + pathToUtf8(path));
            }
            std::vector<std::uint32_t> code(
                static_cast<std::size_t>(end) / sizeof(std::uint32_t));
            input.seekg(0, std::ios::beg);
            input.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(end));
            if (!input) {
                fail("Could not read SPIR-V shader: " + pathToUtf8(path));
            }
            return code;
        }

        bool hasStencilComponent(VkFormat format) {
            return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                   format == VK_FORMAT_D24_UNORM_S8_UINT;
        }

        VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) {
            constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> choices{
                VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
            for (const VkCompositeAlphaFlagBitsKHR choice : choices) {
                if ((supported & choice) != 0) {
                    return choice;
                }
            }
            fail("Vulkan surface reports no supported composite alpha mode");
        }

        const char* physicalDeviceTypeName(VkPhysicalDeviceType type) {
            switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return "CPU";
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            default:
                return "other";
            }
        }

        std::int64_t physicalDeviceTypeScore(VkPhysicalDeviceType type) {
            switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return 10'000;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return 5'000;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return 2'500;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return 1'000;
            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            default:
                return 0;
            }
        }

    } // namespace

    struct ArcViewer::Impl {
        SDL_Window* window = nullptr;
        bool sdlInitialized = false;
        bool running = true;
        bool resizeRequested = false;
        std::optional<std::uint32_t> gpuIndexOverride;

        VkInstance instance = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties physicalProperties{};
        VkPhysicalDeviceFeatures physicalFeatures{};
        VkDevice device = VK_NULL_HANDLE;
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        VkQueue presentQueue = VK_NULL_HANDLE;
        QueueFamilies queueFamilies;

        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D swapchainExtent{};
        std::vector<VkImage> swapchainImages;
        std::vector<VkImageView> swapchainImageViews;
        std::vector<VkFramebuffer> framebuffers;

        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        VkImage depthImage = VK_NULL_HANDLE;
        VkDeviceMemory depthMemory = VK_NULL_HANDLE;
        VkImageView depthImageView = VK_NULL_HANDLE;

        VkRenderPass renderPass = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence frameFence = VK_NULL_HANDLE;

        BufferResource vertexBuffer;
        std::uint32_t vertexCount = 0;

        glm::vec3 sceneCenter{0.0F};
        float sceneRadius = 1.0F;
        glm::vec3 cameraTarget{0.0F};
        float cameraDistance = 3.0F;
        float cameraYaw = glm::radians(45.0F);
        float cameraPitch = glm::radians(20.0F);
        float requestedPointSize = 2.0F;
        float devicePointSize = 1.0F;

        explicit Impl(std::optional<std::uint32_t> gpuIndex)
            : gpuIndexOverride(gpuIndex) {
            try {
                initialize();
            } catch (...) {
                cleanup();
                throw;
            }
        }

        ~Impl() {
            cleanup();
        }

        void initialize() {
            SDL_SetMainReady();
            if (!SDL_Init(SDL_INIT_VIDEO)) {
                fail(std::string("SDL_Init failed: ") + SDL_GetError());
            }
            sdlInitialized = true;

            window = SDL_CreateWindow(
                "LichtFeld Studio Arc Viewer - Drop a PLY file",
                static_cast<int>(kInitialWidth), static_cast<int>(kInitialHeight),
                SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
            if (!window) {
                fail(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
            }

            createInstance();
            if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
                fail(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
            }
            selectPhysicalDevice();
            createLogicalDevice();
            createCommandResources();
            devicePointSize = clampPointSize(requestedPointSize);
            if (!recreateSwapchain()) {
                resizeRequested = true;
            }
        }

        void createInstance() {
            Uint32 extensionCount = 0;
            const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
            if (!extensions || extensionCount == 0) {
                fail(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());
            }

            const VkApplicationInfo applicationInfo{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = nullptr,
                .pApplicationName = "LichtFeld Studio Arc Viewer",
                .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
                .pEngineName = "None",
                .engineVersion = VK_MAKE_VERSION(0, 1, 0),
                .apiVersion = VK_API_VERSION_1_0};
            const VkInstanceCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pApplicationInfo = &applicationInfo,
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = nullptr,
                .enabledExtensionCount = extensionCount,
                .ppEnabledExtensionNames = extensions};
            vkCheck(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
        }

        QueueFamilies findQueueFamilies(VkPhysicalDevice candidate) const {
            std::uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
            std::vector<VkQueueFamilyProperties> properties(count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, properties.data());

            QueueFamilies fallback;
            for (std::uint32_t index = 0; index < count; ++index) {
                const bool graphicsSupported = properties[index].queueCount > 0 &&
                                               (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
                VkBool32 presentSupported = VK_FALSE;
                vkCheck(vkGetPhysicalDeviceSurfaceSupportKHR(
                            candidate, index, surface, &presentSupported),
                        "vkGetPhysicalDeviceSurfaceSupportKHR");
                const bool canPresent =
                    properties[index].queueCount > 0 && presentSupported == VK_TRUE;
                if (graphicsSupported && canPresent) {
                    return QueueFamilies{index, index};
                }
                if (graphicsSupported && !fallback.graphics) {
                    fallback.graphics = index;
                }
                if (canPresent && !fallback.present) {
                    fallback.present = index;
                }
            }
            return fallback;
        }

        bool supportsRequiredDeviceExtensions(VkPhysicalDevice candidate) const {
            std::uint32_t count = 0;
            vkCheck(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr),
                    "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> properties(count);
            vkCheck(vkEnumerateDeviceExtensionProperties(
                        candidate, nullptr, &count, properties.data()),
                    "vkEnumerateDeviceExtensionProperties");
            return std::any_of(properties.begin(), properties.end(), [](const auto& property) {
                return std::strcmp(property.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
            });
        }

        SwapchainSupport querySwapchainSupport(VkPhysicalDevice candidate) const {
            SwapchainSupport support;
            vkCheck(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                        candidate, surface, &support.capabilities),
                    "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

            std::uint32_t formatCount = 0;
            vkCheck(vkGetPhysicalDeviceSurfaceFormatsKHR(
                        candidate, surface, &formatCount, nullptr),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR");
            support.formats.resize(formatCount);
            if (formatCount > 0) {
                vkCheck(vkGetPhysicalDeviceSurfaceFormatsKHR(
                            candidate, surface, &formatCount, support.formats.data()),
                        "vkGetPhysicalDeviceSurfaceFormatsKHR");
            }

            std::uint32_t presentModeCount = 0;
            vkCheck(vkGetPhysicalDeviceSurfacePresentModesKHR(
                        candidate, surface, &presentModeCount, nullptr),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR");
            support.presentModes.resize(presentModeCount);
            if (presentModeCount > 0) {
                vkCheck(vkGetPhysicalDeviceSurfacePresentModesKHR(
                            candidate, surface, &presentModeCount, support.presentModes.data()),
                        "vkGetPhysicalDeviceSurfacePresentModesKHR");
            }
            return support;
        }

        void selectPhysicalDevice() {
            std::uint32_t count = 0;
            vkCheck(vkEnumeratePhysicalDevices(instance, &count, nullptr),
                    "vkEnumeratePhysicalDevices");
            if (count == 0) {
                fail("No Vulkan physical device is available");
            }
            std::vector<VkPhysicalDevice> devices(count);
            vkCheck(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
                    "vkEnumeratePhysicalDevices");

            if (gpuIndexOverride && *gpuIndexOverride >= count) {
                std::ostringstream message;
                message << "Vulkan GPU index " << *gpuIndexOverride
                        << " is out of range; available indices are 0-" << (count - 1);
                fail(message.str());
            }

            std::int64_t bestScore = std::numeric_limits<std::int64_t>::min();
            std::optional<std::uint32_t> selectedIndex;
            for (std::uint32_t index = 0; index < count; ++index) {
                const VkPhysicalDevice candidate = devices[index];
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(candidate, &properties);
                std::cerr << "Vulkan GPU [" << index << "]: " << properties.deviceName
                          << " (" << physicalDeviceTypeName(properties.deviceType)
                          << ", vendor 0x" << std::hex << std::setw(4) << std::setfill('0')
                          << properties.vendorID << std::dec << std::setfill(' ') << ")\n";
                if (gpuIndexOverride && index != *gpuIndexOverride) {
                    continue;
                }

                VkPhysicalDeviceFeatures features{};
                vkGetPhysicalDeviceFeatures(candidate, &features);

                const QueueFamilies families = findQueueFamilies(candidate);
                if (!families.complete() || !supportsRequiredDeviceExtensions(candidate)) {
                    continue;
                }
                const SwapchainSupport support = querySwapchainSupport(candidate);
                if (support.formats.empty() || support.presentModes.empty() ||
                    properties.limits.maxPushConstantsSize < sizeof(PushConstants)) {
                    continue;
                }

                std::int64_t score = physicalDeviceTypeScore(properties.deviceType);
                const bool intelDiscrete = properties.vendorID == kIntelVendorId &&
                                           properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                if (intelDiscrete) {
                    score += 1'000'000;
                }
                if (families.graphics == families.present) {
                    score += 100;
                }

                if (score > bestScore) {
                    bestScore = score;
                    physicalDevice = candidate;
                    physicalProperties = properties;
                    physicalFeatures = features;
                    queueFamilies = families;
                    selectedIndex = index;
                }
            }
            if (physicalDevice == VK_NULL_HANDLE) {
                if (gpuIndexOverride) {
                    VkPhysicalDeviceProperties requestedProperties{};
                    vkGetPhysicalDeviceProperties(
                        devices[*gpuIndexOverride], &requestedProperties);
                    std::ostringstream message;
                    message << "Vulkan GPU index " << *gpuIndexOverride << " ("
                            << requestedProperties.deviceName
                            << ") does not support graphics, presentation, VK_KHR_swapchain, "
                               "and the required push constants";
                    fail(message.str());
                }
                fail("No Vulkan GPU supports graphics, presentation, VK_KHR_swapchain, and the required push constants");
            }

            const bool selectedIntelDiscrete =
                physicalProperties.vendorID == kIntelVendorId &&
                physicalProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            std::cerr << "Selected Vulkan GPU [" << *selectedIndex << "]: "
                      << physicalProperties.deviceName << " ("
                      << physicalDeviceTypeName(physicalProperties.deviceType)
                      << ", vendor 0x" << std::hex << std::setw(4) << std::setfill('0')
                      << physicalProperties.vendorID << std::dec << std::setfill(' ') << ")\n";
            if (gpuIndexOverride) {
                std::cerr << "Explicit GPU override selected index " << *gpuIndexOverride
                          << "; automatic Intel-discrete preference was bypassed.\n";
            } else if (selectedIntelDiscrete) {
                std::cerr << "Selected the preferred compatible Intel discrete GPU.\n";
            } else {
                std::cerr << "No compatible Intel discrete GPU was found; using the "
                             "highest-ranked compatible Vulkan fallback.\n";
            }
            std::cerr << "Push constants: " << sizeof(PushConstants) << " / "
                      << physicalProperties.limits.maxPushConstantsSize << " bytes\n";
        }

        void createLogicalDevice() {
            const std::set<std::uint32_t> uniqueFamilies{
                *queueFamilies.graphics, *queueFamilies.present};
            const float priority = 1.0F;
            std::vector<VkDeviceQueueCreateInfo> queueInfos;
            queueInfos.reserve(uniqueFamilies.size());
            for (const std::uint32_t family : uniqueFamilies) {
                queueInfos.push_back(VkDeviceQueueCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queueFamilyIndex = family,
                    .queueCount = 1,
                    .pQueuePriorities = &priority});
            }

            VkPhysicalDeviceFeatures enabledFeatures{};
            enabledFeatures.largePoints = physicalFeatures.largePoints;
            constexpr const char* extensions[]{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
            const VkDeviceCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
                .pQueueCreateInfos = queueInfos.data(),
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = nullptr,
                .enabledExtensionCount = 1,
                .ppEnabledExtensionNames = extensions,
                .pEnabledFeatures = &enabledFeatures};
            vkCheck(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device),
                    "vkCreateDevice");
            vkGetDeviceQueue(device, *queueFamilies.graphics, 0, &graphicsQueue);
            vkGetDeviceQueue(device, *queueFamilies.present, 0, &presentQueue);
        }

        void createCommandResources() {
            const VkCommandPoolCreateInfo poolInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = *queueFamilies.graphics};
            vkCheck(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool),
                    "vkCreateCommandPool");

            const VkCommandBufferAllocateInfo allocateInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1};
            vkCheck(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer),
                    "vkAllocateCommandBuffers");

            const VkSemaphoreCreateInfo semaphoreInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0};
            vkCheck(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailable),
                    "vkCreateSemaphore");
            vkCheck(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinished),
                    "vkCreateSemaphore");

            const VkFenceCreateInfo fenceInfo{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT};
            vkCheck(vkCreateFence(device, &fenceInfo, nullptr, &frameFence),
                    "vkCreateFence");
        }

        VkSurfaceFormatKHR chooseSurfaceFormat(
            const std::vector<VkSurfaceFormatKHR>& formats) const {
            for (const VkSurfaceFormatKHR& format : formats) {
                if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return format;
                }
            }
            for (const VkSurfaceFormatKHR& format : formats) {
                if (format.format == VK_FORMAT_R8G8B8A8_SRGB &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return format;
                }
            }
            return formats.front();
        }

        VkPresentModeKHR choosePresentMode(
            const std::vector<VkPresentModeKHR>& modes) const {
            const auto mailbox = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR);
            return mailbox != modes.end() ? VK_PRESENT_MODE_MAILBOX_KHR
                                          : VK_PRESENT_MODE_FIFO_KHR;
        }

        std::optional<VkExtent2D> chooseExtent(
            const VkSurfaceCapabilitiesKHR& capabilities) const {
            if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
                if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0) {
                    return std::nullopt;
                }
                return capabilities.currentExtent;
            }
            int width = 0;
            int height = 0;
            if (!SDL_GetWindowSizeInPixels(window, &width, &height)) {
                fail(std::string("SDL_GetWindowSizeInPixels failed: ") + SDL_GetError());
            }
            if (width <= 0 || height <= 0) {
                return std::nullopt;
            }
            return VkExtent2D{
                std::clamp(static_cast<std::uint32_t>(width),
                           capabilities.minImageExtent.width,
                           capabilities.maxImageExtent.width),
                std::clamp(static_cast<std::uint32_t>(height),
                           capabilities.minImageExtent.height,
                           capabilities.maxImageExtent.height)};
        }

        bool recreateSwapchain() {
            const SwapchainSupport support = querySwapchainSupport(physicalDevice);
            const std::optional<VkExtent2D> extent = chooseExtent(support.capabilities);
            if (!extent) {
                return false;
            }
            vkCheck(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
            destroySwapchainResources();
            createSwapchain(support, *extent);
            createSwapchainImageViews();
            depthFormat = chooseDepthFormat();
            createRenderPass();
            createPipeline();
            createDepthResources();
            createFramebuffers();
            resizeRequested = false;
            return true;
        }

        void createSwapchain(const SwapchainSupport& support, VkExtent2D extent) {
            const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
            const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
            std::uint32_t imageCount = support.capabilities.minImageCount + 1;
            if (support.capabilities.maxImageCount > 0) {
                imageCount = std::min(imageCount, support.capabilities.maxImageCount);
            }

            const std::array<std::uint32_t, 2> familyIndices{
                *queueFamilies.graphics, *queueFamilies.present};
            VkSwapchainCreateInfoKHR createInfo{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = 0,
                .surface = surface,
                .minImageCount = imageCount,
                .imageFormat = surfaceFormat.format,
                .imageColorSpace = surfaceFormat.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .preTransform = support.capabilities.currentTransform,
                .compositeAlpha = chooseCompositeAlpha(support.capabilities.supportedCompositeAlpha),
                .presentMode = presentMode,
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE};
            if (queueFamilies.graphics != queueFamilies.present) {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(familyIndices.size());
                createInfo.pQueueFamilyIndices = familyIndices.data();
            }
            vkCheck(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain),
                    "vkCreateSwapchainKHR");
            swapchainFormat = surfaceFormat.format;
            swapchainExtent = extent;

            std::uint32_t actualCount = 0;
            vkCheck(vkGetSwapchainImagesKHR(device, swapchain, &actualCount, nullptr),
                    "vkGetSwapchainImagesKHR");
            swapchainImages.resize(actualCount);
            vkCheck(vkGetSwapchainImagesKHR(
                        device, swapchain, &actualCount, swapchainImages.data()),
                    "vkGetSwapchainImagesKHR");
        }

        void createSwapchainImageViews() {
            swapchainImageViews.resize(swapchainImages.size(), VK_NULL_HANDLE);
            for (std::size_t index = 0; index < swapchainImages.size(); ++index) {
                const VkImageViewCreateInfo viewInfo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .image = swapchainImages[index],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = swapchainFormat,
                    .components = VkComponentMapping{
                        VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY,
                        VK_COMPONENT_SWIZZLE_IDENTITY},
                    .subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                vkCheck(vkCreateImageView(
                            device, &viewInfo, nullptr, &swapchainImageViews[index]),
                        "vkCreateImageView");
            }
        }

        VkFormat chooseDepthFormat() const {
            constexpr std::array<VkFormat, 3> candidates{
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D24_UNORM_S8_UINT,
                VK_FORMAT_D16_UNORM};
            for (const VkFormat candidate : candidates) {
                VkFormatProperties properties{};
                vkGetPhysicalDeviceFormatProperties(physicalDevice, candidate, &properties);
                if ((properties.optimalTilingFeatures &
                     VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                    return candidate;
                }
            }
            fail("No supported Vulkan depth format is available");
        }

        void createRenderPass() {
            const std::array<VkAttachmentDescription, 2> attachments{
                VkAttachmentDescription{
                    .flags = 0,
                    .format = swapchainFormat,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                VkAttachmentDescription{
                    .flags = 0,
                    .format = depthFormat,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
            const VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            const VkAttachmentReference depthReference{
                1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            const VkSubpassDescription subpass{
                .flags = 0,
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .inputAttachmentCount = 0,
                .pInputAttachments = nullptr,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorReference,
                .pResolveAttachments = nullptr,
                .pDepthStencilAttachment = &depthReference,
                .preserveAttachmentCount = 0,
                .pPreserveAttachments = nullptr};
            const VkSubpassDependency dependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dependencyFlags = 0};
            const VkRenderPassCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments = attachments.data(),
                .subpassCount = 1,
                .pSubpasses = &subpass,
                .dependencyCount = 1,
                .pDependencies = &dependency};
            vkCheck(vkCreateRenderPass(device, &createInfo, nullptr, &renderPass),
                    "vkCreateRenderPass");
        }

        VkShaderModule createShaderModule(const std::vector<std::uint32_t>& code) const {
            const VkShaderModuleCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = code.size() * sizeof(std::uint32_t),
                .pCode = code.data()};
            VkShaderModule module = VK_NULL_HANDLE;
            vkCheck(vkCreateShaderModule(device, &createInfo, nullptr, &module),
                    "vkCreateShaderModule");
            return module;
        }

        void createPipeline() {
            const std::filesystem::path shaderDirectory = executableDirectory();
            const std::vector<std::uint32_t> vertexCode =
                readSpirv(shaderDirectory / "point.vert.spv");
            const std::vector<std::uint32_t> fragmentCode =
                readSpirv(shaderDirectory / "point.frag.spv");
            const VkShaderModule vertexModule = createShaderModule(vertexCode);
            VkShaderModule fragmentModule = VK_NULL_HANDLE;
            try {
                fragmentModule = createShaderModule(fragmentCode);
            } catch (...) {
                vkDestroyShaderModule(device, vertexModule, nullptr);
                throw;
            }

            const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{
                VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertexModule,
                    .pName = "main",
                    .pSpecializationInfo = nullptr},
                VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragmentModule,
                    .pName = "main",
                    .pSpecializationInfo = nullptr}};

            const VkVertexInputBindingDescription binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
            const std::array<VkVertexInputAttributeDescription, 2> attributes{
                VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, x)},
                VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, red)}};
            const VkPipelineVertexInputStateCreateInfo vertexInput{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &binding,
                .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size()),
                .pVertexAttributeDescriptions = attributes.data()};
            const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                .primitiveRestartEnable = VK_FALSE};
            const VkPipelineViewportStateCreateInfo viewportState{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = nullptr,
                .scissorCount = 1,
                .pScissors = nullptr};
            const VkPipelineRasterizationStateCreateInfo rasterization{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.0F,
                .depthBiasClamp = 0.0F,
                .depthBiasSlopeFactor = 0.0F,
                .lineWidth = 1.0F};
            const VkPipelineMultisampleStateCreateInfo multisampling{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 0.0F,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE};
            const VkPipelineDepthStencilStateCreateInfo depthStencil{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .front = {},
                .back = {},
                .minDepthBounds = 0.0F,
                .maxDepthBounds = 1.0F};
            const VkPipelineColorBlendAttachmentState blendAttachment{
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
            const VkPipelineColorBlendStateCreateInfo colorBlend{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &blendAttachment,
                .blendConstants = {0.0F, 0.0F, 0.0F, 0.0F}};
            constexpr std::array<VkDynamicState, 2> dynamicStates{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            const VkPipelineDynamicStateCreateInfo dynamicState{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
                .pDynamicStates = dynamicStates.data()};

            const VkPushConstantRange pushRange{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(PushConstants)};
            const VkPipelineLayoutCreateInfo layoutInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = 0,
                .pSetLayouts = nullptr,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pushRange};
            const VkResult layoutResult =
                vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);
            if (layoutResult != VK_SUCCESS) {
                vkDestroyShaderModule(device, fragmentModule, nullptr);
                vkDestroyShaderModule(device, vertexModule, nullptr);
                vkCheck(layoutResult, "vkCreatePipelineLayout");
            }

            const VkGraphicsPipelineCreateInfo pipelineInfo{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
                .pStages = shaderStages.data(),
                .pVertexInputState = &vertexInput,
                .pInputAssemblyState = &inputAssembly,
                .pTessellationState = nullptr,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlend,
                .pDynamicState = &dynamicState,
                .layout = pipelineLayout,
                .renderPass = renderPass,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = -1};
            const VkResult pipelineResult = vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
            vkDestroyShaderModule(device, fragmentModule, nullptr);
            vkDestroyShaderModule(device, vertexModule, nullptr);
            vkCheck(pipelineResult, "vkCreateGraphicsPipelines");
        }

        std::uint32_t findMemoryType(
            std::uint32_t typeFilter,
            VkMemoryPropertyFlags required,
            VkMemoryPropertyFlags preferred,
            VkMemoryPropertyFlags* selectedProperties = nullptr) const {
            VkPhysicalDeviceMemoryProperties properties{};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

            auto find = [&](VkMemoryPropertyFlags desired) -> std::optional<std::uint32_t> {
                for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
                    const VkMemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
                    if ((typeFilter & (1U << index)) != 0 && (flags & desired) == desired) {
                        if (selectedProperties) {
                            *selectedProperties = flags;
                        }
                        return index;
                    }
                }
                return std::nullopt;
            };

            if (const auto preferredIndex = find(required | preferred)) {
                return *preferredIndex;
            }
            if (const auto requiredIndex = find(required)) {
                return *requiredIndex;
            }
            fail("No compatible Vulkan memory type is available");
        }

        void createDepthResources() {
            const VkImageCreateInfo imageInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = depthFormat,
                .extent = VkExtent3D{swapchainExtent.width, swapchainExtent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
            vkCheck(vkCreateImage(device, &imageInfo, nullptr, &depthImage), "vkCreateImage");

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, depthImage, &requirements);
            const VkMemoryAllocateInfo allocateInfo{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = requirements.size,
                .memoryTypeIndex = findMemoryType(
                    requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
            vkCheck(vkAllocateMemory(device, &allocateInfo, nullptr, &depthMemory),
                    "vkAllocateMemory(depth)");
            vkCheck(vkBindImageMemory(device, depthImage, depthMemory, 0),
                    "vkBindImageMemory");

            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (hasStencilComponent(depthFormat)) {
                aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            const VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = depthImage,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = depthFormat,
                .components = VkComponentMapping{
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = VkImageSubresourceRange{aspect, 0, 1, 0, 1}};
            vkCheck(vkCreateImageView(device, &viewInfo, nullptr, &depthImageView),
                    "vkCreateImageView(depth)");
        }

        void createFramebuffers() {
            framebuffers.resize(swapchainImageViews.size(), VK_NULL_HANDLE);
            for (std::size_t index = 0; index < swapchainImageViews.size(); ++index) {
                const std::array<VkImageView, 2> attachments{
                    swapchainImageViews[index], depthImageView};
                const VkFramebufferCreateInfo createInfo{
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .renderPass = renderPass,
                    .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                    .pAttachments = attachments.data(),
                    .width = swapchainExtent.width,
                    .height = swapchainExtent.height,
                    .layers = 1};
                vkCheck(vkCreateFramebuffer(
                            device, &createInfo, nullptr, &framebuffers[index]),
                        "vkCreateFramebuffer");
            }
        }

        float clampPointSize(float value) const {
            float minimum = physicalProperties.limits.pointSizeRange[0];
            float maximum = physicalProperties.limits.pointSizeRange[1];
            if (physicalFeatures.largePoints != VK_TRUE) {
                minimum = 1.0F;
                maximum = 1.0F;
            }
            value = std::clamp(value, minimum, maximum);
            const float granularity = physicalProperties.limits.pointSizeGranularity;
            if (granularity > 0.0F && maximum > minimum) {
                value = minimum + std::round((value - minimum) / granularity) * granularity;
            }
            return std::clamp(value, minimum, maximum);
        }

        BufferResource uploadVertices(const std::vector<Vertex>& vertices) {
            if (vertices.empty() ||
                vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
                vertices.size() > std::numeric_limits<VkDeviceSize>::max() / sizeof(Vertex)) {
                fail("Point cloud vertex count cannot be represented by Vulkan");
            }
            const VkDeviceSize byteSize =
                static_cast<VkDeviceSize>(vertices.size()) * sizeof(Vertex);
            BufferResource resource;
            try {
                const VkBufferCreateInfo bufferInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = byteSize,
                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr};
                vkCheck(vkCreateBuffer(device, &bufferInfo, nullptr, &resource.buffer),
                        "vkCreateBuffer(vertex)");

                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
                VkMemoryPropertyFlags selectedProperties = 0;
                const std::uint32_t memoryType = findMemoryType(
                    requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &selectedProperties);
                const VkMemoryAllocateInfo allocateInfo{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .allocationSize = requirements.size,
                    .memoryTypeIndex = memoryType};
                vkCheck(vkAllocateMemory(device, &allocateInfo, nullptr, &resource.memory),
                        "vkAllocateMemory(vertex)");
                vkCheck(vkBindBufferMemory(device, resource.buffer, resource.memory, 0),
                        "vkBindBufferMemory");

                void* mapped = nullptr;
                vkCheck(vkMapMemory(device, resource.memory, 0, VK_WHOLE_SIZE, 0, &mapped),
                        "vkMapMemory(vertex)");
                std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(byteSize));
                if ((selectedProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                    const VkMappedMemoryRange range{
                        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                        .pNext = nullptr,
                        .memory = resource.memory,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE};
                    const VkResult flushResult = vkFlushMappedMemoryRanges(device, 1, &range);
                    vkUnmapMemory(device, resource.memory);
                    vkCheck(flushResult, "vkFlushMappedMemoryRanges");
                } else {
                    vkUnmapMemory(device, resource.memory);
                }
            } catch (...) {
                destroyBuffer(resource);
                throw;
            }
            return resource;
        }

        void destroyBuffer(BufferResource& resource) noexcept {
            if (device != VK_NULL_HANDLE && resource.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, resource.buffer, nullptr);
            }
            if (device != VK_NULL_HANDLE && resource.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, resource.memory, nullptr);
            }
            resource = {};
        }

        void loadPointCloud(const std::filesystem::path& path) {
            const PointCloud cloud = loadPly(path);
            BufferResource nextBuffer = uploadVertices(cloud.vertices);
            try {
                vkCheck(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
            } catch (...) {
                destroyBuffer(nextBuffer);
                throw;
            }
            destroyBuffer(vertexBuffer);
            vertexBuffer = std::exchange(nextBuffer, {});
            vertexCount = static_cast<std::uint32_t>(cloud.vertices.size());

            const glm::vec3 minimum{
                cloud.framingBoundsMin[0],
                cloud.framingBoundsMin[1],
                cloud.framingBoundsMin[2]};
            const glm::vec3 maximum{
                cloud.framingBoundsMax[0],
                cloud.framingBoundsMax[1],
                cloud.framingBoundsMax[2]};
            sceneCenter = (minimum + maximum) * 0.5F;
            sceneRadius = std::max(glm::length(maximum - minimum) * 0.5F, 1.0e-4F);
            cameraTarget = sceneCenter;
            cameraDistance = std::max(sceneRadius * 2.6F, 0.01F);
            cameraYaw = glm::radians(45.0F);
            cameraPitch = glm::radians(20.0F);

            const std::string title = "LichtFeld Studio Arc Viewer - " +
                                      pathToUtf8(path.filename()) + " (" +
                                      std::to_string(vertexCount) + " points)";
            if (!SDL_SetWindowTitle(window, title.c_str())) {
                std::cerr << "SDL_SetWindowTitle failed: " << SDL_GetError() << '\n';
            }
            std::cerr << "Loaded " << vertexCount << " vertices from "
                      << pathToUtf8(path) << '\n';
        }

        glm::vec3 cameraPosition() const {
            const float cosPitch = std::cos(cameraPitch);
            const glm::vec3 direction{
                cosPitch * std::sin(cameraYaw),
                std::sin(cameraPitch),
                cosPitch * std::cos(cameraYaw)};
            return cameraTarget + direction * cameraDistance;
        }

        glm::mat4 viewProjection() const {
            const glm::vec3 position = cameraPosition();
            const glm::mat4 view = glm::lookAt(position, cameraTarget, glm::vec3{0.0F, 1.0F, 0.0F});
            const float aspect = static_cast<float>(swapchainExtent.width) /
                                 static_cast<float>(swapchainExtent.height);
            const float targetOffset = glm::length(cameraTarget - sceneCenter);
            const float nearPlane = std::max(
                std::min(cameraDistance * 0.001F, sceneRadius * 0.01F), 1.0e-5F);
            const float farPlane = std::max(
                cameraDistance + targetOffset + sceneRadius * 8.0F,
                nearPlane + 1.0F);
            glm::mat4 projection = glm::perspective(
                glm::radians(45.0F), aspect, nearPlane, farPlane);
            projection[1][1] *= -1.0F;
            return projection * view;
        }

        void orbit(float xRelative, float yRelative) {
            cameraYaw -= xRelative * 0.005F;
            cameraPitch -= yRelative * 0.005F;
            cameraPitch = std::clamp(
                cameraPitch, glm::radians(-89.0F), glm::radians(89.0F));
        }

        void pan(float xRelative, float yRelative) {
            const glm::vec3 forward = glm::normalize(cameraTarget - cameraPosition());
            glm::vec3 right = glm::cross(forward, glm::vec3{0.0F, 1.0F, 0.0F});
            if (glm::dot(right, right) < 1.0e-8F) {
                right = glm::vec3{1.0F, 0.0F, 0.0F};
            } else {
                right = glm::normalize(right);
            }
            const glm::vec3 up = glm::normalize(glm::cross(right, forward));
            const float scale = cameraDistance * 0.0015F;
            cameraTarget += (-right * xRelative + up * yRelative) * scale;
        }

        void zoom(float yRelative) {
            cameraDistance *= std::exp(yRelative * 0.01F);
            cameraDistance = std::clamp(
                cameraDistance,
                std::max(sceneRadius * 0.001F, 1.0e-6F),
                std::max(sceneRadius * 10'000.0F, 1.0F));
        }

        void reportRecoverableError(const std::string& message) const {
            std::cerr << "Error: " << message << '\n';
            if (!SDL_ShowSimpleMessageBox(
                    SDL_MESSAGEBOX_ERROR, "LichtFeld Studio Arc Viewer", message.c_str(), window)) {
                std::cerr << "SDL_ShowSimpleMessageBox failed: " << SDL_GetError() << '\n';
            }
        }

        void handleEvents() {
            SDL_Event event{};
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                case SDL_EVENT_QUIT:
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                    running = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (event.key.scancode == SDL_SCANCODE_ESCAPE) {
                        running = false;
                    }
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
                    resizeRequested = true;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if ((event.motion.state & SDL_BUTTON_LMASK) != 0) {
                        orbit(event.motion.xrel, event.motion.yrel);
                    } else if ((event.motion.state & SDL_BUTTON_MMASK) != 0) {
                        pan(event.motion.xrel, event.motion.yrel);
                    } else if ((event.motion.state & SDL_BUTTON_RMASK) != 0) {
                        zoom(event.motion.yrel);
                    }
                    break;
                case SDL_EVENT_MOUSE_WHEEL: {
                    float wheel = event.wheel.y;
                    if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
                        wheel = -wheel;
                    }
                    requestedPointSize *= std::exp(wheel * 0.14F);
                    requestedPointSize = std::clamp(requestedPointSize, 0.01F, 4096.0F);
                    devicePointSize = clampPointSize(requestedPointSize);
                    break;
                }
                case SDL_EVENT_DROP_FILE:
                    if (event.drop.data) {
                        try {
                            loadPointCloud(pathFromUtf8(event.drop.data));
                        } catch (const std::exception& error) {
                            reportRecoverableError(error.what());
                        }
                    }
                    break;
                default:
                    break;
                }
            }
        }

        void recordCommandBuffer(std::uint32_t imageIndex) {
            const VkCommandBufferBeginInfo beginInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pInheritanceInfo = nullptr};
            vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

            const std::array<VkClearValue, 2> clearValues{
                VkClearValue{.color = {{0.025F, 0.028F, 0.032F, 1.0F}}},
                VkClearValue{.depthStencil = {1.0F, 0}}};
            const VkRenderPassBeginInfo renderPassInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = renderPass,
                .framebuffer = framebuffers[imageIndex],
                .renderArea = VkRect2D{{0, 0}, swapchainExtent},
                .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
                .pClearValues = clearValues.data()};
            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            const VkViewport viewport{
                .x = 0.0F,
                .y = 0.0F,
                .width = static_cast<float>(swapchainExtent.width),
                .height = static_cast<float>(swapchainExtent.height),
                .minDepth = 0.0F,
                .maxDepth = 1.0F};
            const VkRect2D scissor{{0, 0}, swapchainExtent};
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            if (vertexBuffer.buffer != VK_NULL_HANDLE && vertexCount > 0) {
                constexpr VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, &offset);
                const PushConstants pushConstants{viewProjection(), devicePointSize};
                vkCmdPushConstants(
                    commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                    0, sizeof(pushConstants), &pushConstants);
                vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
            }

            vkCmdEndRenderPass(commandBuffer);
            vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        }

        void drawFrame() {
            if (resizeRequested || swapchain == VK_NULL_HANDLE) {
                if (!recreateSwapchain()) {
                    SDL_Delay(16);
                    return;
                }
            }

            vkCheck(vkWaitForFences(
                        device, 1, &frameFence, VK_TRUE,
                        std::numeric_limits<std::uint64_t>::max()),
                    "vkWaitForFences");
            std::uint32_t imageIndex = 0;
            const VkResult acquireResult = vkAcquireNextImageKHR(
                device, swapchain, std::numeric_limits<std::uint64_t>::max(),
                imageAvailable, VK_NULL_HANDLE, &imageIndex);
            if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
                resizeRequested = true;
                return;
            }
            if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
                vkCheck(acquireResult, "vkAcquireNextImageKHR");
            }

            vkCheck(vkResetFences(device, 1, &frameFence), "vkResetFences");
            vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
            recordCommandBuffer(imageIndex);

            constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            const VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &imageAvailable,
                .pWaitDstStageMask = &waitStage,
                .commandBufferCount = 1,
                .pCommandBuffers = &commandBuffer,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &renderFinished};
            vkCheck(vkQueueSubmit(graphicsQueue, 1, &submitInfo, frameFence),
                    "vkQueueSubmit");

            const VkPresentInfoKHR presentInfo{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &renderFinished,
                .swapchainCount = 1,
                .pSwapchains = &swapchain,
                .pImageIndices = &imageIndex,
                .pResults = nullptr};
            const VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
            if (presentQueue != graphicsQueue &&
                (presentResult == VK_SUCCESS || presentResult == VK_SUBOPTIMAL_KHR ||
                 presentResult == VK_ERROR_OUT_OF_DATE_KHR)) {
                vkCheck(vkQueueWaitIdle(presentQueue), "vkQueueWaitIdle(present)");
            }
            if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
                presentResult == VK_SUBOPTIMAL_KHR ||
                acquireResult == VK_SUBOPTIMAL_KHR) {
                resizeRequested = true;
            } else {
                vkCheck(presentResult, "vkQueuePresentKHR");
            }
        }

        int run() {
            while (running) {
                handleEvents();
                if (running) {
                    drawFrame();
                }
            }
            if (device != VK_NULL_HANDLE) {
                vkCheck(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
            }
            return 0;
        }

        void destroySwapchainResources() {
            if (device == VK_NULL_HANDLE) {
                return;
            }
            for (const VkFramebuffer framebuffer : framebuffers) {
                if (framebuffer != VK_NULL_HANDLE) {
                    vkDestroyFramebuffer(device, framebuffer, nullptr);
                }
            }
            framebuffers.clear();
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = VK_NULL_HANDLE;
            }
            if (renderPass != VK_NULL_HANDLE) {
                vkDestroyRenderPass(device, renderPass, nullptr);
                renderPass = VK_NULL_HANDLE;
            }
            if (depthImageView != VK_NULL_HANDLE) {
                vkDestroyImageView(device, depthImageView, nullptr);
                depthImageView = VK_NULL_HANDLE;
            }
            if (depthImage != VK_NULL_HANDLE) {
                vkDestroyImage(device, depthImage, nullptr);
                depthImage = VK_NULL_HANDLE;
            }
            if (depthMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, depthMemory, nullptr);
                depthMemory = VK_NULL_HANDLE;
            }
            for (const VkImageView imageView : swapchainImageViews) {
                if (imageView != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, imageView, nullptr);
                }
            }
            swapchainImageViews.clear();
            swapchainImages.clear();
            if (swapchain != VK_NULL_HANDLE) {
                vkDestroySwapchainKHR(device, swapchain, nullptr);
                swapchain = VK_NULL_HANDLE;
            }
        }

        void cleanup() noexcept {
            if (device != VK_NULL_HANDLE) {
                (void)vkDeviceWaitIdle(device);
                destroyBuffer(vertexBuffer);
                destroySwapchainResources();
                if (frameFence != VK_NULL_HANDLE) {
                    vkDestroyFence(device, frameFence, nullptr);
                }
                if (renderFinished != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, renderFinished, nullptr);
                }
                if (imageAvailable != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, imageAvailable, nullptr);
                }
                if (commandPool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device, commandPool, nullptr);
                }
                vkDestroyDevice(device, nullptr);
                device = VK_NULL_HANDLE;
            }
            if (surface != VK_NULL_HANDLE && instance != VK_NULL_HANDLE) {
                SDL_Vulkan_DestroySurface(instance, surface, nullptr);
                surface = VK_NULL_HANDLE;
            }
            if (instance != VK_NULL_HANDLE) {
                vkDestroyInstance(instance, nullptr);
                instance = VK_NULL_HANDLE;
            }
            if (window) {
                SDL_DestroyWindow(window);
                window = nullptr;
            }
            if (sdlInitialized) {
                SDL_Quit();
                sdlInitialized = false;
            }
        }
    };

    ArcViewer::ArcViewer(std::optional<std::uint32_t> gpuIndex)
        : impl_(std::make_unique<Impl>(gpuIndex)) {}

    ArcViewer::~ArcViewer() = default;

    void ArcViewer::loadPointCloud(const std::filesystem::path& path) {
        impl_->loadPointCloud(path);
    }

    int ArcViewer::run() {
        return impl_->run();
    }

} // namespace lfs::arc
