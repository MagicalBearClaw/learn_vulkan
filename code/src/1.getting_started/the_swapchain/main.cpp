// 1.4 The Swapchain
//
// Adds to 1.3: a swapchain and a view onto each of its images.
//
// The swapchain is the set of images the window system will actually display, plus
// the rules for handing them back and forth. You do not create the images; the
// presentation engine owns them and lends them to you one at a time.
//
// Three decisions define it, and each one is a negotiation with what the surface
// actually supports:
//
//   format        how a pixel is stored, and whether the hardware converts to sRGB
//   present mode  what happens when you produce frames faster than the display
//                 consumes them
//   extent        how big the images are, in pixels
//
// A swapchain is also not permanent. Resize the window and it becomes invalid, so
// rebuilding it is a normal part of running, not an error path.

#include <vkc/capture.hpp>
#include <vkc/check.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// The one device extension this series needs. Dynamic rendering and synchronization2
// were folded into core in Vulkan 1.3, so unlike a 1.0-era tutorial there is nothing
// else to ask for here.
constexpr std::array kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

bool layer_available(std::string_view name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
    return std::ranges::any_of(layers, [name](const VkLayerProperties& layer) {
        return name == layer.layerName;
    });
}

bool instance_extension_available(std::string_view name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()));
    return std::ranges::any_of(extensions, [name](const VkExtensionProperties& ext) {
        return name == ext.extensionName;
    });
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user_data*/) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        spdlog::error("[vulkan] {}", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        spdlog::warn("[vulkan] {}", data->pMessage);
    } else {
        spdlog::info("[vulkan] {}", data->pMessage);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info() {
    return VkDebugUtilsMessengerCreateInfoEXT{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback,
        .pUserData = nullptr,
    };
}

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "other";
    }
}

bool device_supports_extensions(VkPhysicalDevice gpu) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> available(count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, available.data()));

    return std::ranges::all_of(kRequiredDeviceExtensions, [&](const char* required) {
        return std::ranges::any_of(available, [required](const VkExtensionProperties& e) {
            return std::strcmp(required, e.extensionName) == 0;
        });
    });
}

// A queue family that can do graphics, compute and presentation at once.
//
// Vulkan lets you split work across specialised families -- a transfer-only queue for
// uploads, an async-compute queue -- and doing so is a real optimisation. It also
// means transferring ownership of every resource that crosses between them. That is a
// chapter of its own much later; for now, one queue that does everything removes an
// entire category of bug from the next fifty chapters.
std::optional<uint32_t> find_universal_queue_family(VkPhysicalDevice gpu,
                                                    VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        constexpr VkQueueFlags wanted = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((families[i].queueFlags & wanted) != wanted) {
            continue;
        }
        // Being able to draw does not imply being able to present to *this* surface.
        // They are separate questions and Vulkan makes you ask both.
        VkBool32 can_present = VK_FALSE;
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &can_present));
        if (can_present == VK_TRUE) {
            return i;
        }
    }
    return std::nullopt;
}

int score_device(const VkPhysicalDeviceProperties& props) {
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

// An 8-bit-per-channel sRGB target. Asking for an sRGB *format* means the hardware
// converts from linear to sRGB as it writes, which is gamma correction done properly
// and for free. The gamma chapter explains why that matters; for now it is simply the
// right default, and picking a UNORM format here would quietly make everything later
// in the series too dark.
const char* format_name(VkFormat format) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_SRGB: return "VK_FORMAT_B8G8R8A8_SRGB";
        case VK_FORMAT_R8G8B8A8_SRGB: return "VK_FORMAT_R8G8B8A8_SRGB";
        case VK_FORMAT_B8G8R8A8_UNORM: return "VK_FORMAT_B8G8R8A8_UNORM";
        case VK_FORMAT_R8G8B8A8_UNORM: return "VK_FORMAT_R8G8B8A8_UNORM";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
        default: return "some other format";
    }
}

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& available) {
    for (const VkSurfaceFormatKHR& candidate : available) {
        const bool wanted = candidate.format == VK_FORMAT_B8G8R8A8_SRGB ||
                            candidate.format == VK_FORMAT_R8G8B8A8_SRGB;
        if (wanted && candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return candidate;
        }
    }
    // The specification guarantees at least one format, so this is always safe.
    return available.front();
}

// FIFO is a queue: finished frames wait their turn and are shown on a display refresh.
// It is the only mode every implementation must support, and it never tears.
// MAILBOX is the same guarantee except that a newer frame replaces a waiting one
// instead of queueing behind it, which trades GPU work for latency. Take it if it is
// offered.
VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& available) {
    const bool has_mailbox =
        std::ranges::find(available, VK_PRESENT_MODE_MAILBOX_KHR) != available.end();
    return has_mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D window_size) {
    // A currentExtent of 0xFFFFFFFF is the window system saying "you choose".
    // Anything else is it saying "this is the size, take it".
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }
    return VkExtent2D{
        .width = std::clamp(window_size.width, caps.minImageExtent.width,
                            caps.maxImageExtent.width),
        .height = std::clamp(window_size.height, caps.minImageExtent.height,
                             caps.maxImageExtent.height),
    };
}

class SwapchainApp {
public:
    explicit SwapchainApp(const vkc::Args& args) : args_(args) {
        init_window();
        init_instance();
        create_surface();
        select_physical_device();
        create_device();
        create_allocator();
        create_swapchain();
    }

    ~SwapchainApp() {
        // Strict reverse order. Vulkan will not warn you at run time if you get this
        // wrong, but the validation layers will, loudly.
        destroy_swapchain();
        if (allocator_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator_);
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
        }
        if (surface_ != VK_NULL_HANDLE) {
            SDL_Vulkan_DestroySurface(instance_, surface_, nullptr);
        }
        if (debug_messenger_ != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
        if (window_ != nullptr) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    SwapchainApp(const SwapchainApp&) = delete;
    SwapchainApp& operator=(const SwapchainApp&) = delete;
    SwapchainApp(SwapchainApp&&) = delete;
    SwapchainApp& operator=(SwapchainApp&&) = delete;

    int run() {
        report();

        uint64_t frames = 0;
        bool running = true;
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT ||
                    event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED ||
                    (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)) {
                    running = false;
                } else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                    // The old swapchain is now the wrong size. Rebuilding here is
                    // simple because nothing is in flight yet; from 1.6 onward there
                    // will be frames on the GPU to wait for first.
                    recreate_swapchain();
                }
            }
            ++frames;
            if (args_.frame_limit != 0 && frames >= args_.frame_limit) {
                running = false;
            }
        }
        return 0;
    }

private:
    void init_window() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(std::format("SDL_Init failed: {}", SDL_GetError()));
        }
        // Resizable, except on a --screenshot run: tiling window managers choose the size
        // of a resizable window themselves, which would make the book's reference images
        // depend on whatever else is open. A fixed-size window gets the size asked for.
        window_ = SDL_CreateWindow("LearnVulkan - The Swapchain",
                                   static_cast<int>(args_.width),
                                   static_cast<int>(args_.height),
                                   SDL_WINDOW_VULKAN |
                                   (args_.screenshot.empty() ? SDL_WINDOW_RESIZABLE : 0));
        if (window_ == nullptr) {
            throw std::runtime_error(
                std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
        }
    }

    void init_instance() {
        VK_CHECK(volkInitialize());

        const VkApplicationInfo app_info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "LearnVulkan",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "LearnVulkan",
            .engineVersion = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = VK_API_VERSION_1_3,
        };

        uint32_t sdl_extension_count = 0;
        const char* const* sdl_extensions =
            SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
        if (sdl_extensions == nullptr) {
            throw std::runtime_error(std::format(
                "SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
        }
        std::vector<const char*> extensions(sdl_extensions,
                                            sdl_extensions + sdl_extension_count);

        validation_enabled_ = args_.validation && layer_available(kValidationLayer);
        debug_utils_enabled_ =
            instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils_enabled_) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        std::vector<const char*> layers;
        if (validation_enabled_) {
            layers.push_back(kValidationLayer);
        }

        const VkDebugUtilsMessengerCreateInfoEXT messenger_info = debug_messenger_info();
        const bool debug_at_creation = validation_enabled_ && debug_utils_enabled_;

        const VkInstanceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = debug_at_creation ? &messenger_info : nullptr,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };
        VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));
        volkLoadInstanceOnly(instance_);

        if (debug_at_creation) {
            VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &messenger_info, nullptr,
                                                    &debug_messenger_));
        }
    }

    void create_surface() {
        // The surface is the bridge between Vulkan and the window system. It belongs
        // to the instance, not the device, because which GPUs can present to it is
        // one of the things we are about to ask.
        if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
            throw std::runtime_error(
                std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
        }
    }

    void select_physical_device() {
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
        if (count == 0) {
            throw std::runtime_error(
                "No Vulkan-capable GPU found. Check your graphics driver.");
        }
        std::vector<VkPhysicalDevice> devices(count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

        spdlog::info("Found {} physical device(s):", count);

        int best_score = -1;
        std::string rejections;

        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            spdlog::info("  {} ({})", props.deviceName, device_type_name(props.deviceType));

            if (props.apiVersion < VK_API_VERSION_1_3) {
                rejections += std::format("\n  {}: reports Vulkan {}.{}, needs 1.3",
                                          props.deviceName,
                                          VK_API_VERSION_MAJOR(props.apiVersion),
                                          VK_API_VERSION_MINOR(props.apiVersion));
                continue;
            }
            if (!device_supports_extensions(candidate)) {
                rejections +=
                    std::format("\n  {}: no VK_KHR_swapchain", props.deviceName);
                continue;
            }
            const std::optional<uint32_t> family =
                find_universal_queue_family(candidate, surface_);
            if (!family.has_value()) {
                rejections += std::format(
                    "\n  {}: no queue family with graphics + compute + present",
                    props.deviceName);
                continue;
            }

            const int score = score_device(props);
            if (score > best_score) {
                best_score = score;
                gpu_ = candidate;
                gpu_properties_ = props;
                queue_family_ = *family;
            }
        }

        if (gpu_ == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No suitable GPU. Every device was rejected:" + rejections);
        }
    }

    void create_device() {
        // Queues are not created individually; you ask a family for a number of them
        // and Vulkan hands them over with the device. Priority matters only when you
        // request several, and even then it is a hint.
        const float priority = 1.0F;
        const VkDeviceQueueCreateInfo queue_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = queue_family_,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        // Features are opt-in, and these structs have dozens of fields each. Value
        // initialise to all-false and switch on only what is used: asking for a
        // feature you do not need can cost performance, and on some drivers it can
        // cost device creation outright.
        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.synchronization2 = VK_TRUE;   // the barrier form used from 1.5
        features13.dynamicRendering = VK_TRUE;   // no render passes, ever

        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features12.pNext = &features13;
        features12.bufferDeviceAddress = VK_TRUE;
        features12.descriptorIndexing = VK_TRUE;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features12;
        features2.features.samplerAnisotropy = VK_TRUE;

        const VkDeviceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features2,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
            // Device layers stopped being a thing in Vulkan 1.0.13. The instance
            // layer covers the device too.
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount =
                static_cast<uint32_t>(kRequiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = kRequiredDeviceExtensions.data(),
            // Must be null when VkPhysicalDeviceFeatures2 is chained onto pNext.
            .pEnabledFeatures = nullptr,
        };

        VK_CHECK(vkCreateDevice(gpu_, &create_info, nullptr, &device_));

        // Device-level entry points -- vkGetDeviceQueue included -- are null until
        // this runs, so it has to happen before the very next line rather than later.
        volkLoadDevice(device_);

        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    }

    void create_allocator() {
        // Vulkan's own memory API makes you match memory types to requirements, then
        // sub-allocate out of large blocks yourself, because the driver will not.
        // That is real engineering, and it is not graphics. VMA does it properly.
        VmaVulkanFunctions functions{};
        functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        const VmaAllocatorCreateInfo info{
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = gpu_,
            .device = device_,
            .preferredLargeHeapBlockSize = 0,
            .pAllocationCallbacks = nullptr,
            .pDeviceMemoryCallbacks = nullptr,
            .pHeapSizeLimit = nullptr,
            .pVulkanFunctions = &functions,
            .instance = instance_,
            .vulkanApiVersion = VK_API_VERSION_1_3,
            .pTypeExternalMemoryHandleTypes = nullptr,
        };
        VK_CHECK(vmaCreateAllocator(&info, &allocator_));
    }

    void create_swapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surface_, &caps));

        uint32_t format_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &format_count,
                                                      nullptr));
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &format_count,
                                                      formats.data()));

        uint32_t mode_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &mode_count,
                                                           nullptr));
        std::vector<VkPresentModeKHR> modes(mode_count);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &mode_count,
                                                           modes.data()));

        const VkSurfaceFormatKHR surface_format = choose_surface_format(formats);
        present_mode_ = choose_present_mode(modes);
        format_ = surface_format.format;

        int pixel_width = 0;
        int pixel_height = 0;
        SDL_GetWindowSizeInPixels(window_, &pixel_width, &pixel_height);
        extent_ = choose_extent(caps, VkExtent2D{static_cast<uint32_t>(pixel_width),
                                                 static_cast<uint32_t>(pixel_height)});

        // One more than the minimum, so that the CPU always has an image to draw into
        // while the presentation engine is busy with another. A maxImageCount of 0
        // means there is no upper limit.
        uint32_t image_count = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
            image_count = caps.maxImageCount;
        }

        const VkSwapchainCreateInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = format_,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent_,
            // Always 1 unless you are rendering stereo for a headset.
            .imageArrayLayers = 1,
            // COLOR_ATTACHMENT to render into them; TRANSFER_DST so a later chapter
            // can blit a finished offscreen image on instead; TRANSFER_SRC so the
            // screenshot tooling can copy one back out.
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            // One queue family does everything here, so the images never need to be
            // shared between families.
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .preTransform = caps.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode_,
            // Lets the driver skip pixels another window is covering.
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };

        VK_CHECK(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_));

        // We asked for a minimum; the driver may well have given us more.
        uint32_t actual_count = 0;
        VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count, nullptr));
        images_.resize(actual_count);
        VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count,
                                         images_.data()));

        // A VkImage cannot be rendered into directly. A view says which part of it to
        // use and how to interpret the bits -- here, all of it, as a 2D colour image.
        views_.resize(actual_count);
        for (uint32_t i = 0; i < actual_count; ++i) {
            const VkImageViewCreateInfo view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = images_[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = format_,
                .components = {},  // all VK_COMPONENT_SWIZZLE_IDENTITY
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &views_[i]));
        }
    }

    void destroy_swapchain() noexcept {
        for (VkImageView view : views_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        views_.clear();
        // The images belong to the swapchain. Destroying it frees them; destroying
        // them yourself is an error.
        images_.clear();

        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
    }

    void recreate_swapchain() {
        // Nothing may be destroyed while the GPU could still be using it. Waiting for
        // the whole device is the blunt version of this; a later chapter does it
        // without the stall.
        VK_CHECK(vkDeviceWaitIdle(device_));
        destroy_swapchain();
        create_swapchain();
        spdlog::info("Swapchain rebuilt at {}x{}", extent_.width, extent_.height);
    }

    void report() const {
        spdlog::info("Chose {} ({})", gpu_properties_.deviceName,
                     device_type_name(gpu_properties_.deviceType));
        spdlog::info("  driver Vulkan {}.{}.{}",
                     VK_API_VERSION_MAJOR(gpu_properties_.apiVersion),
                     VK_API_VERSION_MINOR(gpu_properties_.apiVersion),
                     VK_API_VERSION_PATCH(gpu_properties_.apiVersion));
        spdlog::info("  universal queue family: {}", queue_family_);

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(gpu_, &memory);
        spdlog::info("  memory heaps:");
        for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
            const bool device_local =
                (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            spdlog::info("    heap {}: {} MiB{}", i,
                         memory.memoryHeaps[i].size / (1024 * 1024),
                         device_local ? " (device local)" : "");
        }
        spdlog::info("Swapchain:");
        spdlog::info("  {} images at {}x{}", images_.size(), extent_.width,
                     extent_.height);
        spdlog::info("  format: {}", format_name(format_));
        spdlog::info("  present mode: {}",
                     present_mode_ == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO");
        spdlog::info(
            "Images exist, but nothing has been recorded into them yet. Try resizing "
            "the window - the swapchain is rebuilt each time.");
    }

    vkc::Args args_;
    SDL_Window* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties gpu_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;     // owned by the swapchain
    std::vector<VkImageView> views_;  // owned by us

    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        SwapchainApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
