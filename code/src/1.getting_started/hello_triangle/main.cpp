// 1.7 Hello Triangle
//
// Adds to 1.6: shader modules, a pipeline layout, a graphics pipeline, and a draw
// call. At the end of this chapter there is finally a triangle on the screen.
//
// The new idea is the *pipeline object*. In OpenGL you set state one call at a time --
// bind a shader, enable blending, set a depth function -- and the driver works out
// what that combination means at draw time, every draw. Vulkan makes you declare the
// entire configuration up front and compiles it into a VkPipeline. Shaders, blending,
// depth testing, the primitive topology and the vertex layout are all baked in
// together.
//
// That is why pipeline creation is such a large function, and why it is a one-off
// cost: the expensive work happens here rather than inside the frame loop.
//
// Almost all of it is fixed at creation. The two things deliberately left dynamic are
// the viewport and the scissor rectangle, so that resizing the window does not mean
// rebuilding the pipeline.

#include <vkc/capture.hpp>
#include <vkc/check.hpp>
#include <vkc/paths.hpp>

#include <slang-com-ptr.h>
#include <slang.h>

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
#include <string>
#include <string_view>
#include <vector>

namespace {

// How far the CPU may run ahead of the GPU.
//
// One means the CPU waits every frame, which is chapter 1.5. Three or more adds
// latency -- an input is a further frame old by the time it is displayed -- without
// buying throughput on a v-synced target. Two is the usual answer.
constexpr uint32_t kFramesInFlight = 2;

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

// Moves an image from one layout to another.
//
// A layout is how the driver has arranged the pixels in memory. The arrangement that
// is fast to render into is not the one that is fast to display from, so the image
// has to be told to change, and nothing changes it implicitly. This is the single
// biggest source of validation errors for anyone new to Vulkan.
//
// The stage and access masks say "which work must finish before" and "which work must
// wait until after". ALL_COMMANDS on both sides is the sledgehammer: correct, and
// slower than it needs to be. A later chapter narrows them and measures the
// difference.
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                      VkImageLayout to) {
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .oldLayout = from,
        .newLayout = to,
        // Not transferring ownership between queue families, so both are IGNORED.
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
    };

    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(cmd, &dependency);
}

class TriangleApp {
public:
    explicit TriangleApp(const vkc::Args& args) : args_(args) {
        init_window();
        init_instance();
        create_surface();
        select_physical_device();
        create_device();
        create_allocator();
        create_swapchain();
        create_frames();
        create_image_semaphores();
        create_pipeline();
        capture_.init(device_, allocator_);
    }

    ~TriangleApp() {
        // Strict reverse order. Vulkan will not warn you at run time if you get this
        // wrong, but the validation layers will, loudly.
        // Wait for the GPU before tearing anything down: it may still be reading
        // the command buffer we are about to free.
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
        }
        capture_.destroy();
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        destroy_image_semaphores();
        for (PerFrame& frame : frames_) {
            vkDestroyFence(device_, frame.in_flight, nullptr);
            vkDestroySemaphore(device_, frame.image_available, nullptr);
            // Destroying a pool frees every command buffer allocated from it.
            vkDestroyCommandPool(device_, frame.pool, nullptr);
        }
        frames_.clear();
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

    TriangleApp(const TriangleApp&) = delete;
    TriangleApp& operator=(const TriangleApp&) = delete;
    TriangleApp(TriangleApp&&) = delete;
    TriangleApp& operator=(TriangleApp&&) = delete;

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
            // Minimised windows have a zero-sized drawable, so there is nothing
            // to render into and no swapchain to build.
            if ((SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0) {
                SDL_WaitEvent(nullptr);
                continue;
            }

            const bool last_frame =
                args_.frame_limit != 0 && frames + 1 >= args_.frame_limit;
            draw_frame(last_frame && !args_.screenshot.empty());

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
        window_ = SDL_CreateWindow("LearnVulkan - Hello Triangle",
                                   static_cast<int>(args_.width),
                                   static_cast<int>(args_.height),
                                   SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
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
        // The image count can change when the swapchain is rebuilt, and there is one
        // render-finished semaphore per image.
        destroy_image_semaphores();
        create_image_semaphores();
        spdlog::info("Swapchain rebuilt at {}x{}", extent_.width, extent_.height);
    }

    void create_frames() {
        frames_.resize(kFramesInFlight);

        for (PerFrame& frame : frames_) {
            const VkCommandPoolCreateInfo pool_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = queue_family_,
            };
            VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &frame.pool));

            const VkCommandBufferAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = frame.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            VK_CHECK(vkAllocateCommandBuffers(device_, &alloc_info, &frame.cmd));

            const VkSemaphoreCreateInfo semaphore_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            };
            VK_CHECK(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                                       &frame.image_available));

            const VkFenceCreateInfo fence_info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                // Created already signalled. The first frame waits on this fence
                // before any work has been submitted, and an unsignalled fence would
                // mean waiting forever for something that never happened.
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };
            VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &frame.in_flight));
        }
    }

    // One render-finished semaphore per swapchain *image*, not per frame in flight.
    //
    // This is a genuinely easy mistake to make. The semaphore a present operation
    // waits on must stay untouched until that present has actually happened, and
    // presents complete in swapchain-image order, not in frame-slot order. With two
    // frame slots and three images, reusing a frame's semaphore lets you signal one
    // that a pending present is still waiting on. The validation layers catch it, but
    // only sometimes, and only under load.
    void create_image_semaphores() {
        render_finished_.resize(images_.size());
        for (VkSemaphore& semaphore : render_finished_) {
            const VkSemaphoreCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            };
            VK_CHECK(vkCreateSemaphore(device_, &info, nullptr, &semaphore));
        }
    }

    void destroy_image_semaphores() noexcept {
        for (VkSemaphore semaphore : render_finished_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        render_finished_.clear();
    }

    // Slang reports what went wrong through a blob rather than a return code, so the
    // calls below all pass one in and this turns it into something printable.
    [[nodiscard]] static std::string blob_text(slang::IBlob* blob) {
        if (blob == nullptr || blob->getBufferSize() == 0) {
            return {};
        }
        return std::string(static_cast<const char*>(blob->getBufferPointer()),
                           blob->getBufferSize());
    }

    // Compiles a .slang file to SPIR-V and hands the result to the driver.
    //
    // Vulkan will not take shader source -- it takes SPIR-V -- so something has to do
    // the translation. This project links libslang and does it here, when the program
    // starts, which means the .slang file beside the binary is the shader: edit it, run
    // again, see the change, with no build step in between.
    //
    // Everything from createGlobalSession down to getTargetCode is Slang's compiler
    // API. The last six lines are the Vulkan part, and they are the same six lines they
    // would be if the bytes had come from a .spv file on disk.
    [[nodiscard]] VkShaderModule load_shader(const char* name) const {
        // The global session owns the compiler and its standard library. It is the
        // expensive object -- create it once and share it. One shader here, so one
        // session; vkcommon caches it from 1.8 onward.
        Slang::ComPtr<slang::IGlobalSession> global;
        if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
            throw std::runtime_error("slang::createGlobalSession failed");
        }

        // What to generate: SPIR-V 1.6, which is the version Vulkan 1.3 consumes.
        slang::TargetDesc target{};
        target.format = SLANG_SPIRV;
        target.profile = global->findProfile("spirv_1_6");

        const slang::CompilerOptionEntry options[]{
            // Slang's own SPIR-V backend, rather than routing through generated GLSL.
            {slang::CompilerOptionName::EmitSpirvDirectly,
             {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
            // Keep the entry points named after the Slang functions. Without this, a
            // module gets its entry point renamed to "main" and the names stop matching
            // what VkPipelineShaderStageCreateInfo::pName asks for below.
            {slang::CompilerOptionName::VulkanUseEntryPointName,
             {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
            // Debug info, so RenderDoc can show the Slang source beside the SPIR-V.
            {slang::CompilerOptionName::DebugInformation,
             {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD, 0,
              nullptr, nullptr}},
        };

        // Where to look for the file. The build staged it next to the binary.
        const std::string search_path = vkc::shader_dir(LVK_CHAPTER_ID).string();
        const char* search_paths[]{search_path.c_str()};

        slang::SessionDesc session_desc{};
        session_desc.targets = &target;
        session_desc.targetCount = 1;
        session_desc.searchPaths = search_paths;
        session_desc.searchPathCount = 1;
        session_desc.compilerOptionEntries = options;
        session_desc.compilerOptionEntryCount = 3;

        // Matrix layout, and this is the one that will bite you.
        //
        // The session default is SLANG_MATRIX_LAYOUT_ROW_MAJOR, which does not match
        // glm -- a glm::mat4 is sixteen floats stored column by column, and it is
        // memcpy'd into a uniform buffer or a push constant with no transpose on the
        // way. Leave the default and every transform comes out transposed: the image
        // still draws, so nothing errors, it is just wrong.
        //
        // Setting COLUMN_MAJOR makes Slang emit a RowMajor decoration in the SPIR-V,
        // which looks like the opposite of what was asked for and is not. Slang pairs
        // that decoration with OpVectorTimesMatrix, and storing transposed while
        // multiplying on the other side is the same arithmetic as storing plainly and
        // multiplying normally. The two conventions cancel. What matters is that this
        // line makes mul(M, v) mean "apply M to v" for a matrix whose bytes came from
        // glm.
        //
        // slangc's command line defaults to this; the API does not.
        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

        Slang::ComPtr<slang::ISession> session;
        if (SLANG_FAILED(global->createSession(session_desc, session.writeRef()))) {
            throw std::runtime_error("slang::IGlobalSession::createSession failed");
        }

        // Slang names a file by module, without the extension: triangle.slang on disk
        // is the module "triangle", found by walking the search paths.
        std::string module_name(name);
        if (module_name.ends_with(".slang")) {
            module_name.resize(module_name.size() - 6);
        }

        Slang::ComPtr<slang::IBlob> diagnostics;
        slang::IModule* module =
            session->loadModule(module_name.c_str(), diagnostics.writeRef());
        if (module == nullptr) {
            throw std::runtime_error(
                std::format("Could not compile '{}' from '{}'.\n{}", module_name,
                            search_path, blob_text(diagnostics)));
        }

        // Each [shader("...")] function in the file is an entry point. Composing them
        // with the module and linking is what produces one SPIR-V binary holding the
        // whole pipeline -- both stages, one module.
        std::vector<slang::IComponentType*> components{module};
        const SlangInt entry_point_count = module->getDefinedEntryPointCount();
        std::vector<Slang::ComPtr<slang::IEntryPoint>> entry_points(
            static_cast<size_t>(entry_point_count));
        for (SlangInt i = 0; i < entry_point_count; ++i) {
            const auto index = static_cast<size_t>(i);
            if (SLANG_FAILED(
                    module->getDefinedEntryPoint(i, entry_points[index].writeRef()))) {
                throw std::runtime_error("slang::IModule::getDefinedEntryPoint failed");
            }
            components.push_back(entry_points[index]);
        }

        Slang::ComPtr<slang::IComponentType> composed;
        diagnostics = nullptr;
        if (SLANG_FAILED(session->createCompositeComponentType(
                components.data(), static_cast<SlangInt>(components.size()),
                composed.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Composing '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        Slang::ComPtr<slang::IComponentType> linked;
        diagnostics = nullptr;
        if (SLANG_FAILED(composed->link(linked.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Linking '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        Slang::ComPtr<slang::IBlob> spirv;
        diagnostics = nullptr;
        if (SLANG_FAILED(
                linked->getTargetCode(0, spirv.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Generating SPIR-V for '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        // From here it is ordinary Vulkan. A shader module is just a SPIR-V blob handed
        // to the driver; it is not compiled to machine code here -- that happens when
        // the pipeline is created, which is when the driver finally knows the rest of
        // the state the shader runs under.
        //
        // codeSize is in bytes while pCode is a uint32_t*, because SPIR-V is a stream of
        // 32-bit words and must be 4-byte aligned. Slang's blob already is.
        const VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = spirv->getBufferSize(),
            .pCode = static_cast<const uint32_t*>(spirv->getBufferPointer()),
        };
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device_, &info, nullptr, &shader_module));
        return shader_module;
    }

    void create_pipeline() {
        // One module for both stages. triangle.slang declares a vertexMain and a
        // fragmentMain, and the compile above produced a single SPIR-V binary holding
        // both of them.
        const VkShaderModule shader = load_shader("triangle.slang");

        const VkPipelineShaderStageCreateInfo stages[2]{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = shader,
                // The entry point, by name. A SPIR-V module can hold as many as you
                // like, and pName is what selects one -- which is why the same module
                // appears twice here with a different name each time.
                //
                // Nothing requires the name "main". Slang keeps the function's own name
                // because the build passes -fvk-use-entrypoint-name; GLSL has no choice
                // in the matter, since glslang always emits an entry point called main.
                .pName = "vertexMain",
                .pSpecializationInfo = nullptr,
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = shader,
                .pName = "fragmentMain",
                .pSpecializationInfo = nullptr,
            },
        };

        // No vertex buffers: the vertex shader generates its own positions from
        // the vertex index. Chapter 1.8 fills this struct in properly.
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
        };

        // Every three vertices form one triangle.
        const VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        // The counts matter even though the pointers are null: the pipeline needs to
        // know there is one of each, and the values arrive per-frame via
        // vkCmdSetViewport and vkCmdSetScissor.
        const VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr,
        };

        const VkPipelineRasterizationStateCreateInfo rasterisation{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            // VK_TRUE here would throw away all geometry before rasterisation, which
            // is occasionally useful and always confusing to leave on by accident.
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            // No culling yet. Face culling gets its own chapter, and turning it on now
            // would silently discard the triangle if its winding were the other way.
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0F,
            .depthBiasClamp = 0.0F,
            .depthBiasSlopeFactor = 0.0F,
            .lineWidth = 1.0F,
        };

        // One sample per pixel: no multisampling. The anti-aliasing chapter revisits
        // this, and it is one of the fields that must match the attachments exactly.
        const VkPipelineMultisampleStateCreateInfo multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0F,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };

        // blendEnable VK_FALSE means the fragment shader's output replaces whatever
        // is in the attachment. colorWriteMask still has to list the channels to
        // write; leaving it at zero writes nothing at all, for a black screen and no
        // error message.
        const VkPipelineColorBlendAttachmentState blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        const VkPipelineColorBlendStateCreateInfo blend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &blend_attachment,
            .blendConstants = {0.0F, 0.0F, 0.0F, 0.0F},
        };

        const VkDynamicState dynamic_states[2]{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        const VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamic_states,
        };

        // The layout declares what external data the shaders can see: descriptor sets
        // and push constants. This shader reads nothing, so the layout is empty -- but
        // it still has to exist.
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };
        VK_CHECK(vkCreatePipelineLayout(device_, &layout_info, nullptr,
                                        &pipeline_layout_));

        // With dynamic rendering there is no VkRenderPass to be compatible with, so
        // the pipeline is told the attachment formats directly. These must match the
        // VkRenderingInfo used at draw time, or nothing will be drawn.
        const VkPipelineRenderingCreateInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &format_,
            .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };

        const VkGraphicsPipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering_info,
            .flags = 0,
            .stageCount = 2,
            .pStages = stages,
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterisation,
            .pMultisampleState = &multisample,
            .pDepthStencilState = nullptr,  // no depth attachment in this chapter
            .pColorBlendState = &blend,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout_,
            // Null, because dynamic rendering replaced them.
            .renderPass = VK_NULL_HANDLE,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                           nullptr, &pipeline_));

        // The module has been compiled into the pipeline and is no longer needed.
        vkDestroyShaderModule(device_, shader, nullptr);
    }

    void draw_frame(bool capture_this_frame) {
        PerFrame& frame = frames_[frame_number_ % kFramesInFlight];

        // Wait until the GPU has finished the *previous* frame that used this slot --
        // not the frame immediately before this one. With two slots in flight, that
        // is two frames ago, which is exactly why this usually does not block.
        VK_CHECK(vkWaitForFences(device_, 1, &frame.in_flight, VK_TRUE, UINT64_MAX));

        // Acquire signals a semaphore this time, not a fence. The CPU does not need
        // to know when the image is ready; the GPU does, and it will wait on this
        // semaphore before writing any colour.
        uint32_t image_index = 0;
        const VkResult acquired =
            vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                  frame.image_available, VK_NULL_HANDLE, &image_index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            // Nothing was acquired and the semaphore was not signalled, so we can
            // simply rebuild and skip this frame.
            recreate_swapchain();
            return;
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
            vkc::throw_vulkan_error(acquired, "vkAcquireNextImageKHR", __FILE__,
                                    __LINE__);
        }

        // Reset only now that a submit is certain. Resetting before the early return
        // above would leave the fence unsignalled with no work to signal it, and the
        // next frame in this slot would hang forever.
        VK_CHECK(vkResetFences(device_, 1, &frame.in_flight));
        VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        VK_CHECK(vkBeginCommandBuffer(frame.cmd, &begin_info));

        transition_image(frame.cmd, images_[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // A flat dark background now, rather than the cycling colour of 1.5 and 1.6.
        // The triangle is the subject of this chapter and a moving background would
        // only fight with it.
        const VkClearValue clear{.color = {{0.02F, 0.02F, 0.04F, 1.0F}}};

        const VkRenderingAttachmentInfo colour_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = views_[image_index],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clear,
        };

        const VkRenderingInfo rendering{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {{0, 0}, extent_},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour_attachment,
            .pDepthAttachment = nullptr,
            .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(frame.cmd, &rendering);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        // The two pieces of state left dynamic. The viewport maps clip space onto the
        // image; the scissor throws away anything outside a rectangle.
        //
        // Note minDepth/maxDepth of 0..1: that is Vulkan's depth range, and it is why
        // the project defines GLM_FORCE_DEPTH_ZERO_TO_ONE. OpenGL's is -1..1.
        const VkViewport viewport{
            .x = 0.0F,
            .y = 0.0F,
            .width = static_cast<float>(extent_.width),
            .height = static_cast<float>(extent_.height),
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        const VkRect2D scissor{.offset = {0, 0}, .extent = extent_};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        // Three vertices, one instance. The vertex shader runs three times, with
        // SV_VulkanVertexID counting 0, 1, 2.
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);

        if (capture_this_frame) {
            capture_.record(frame.cmd, images_[image_index], extent_);
        } else {
            transition_image(frame.cmd, images_[image_index],
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }

        VK_CHECK(vkEndCommandBuffer(frame.cmd));

        // Wait on the acquire semaphore, but only at COLOR_ATTACHMENT_OUTPUT -- the
        // stage that actually writes colour. Vertex shading, and everything else
        // earlier in the pipeline, can start before the image is ready. Waiting at
        // ALL_COMMANDS instead would work and would needlessly serialise the frame.
        const VkSemaphoreSubmitInfo wait{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = frame.image_available,
            .value = 0,  // ignored by binary semaphores
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .deviceIndex = 0,
        };

        const VkSemaphoreSubmitInfo signal{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = render_finished_[image_index],
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };

        const VkCommandBufferSubmitInfo cmd_submit{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = frame.cmd,
            .deviceMask = 0,
        };

        const VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_submit,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal,
        };

        // The fence is signalled when this submission completes, which is what the
        // top of this function waits on two frames from now.
        VK_CHECK(vkQueueSubmit2(queue_, 1, &submit, frame.in_flight));

        // Present waits on the GPU side for rendering to finish. The CPU does not
        // block here at all -- it goes straight back to the event loop and starts
        // recording the next frame.
        const VkPresentInfoKHR present{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_[image_index],
            .swapchainCount = 1,
            .pSwapchains = &swapchain_,
            .pImageIndices = &image_index,
            .pResults = nullptr,
        };
        const VkResult presented = vkQueuePresentKHR(queue_, &present);

        ++frame_number_;

        // A screenshot is the one place the CPU genuinely must wait, because it has
        // to read back what the GPU wrote. It happens once, on the last frame.
        if (capture_this_frame) {
            VK_CHECK(vkDeviceWaitIdle(device_));
            capture_.write(args_.screenshot, format_);
        }

        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
            recreate_swapchain();
        } else if (presented != VK_SUCCESS) {
            vkc::throw_vulkan_error(presented, "vkQueuePresentKHR", __FILE__, __LINE__);
        }
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
        spdlog::info("Drawing a triangle. Escape or close to quit.");
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

    struct PerFrame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore image_available = VK_NULL_HANDLE;
        VkFence in_flight = VK_NULL_HANDLE;
    };
    std::vector<PerFrame> frames_;
    std::vector<VkSemaphore> render_finished_;  // one per swapchain image
    uint64_t frame_number_ = 0;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    vkc::Capture capture_;

    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        TriangleApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
