#include "vkc/context.hpp"

#include "vkc/check.hpp"
#include "vkc/window.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkc {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// Extensions every sample in the series needs on the device. Vulkan 1.3 folded
// dynamic rendering and synchronization2 into core, so the only one left is the
// swapchain itself.
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
    // Returning VK_FALSE means "carry on"; VK_TRUE would abort the offending call
    // and is only meaningful to layer developers.
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

// Index of a queue family that can do graphics, compute and presentation at once,
// or nullopt if this GPU has none.
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
        VkBool32 can_present = VK_FALSE;
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &can_present));
        if (can_present == VK_TRUE) {
            return i;
        }
    }
    return std::nullopt;
}

// Discrete GPUs first, then integrated, then whatever is left. Crude, and exactly
// good enough: a tutorial that picks the wrong GPU is a tutorial that runs slowly,
// not one that runs wrongly.
int score_device(const VkPhysicalDeviceProperties& props) {
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
        default: return 0;
    }
}

}  // namespace

Context::Context(const Config& config, Window& window) : window_(&window) {
    // volk loads vkGetInstanceProcAddr and the handful of entry points that exist
    // before an instance does. Nothing Vulkan works before this call.
    VK_CHECK(volkInitialize());

    create_instance(config);
    volkLoadInstanceOnly(instance_);

    if (validation_enabled_ && debug_utils_enabled_) {
        create_debug_messenger();
    }

    surface_ = window.create_surface(instance_);

    select_physical_device();
    create_device();
    create_allocator();

    spdlog::info("GPU: {} (Vulkan {}.{}.{})", gpu_properties_.deviceName,
                 VK_API_VERSION_MAJOR(gpu_properties_.apiVersion),
                 VK_API_VERSION_MINOR(gpu_properties_.apiVersion),
                 VK_API_VERSION_PATCH(gpu_properties_.apiVersion));
}

Context::~Context() {
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (window_ != nullptr) {
        window_->destroy_surface(instance_);
    }
    if (debug_messenger_ != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void Context::create_instance(const Config& config) {
    const std::string app_name(config.app_name);

    const VkApplicationInfo app_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = app_name.c_str(),
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "vkcommon",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        // Vulkan 1.3 is the floor for this series: dynamic rendering and
        // synchronization2 are core there, and that is what removes most of the
        // boilerplate older tutorials have to spend five chapters on.
        .apiVersion = VK_API_VERSION_1_3,
    };

    std::vector<const char*> extensions;
    for (const char* ext : Window::required_instance_extensions()) {
        extensions.push_back(ext);
    }

    validation_enabled_ = config.enable_validation && layer_available(kValidationLayer);
    if (config.enable_validation && !validation_enabled_) {
        spdlog::warn(
            "Validation layers were requested but {} is not installed. Install the "
            "Vulkan SDK or your distribution's validation-layers package: running "
            "these samples without validation hides real bugs.",
            kValidationLayer);
    }

    debug_utils_enabled_ = instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debug_utils_enabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> layers;
    if (validation_enabled_) {
        layers.push_back(kValidationLayer);
    }

    // Chaining the messenger info onto the create info catches errors raised by
    // vkCreateInstance itself, before a real messenger object can exist.
    const VkDebugUtilsMessengerCreateInfoEXT messenger_info = debug_messenger_info();

    const VkInstanceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = (validation_enabled_ && debug_utils_enabled_) ? &messenger_info : nullptr,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));
}

void Context::create_debug_messenger() {
    const VkDebugUtilsMessengerCreateInfoEXT info = debug_messenger_info();
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &info, nullptr, &debug_messenger_));
}

void Context::select_physical_device() {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    if (count == 0) {
        throw std::runtime_error(
            "No Vulkan-capable GPU found. Check that your graphics driver is "
            "installed and current.");
    }
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

    int best_score = -1;
    std::string rejection_log;

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);

        if (props.apiVersion < VK_API_VERSION_1_3) {
            rejection_log += std::format("\n  {}: reports Vulkan {}.{}, needs 1.3",
                                         props.deviceName,
                                         VK_API_VERSION_MAJOR(props.apiVersion),
                                         VK_API_VERSION_MINOR(props.apiVersion));
            continue;
        }
        if (!device_supports_extensions(candidate)) {
            rejection_log +=
                std::format("\n  {}: missing VK_KHR_swapchain", props.deviceName);
            continue;
        }
        const std::optional<uint32_t> family =
            find_universal_queue_family(candidate, surface_);
        if (!family.has_value()) {
            rejection_log += std::format(
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
            "No suitable GPU found. Every device was rejected:" + rejection_log);
    }
}

void Context::create_device() {
    const float priority = 1.0F;
    const VkDeviceQueueCreateInfo queue_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = queue_family_,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    // Features are opt-in. These three are what the series is built on, and asking
    // for them here is what makes vkCmdBeginRendering and vkCmdPipelineBarrier2
    // legal to call at all.
    // These structs have dozens of fields each. Value-initialise to all-false and
    // then switch on only what is wanted: asking for a feature you do not use can
    // cost performance, and on some drivers it can cost device creation entirely.
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    // `discard` in Slang compiles to OpDemoteToHelperInvocation rather than the older
    // OpKill, and that SPIR-V capability has to be asked for by name. Without it
    // vkCreateShaderModule rejects the module -- which is how chapter 4.2 found out it
    // was needed, since nothing in the series discards a fragment before then.
    features13.shaderDemoteToHelperInvocation = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    // descriptorIndexing is a headline, not a switch: it reports that the implementation
    // supports the descriptor-indexing family, and enables none of the individual
    // capabilities. Each of those is its own boolean, and a shader that declares one
    // without it enabled is rejected by vkCreateShaderModule. Chapter 3.4 needs exactly
    // this one, for its unbounded `Sampler2D base_colour_maps[]`.
    features12.runtimeDescriptorArray = VK_TRUE;

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
        // Device layers have not been a thing since Vulkan 1.0.13; the instance
        // layer covers the device too.
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(kRequiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = kRequiredDeviceExtensions.data(),
        // pEnabledFeatures must stay null when VkPhysicalDeviceFeatures2 is chained.
        .pEnabledFeatures = nullptr,
    };

    VK_CHECK(vkCreateDevice(gpu_, &create_info, nullptr, &device_));

    // Re-load every entry point against the device now that one exists. Until this
    // runs, every device-level function pointer -- vkGetDeviceQueue included -- is
    // still null, so this has to happen before the very next call and not later.
    // It also skips the loader's dispatch-table indirection for the rest of the
    // program's life.
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

void Context::create_allocator() {
    // VMA needs to be told where the entry points live, because volk has already
    // taken them out of the global namespace.
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

void Context::set_debug_name(uint64_t handle, VkObjectType type,
                             const char* name) const {
    if (!debug_utils_enabled_ || handle == 0) {
        return;
    }
    const VkDebugUtilsObjectNameInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .pNext = nullptr,
        .objectType = type,
        .objectHandle = handle,
        .pObjectName = name,
    };
    VK_CHECK(vkSetDebugUtilsObjectNameEXT(device_, &info));
}

void Context::wait_idle() const { VK_CHECK(vkDeviceWaitIdle(device_)); }

}  // namespace vkc
