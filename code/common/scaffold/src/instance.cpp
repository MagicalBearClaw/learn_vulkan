#include "vkc/instance.hpp"

#include "vkc/check.hpp"
#include "vkc/window.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <vector>

namespace vkc {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

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

}  // namespace

Instance::Instance(const Config& config) {
    // volk loads vkGetInstanceProcAddr and the handful of entry points that exist
    // before an instance does. Nothing Vulkan works before this call.
    VK_CHECK(volkInitialize());

    create_instance(config);
    volkLoadInstanceOnly(instance_);

    if (validation_enabled_ && debug_utils_enabled_) {
        create_debug_messenger();
    }
}

Instance::~Instance() {
    if (debug_messenger_ != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void Instance::create_instance(const Config& config) {
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

void Instance::create_debug_messenger() {
    const VkDebugUtilsMessengerCreateInfoEXT info = debug_messenger_info();
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &info, nullptr, &debug_messenger_));
}

}  // namespace vkc
