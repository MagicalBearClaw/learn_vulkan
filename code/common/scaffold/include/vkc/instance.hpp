#pragma once

#include <volk.h>

#include <string_view>

namespace vkc {

// The Vulkan instance, the validation layers, and the debug messenger they report
// through. Chapter 1.2 writes every line of this by hand and explains it; from 1.3
// onward it lives here.
//
// An instance is a connection to the Vulkan *implementation* on this machine -- the
// loader, the layers, and the drivers it found. It is not a connection to a GPU. That
// is what Device is for, and the split between the two classes is the split between
// chapters 1.2 and 1.3.
class Instance {
public:
    struct Config {
        std::string_view app_name = "LearnVulkan";
        bool enable_validation = true;
    };

    explicit Instance(const Config& config);
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    [[nodiscard]] VkInstance handle() const noexcept { return instance_; }

    // Whether VK_LAYER_KHRONOS_validation was actually enabled. It is requested by
    // default and quietly unavailable on a machine without the layers installed, so
    // this is the honest answer rather than what was asked for.
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_enabled_; }

    // Whether VK_EXT_debug_utils is present. It provides both the messenger and the
    // object-naming that turns handle soup into readable validation messages.
    [[nodiscard]] bool debug_utils_enabled() const noexcept {
        return debug_utils_enabled_;
    }

private:
    void create_instance(const Config& config);
    void create_debug_messenger();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
};

}  // namespace vkc
