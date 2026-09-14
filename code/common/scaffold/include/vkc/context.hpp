#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <string_view>

namespace vkc {

class Window;

// Everything that is created once at start-up and lives for the whole run: the
// instance, the debug messenger, the chosen GPU, the logical device, the queue we
// submit to, and the memory allocator.
//
// This series targets Vulkan 1.3 with dynamic rendering and synchronization2 always
// enabled, so there are no VkRenderPass or VkFramebuffer objects anywhere in it, and
// every barrier uses the VkDependencyInfo form.
class Context {
public:
    struct Config {
        std::string_view app_name = "LearnVulkan";
        bool enable_validation = true;
    };

    Context(const Config& config, Window& window);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return gpu_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkSurfaceKHR surface() const noexcept { return surface_; }
    [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }

    // A single queue that supports graphics, compute, transfer and presentation.
    // Every desktop GPU exposes at least one such queue family, and using one queue
    // for everything removes an entire class of ownership-transfer barriers from the
    // early chapters. Multiple queues get their own chapter much later.
    [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
    [[nodiscard]] uint32_t queue_family() const noexcept { return queue_family_; }

    [[nodiscard]] const VkPhysicalDeviceProperties& gpu_properties() const noexcept {
        return gpu_properties_;
    }

    // Attaches a readable name to a Vulkan object. Costs nothing in release builds
    // and turns RenderDoc captures and validation messages from handle soup into
    // something you can actually read.
    void set_debug_name(uint64_t handle, VkObjectType type, const char* name) const;

    template <typename Handle>
    void name(Handle handle, VkObjectType type, const char* label) const {
        set_debug_name(reinterpret_cast<uint64_t>(handle), type, label);
    }

    // Blocks until the GPU is completely idle. Only ever correct at teardown or
    // during a window resize; never in a frame loop.
    void wait_idle() const;

private:
    void create_instance(const Config& config);
    void create_debug_messenger();
    void select_physical_device();
    void create_device();
    void create_allocator();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties gpu_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    Window* window_ = nullptr;
    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
};

}  // namespace vkc
