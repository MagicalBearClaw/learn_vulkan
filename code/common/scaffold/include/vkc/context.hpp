#pragma once

#include "vkc/device.hpp"
#include "vkc/instance.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace vkc {

class Window;

// Everything that is created once at start-up and lives for the whole run.
//
// This is a facade over two objects that are taught separately and can be used
// separately: Instance is chapter 1.2's subject, Device is chapter 1.3's, and the
// surface that joins them comes from the Window. A chapter that has read both can use
// Instance and Device directly; everything from 1.4 onward takes the whole Context,
// because from there on it always wants all three.
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

    // The two halves, for a chapter that wants to name them.
    [[nodiscard]] const Instance& vk_instance() const noexcept { return *instance_; }
    [[nodiscard]] const Device& vk_device() const noexcept { return *device_; }

    [[nodiscard]] VkInstance instance() const noexcept { return instance_->handle(); }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept {
        return device_->physical_device();
    }
    [[nodiscard]] VkDevice device() const noexcept { return device_->handle(); }
    [[nodiscard]] VkSurfaceKHR surface() const noexcept { return surface_; }
    [[nodiscard]] VmaAllocator allocator() const noexcept {
        return device_->allocator();
    }

    // A single queue that supports graphics, compute, transfer and presentation.
    // Every desktop GPU exposes at least one such queue family, and using one queue
    // for everything removes an entire class of ownership-transfer barriers from the
    // early chapters. Multiple queues get their own chapter much later.
    [[nodiscard]] VkQueue queue() const noexcept { return device_->queue(); }
    [[nodiscard]] uint32_t queue_family() const noexcept {
        return device_->queue_family();
    }

    [[nodiscard]] const VkPhysicalDeviceProperties& gpu_properties() const noexcept {
        return device_->gpu_properties();
    }

    // Attaches a readable name to a Vulkan object. Costs nothing in release builds
    // and turns RenderDoc captures and validation messages from handle soup into
    // something you can actually read.
    void set_debug_name(uint64_t handle, VkObjectType type, const char* name) const {
        device_->set_debug_name(handle, type, name);
    }

    template <typename Handle>
    void name(Handle handle, VkObjectType type, const char* label) const {
        set_debug_name(reinterpret_cast<uint64_t>(handle), type, label);
    }

    // Blocks until the GPU is completely idle. Only ever correct at teardown or
    // during a window resize; never in a frame loop.
    void wait_idle() const { device_->wait_idle(); }

private:
    std::unique_ptr<Instance> instance_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::unique_ptr<Device> device_;

    Window* window_ = nullptr;
};

}  // namespace vkc
