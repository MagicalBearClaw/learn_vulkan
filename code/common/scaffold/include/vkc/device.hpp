#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>

namespace vkc {

class Instance;

// The GPU this program talks to: the physical device it chose, the logical device it
// opened onto it, the one queue it submits to, and the memory allocator that serves
// it. Chapter 1.3 writes every line of this by hand and explains it; from 1.4 onward
// it lives here.
//
// A surface has to exist before this can be built, because "can this GPU present to
// that window?" is one of the questions device selection asks.
class Device {
public:
    Device(const Instance& instance, VkSurfaceKHR surface);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return gpu_; }
    [[nodiscard]] VkDevice handle() const noexcept { return device_; }
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

    // Attaches a readable name to a Vulkan object. Does nothing when
    // VK_EXT_debug_utils is unavailable, and costs nothing in release builds.
    void set_debug_name(uint64_t handle, VkObjectType type, const char* name) const;

    // Blocks until the GPU is completely idle. Only ever correct at teardown or
    // during a window resize; never in a frame loop.
    void wait_idle() const;

private:
    void select_physical_device(VkInstance instance, VkSurfaceKHR surface);
    void create_device();
    void create_allocator(VkInstance instance);

    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties gpu_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    bool debug_utils_enabled_ = false;
};

}  // namespace vkc
