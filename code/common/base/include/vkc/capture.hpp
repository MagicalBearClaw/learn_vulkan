#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace vkc {

// Options every sample in the book understands, so that tools/capture.py can drive
// any chapter the same way.
//
// This is book infrastructure rather than a Vulkan lesson: it exists so the article
// screenshots stay in step with the code that produces them. No chapter depends on it
// to teach anything, and you can ignore it entirely when reading a sample.
struct Args {
    uint32_t width = 1280;
    uint32_t height = 720;
    bool validation = true;
    uint64_t frame_limit = 0;  // 0 means "run until the window is closed"
    std::filesystem::path screenshot;
};

// Parses --width, --height, --frames, --screenshot, --no-validation and --help.
[[nodiscard]] Args parse_args(int argc, char** argv, Args defaults = {});

// Copies a rendered swapchain image back to the CPU and writes it out as a PNG.
//
// Also book infrastructure. The one part worth noticing if you are reading a sample:
// the copy has to be recorded into the same command buffer that drew the frame,
// because a swapchain image may only be touched between vkAcquireNextImageKHR and
// vkQueuePresentKHR.
class Capture {
public:
    Capture() = default;
    ~Capture();

    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
    Capture(Capture&&) = delete;
    Capture& operator=(Capture&&) = delete;

    void init(VkDevice device, VmaAllocator allocator);

    // Records: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC -> copy -> PRESENT_SRC_KHR.
    // The image still gets presented; the screenshot is taken on the way past.
    void record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent);

    // Writes what record() copied. Call only once the GPU has finished the frame.
    void write(const std::filesystem::path& path, VkFormat format) const;

    [[nodiscard]] bool has_pending_capture() const noexcept { return pending_; }

    void destroy() noexcept;

private:
    void allocate(VkExtent2D extent);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocation_info_{};
    VkExtent2D extent_{};
    bool pending_ = false;
};

}  // namespace vkc
