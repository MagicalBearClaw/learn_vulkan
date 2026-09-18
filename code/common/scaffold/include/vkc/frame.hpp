#pragma once

#include <volk.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace vkc {

class Context;
class Swapchain;

// Number of frames the CPU is allowed to run ahead of the GPU.
//
// One would mean the CPU waits for the GPU every single frame and half the machine
// idles. Three or more adds latency for no throughput gain on a v-synced target.
// Two is the usual answer, and the reason every sample needs per-frame copies of
// its command buffer and its synchronisation objects rather than one of each.
inline constexpr uint32_t kFramesInFlight = 2;

// Records the standard image-layout transition as a Vulkan 1.3 dependency.
//
// Layout transitions are the price Vulkan charges for not guessing: an image that
// the GPU is about to render into wants a different memory layout from one it is
// about to present, and nothing moves it between the two unless you say so.
// The stage/access masks here are deliberately broad. A chapter later in the series
// tightens them and measures the difference.
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                      VkImageLayout to);

// The same transition, said precisely: which stage produced the data and which access
// wrote it, which stage is about to consume it and which access will read it.
//
// transition_image() above names ALL_COMMANDS on both sides, which is correct and
// forbids overlap that was never a problem. That is a fine trade at the top and bottom
// of a frame, where there is nothing to overlap with. It is the wrong trade in the
// middle of one -- between two passes that share an image -- because there the masks
// decide how much of the two passes may run at once. Chapter 4.3 writes this function
// out and works through both of its calls mask by mask.
void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                   VkImageLayout to, VkPipelineStageFlags2 src_stage,
                   VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                   VkAccessFlags2 dst_access,
                   VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

// Opens dynamic rendering into a colour view and a depth view, clearing both.
//
// Chapter 1.5 introduced VkRenderingInfo and 1.13 added the depth attachment. Both
// attachments clear on load, so neither needs its previous contents; the depth buffer's
// storeOp is DONT_CARE because nothing reads it once the frame is over.
void begin_rendering(VkCommandBuffer cmd, VkImageView colour_view, VkImageView depth_view,
                     VkExtent2D extent, const VkClearColorValue& clear_colour);

// What a sample gets handed for the duration of one frame.
struct FrameInfo {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t image_index = 0;
    VkImage image = VK_NULL_HANDLE;      // the swapchain image being drawn into
    VkImageView view = VK_NULL_HANDLE;   // its view, for dynamic rendering
    VkExtent2D extent{};
    uint64_t frame_number = 0;           // monotonic, never resets
};

// Owns the per-frame command buffers and synchronisation primitives, and drives the
// acquire -> record -> submit -> present cycle.
class FrameContext {
public:
    FrameContext(Context& context, Swapchain& swapchain);
    ~FrameContext();

    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
    FrameContext(FrameContext&&) = delete;
    FrameContext& operator=(FrameContext&&) = delete;

    // Waits for this frame slot to be free, acquires a swapchain image, and opens a
    // command buffer. Returns nullopt when the swapchain went out of date, in which
    // case the caller should rebuild it and skip the frame rather than draw garbage.
    [[nodiscard]] std::optional<FrameInfo> begin();

    // Closes the command buffer, submits it, and presents. Returns false if the
    // swapchain needs rebuilding before the next frame.
    [[nodiscard]] bool end(const FrameInfo& frame);

    // Called after the swapchain has been recreated: the per-image semaphores are
    // sized to the image count and have to follow it.
    void on_swapchain_recreated();

private:
    void create_per_frame_objects();
    void create_per_image_objects();
    void destroy_per_image_objects() noexcept;

    struct PerFrame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        // Signalled by the presentation engine when the acquired image is actually
        // ready to be written to.
        VkSemaphore image_available = VK_NULL_HANDLE;
        // Signalled by the GPU when this frame's work is done, so the CPU knows the
        // command buffer can be reset and reused.
        VkFence in_flight = VK_NULL_HANDLE;
    };

    Context& context_;
    Swapchain& swapchain_;

    std::vector<PerFrame> frames_;
    // One per swapchain image, not per frame in flight: the semaphore a present
    // operation waits on must not be reused while that image is still queued for
    // display, and the image count and the in-flight count are different numbers.
    std::vector<VkSemaphore> render_finished_;

    uint64_t frame_number_ = 0;
};

}  // namespace vkc
