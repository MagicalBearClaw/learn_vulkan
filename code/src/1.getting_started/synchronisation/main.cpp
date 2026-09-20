// 1.6 Synchronisation
//
// Adds to 1.5: semaphores, fences used properly, and frames in flight.
//
// Chapter 1.5 ended every frame by waiting for the GPU to go completely idle. It was
// correct and it was slow: the CPU sat still while the GPU worked, then the GPU sat
// still while the CPU worked. Half the machine was always doing nothing.
//
// The fix is to stop asking "has the GPU finished?" and start saying "do this after
// that". Vulkan gives two tools for it, and the difference between them is who is
// waiting:
//
//   VkSemaphore  orders work *on the GPU* against other work on the GPU. The CPU
//                never waits on one and cannot read its state.
//   VkFence      lets the *CPU* find out that the GPU has finished something.
//
// A frame needs both. Semaphores chain acquire -> render -> present without the CPU
// being involved at all. A fence tells the CPU when a frame's command buffer is free
// to be re-recorded.
//
// That last point is why "frames in flight" exists. If there is one command buffer,
// the CPU must wait for the GPU before recording the next frame. With two sets, the
// CPU records frame N+1 while the GPU is still working on frame N, and neither waits
// for the other in the common case.
//
// This chapter and 1.5 together are what becomes vkc::FrameContext, which is why
// neither of them may link it.

#include <vkc/capture.hpp>
#include <vkc/check.hpp>
#include <vkc/context.hpp>
#include <vkc/swapchain.hpp>
#include <vkc/window.hpp>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

// How far the CPU may run ahead of the GPU.
//
// One means the CPU waits every frame, which is chapter 1.5. Three or more adds
// latency -- an input is a further frame old by the time it is displayed -- without
// buying throughput on a v-synced target. Two is the usual answer.
constexpr uint32_t kFramesInFlight = 2;

// Chapter 1.5's transition, unchanged.
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

class FrameLoopApp {
public:
    explicit FrameLoopApp(const vkc::Args& args)
        : args_(args),
          window_("LearnVulkan - Synchronisation", args.width, args.height,
                  /*resizable=*/args.screenshot.empty()),
          context_(vkc::Context::Config{
                       .app_name = "LearnVulkan",
                       .enable_validation = args.validation,
                   },
                   window_),
          swapchain_(context_, window_) {
        create_frames();
        create_image_semaphores();
        capture_.init(context_.device(), context_.allocator());
    }

    ~FrameLoopApp() {
        context_.wait_idle();

        capture_.destroy();
        destroy_image_semaphores();
        for (PerFrame& frame : frames_) {
            vkDestroyFence(context_.device(), frame.in_flight, nullptr);
            vkDestroySemaphore(context_.device(), frame.image_available, nullptr);
            // Destroying a pool frees every command buffer allocated from it.
            vkDestroyCommandPool(context_.device(), frame.pool, nullptr);
        }
        frames_.clear();
    }

    FrameLoopApp(const FrameLoopApp&) = delete;
    FrameLoopApp& operator=(const FrameLoopApp&) = delete;
    FrameLoopApp(FrameLoopApp&&) = delete;
    FrameLoopApp& operator=(FrameLoopApp&&) = delete;

    int run() {
        spdlog::info("{} frames in flight. Escape or close to quit.", kFramesInFlight);

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
                    rebuild_swapchain();
                }
            }

            if (window_.is_minimised()) {
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
    struct PerFrame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore image_available = VK_NULL_HANDLE;
        VkFence in_flight = VK_NULL_HANDLE;
    };

    void create_frames() {
        frames_.resize(kFramesInFlight);

        for (PerFrame& frame : frames_) {
            const VkCommandPoolCreateInfo pool_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = context_.queue_family(),
            };
            VK_CHECK(vkCreateCommandPool(context_.device(), &pool_info, nullptr,
                                         &frame.pool));

            const VkCommandBufferAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = frame.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            VK_CHECK(vkAllocateCommandBuffers(context_.device(), &alloc_info,
                                              &frame.cmd));

            const VkSemaphoreCreateInfo semaphore_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            };
            VK_CHECK(vkCreateSemaphore(context_.device(), &semaphore_info, nullptr,
                                       &frame.image_available));

            const VkFenceCreateInfo fence_info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                // Created already signalled. The first frame waits on this fence
                // before any work has been submitted, and an unsignalled fence would
                // mean waiting forever for something that never happened.
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };
            VK_CHECK(vkCreateFence(context_.device(), &fence_info, nullptr,
                                   &frame.in_flight));
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
        render_finished_.resize(swapchain_.image_count());
        for (VkSemaphore& semaphore : render_finished_) {
            const VkSemaphoreCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
            };
            VK_CHECK(vkCreateSemaphore(context_.device(), &info, nullptr, &semaphore));
        }
    }

    void destroy_image_semaphores() noexcept {
        for (VkSemaphore semaphore : render_finished_) {
            vkDestroySemaphore(context_.device(), semaphore, nullptr);
        }
        render_finished_.clear();
    }

    void rebuild_swapchain() {
        swapchain_.recreate();
        // The image count can change when the swapchain is rebuilt, and there is one
        // render-finished semaphore per image.
        destroy_image_semaphores();
        create_image_semaphores();
    }

    void draw_frame(bool capture_this_frame) {
        const VkDevice device = context_.device();
        PerFrame& frame = frames_[frame_number_ % kFramesInFlight];

        // Wait until the GPU has finished the *previous* frame that used this slot --
        // not the frame immediately before this one. With two slots in flight, that
        // is two frames ago, which is exactly why this usually does not block.
        VK_CHECK(vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX));

        // Acquire signals a semaphore this time, not a fence. The CPU does not need
        // to know when the image is ready; the GPU does, and it will wait on this
        // semaphore before writing any colour.
        uint32_t image_index = 0;
        const VkResult acquired =
            vkAcquireNextImageKHR(device, swapchain_.handle(), UINT64_MAX,
                                  frame.image_available, VK_NULL_HANDLE, &image_index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            // Nothing was acquired and the semaphore was not signalled, so we can
            // simply rebuild and skip this frame.
            rebuild_swapchain();
            return;
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
            vkc::throw_vulkan_error(acquired, "vkAcquireNextImageKHR", __FILE__,
                                    __LINE__);
        }

        // Reset only now that a submit is certain. Resetting before the early return
        // above would leave the fence unsignalled with no work to signal it, and the
        // next frame in this slot would hang forever.
        VK_CHECK(vkResetFences(device, 1, &frame.in_flight));
        VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        VK_CHECK(vkBeginCommandBuffer(frame.cmd, &begin_info));

        const VkImage image = swapchain_.image(image_index);
        const VkExtent2D extent = swapchain_.extent();

        transition_image(frame.cmd, image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Driven by the frame counter rather than the clock, so that frame N always
        // has the same colour. tools/capture.py depends on that: a screenshot taken
        // at frame 90 has to be identical every run or it cannot detect a real change.
        const float seconds = static_cast<float>(frame_number_) / 50.0F;
        const VkClearValue clear{
            .color = {{0.5F + 0.5F * std::sin(seconds),
                       0.5F + 0.5F * std::sin(seconds + 2.0F),
                       0.5F + 0.5F * std::sin(seconds + 4.0F), 1.0F}},
        };

        const VkRenderingAttachmentInfo colour_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = swapchain_.view(image_index),
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
            .renderArea = {{0, 0}, extent},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour_attachment,
            .pDepthAttachment = nullptr,
            .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(frame.cmd, &rendering);
        vkCmdEndRendering(frame.cmd);

        if (capture_this_frame) {
            capture_.record(frame.cmd, image, extent);
        } else {
            transition_image(frame.cmd, image,
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
        VK_CHECK(vkQueueSubmit2(context_.queue(), 1, &submit, frame.in_flight));

        // Present waits on the GPU side for rendering to finish. The CPU does not
        // block here at all -- it goes straight back to the event loop and starts
        // recording the next frame.
        const VkSwapchainKHR swapchain = swapchain_.handle();
        const VkPresentInfoKHR present{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_[image_index],
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
            .pResults = nullptr,
        };
        const VkResult presented = vkQueuePresentKHR(context_.queue(), &present);

        ++frame_number_;

        // A screenshot is the one place the CPU genuinely must wait, because it has
        // to read back what the GPU wrote. It happens once, on the last frame.
        if (capture_this_frame) {
            context_.wait_idle();
            capture_.write(args_.screenshot, swapchain_.format());
        }

        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
            rebuild_swapchain();
        } else if (presented != VK_SUCCESS) {
            vkc::throw_vulkan_error(presented, "vkQueuePresentKHR", __FILE__, __LINE__);
        }
    }

    vkc::Args args_;
    vkc::Window window_;
    vkc::Context context_;
    vkc::Swapchain swapchain_;

    std::vector<PerFrame> frames_;
    std::vector<VkSemaphore> render_finished_;  // one per swapchain image
    uint64_t frame_number_ = 0;

    vkc::Capture capture_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        FrameLoopApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
