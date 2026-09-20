// 1.5 Command Buffers
//
// Adds to 1.4: a command pool, a command buffer, and the first thing you will
// actually see -- a cleared window.
//
// Vulkan has no "draw this now" call. You record commands into a VkCommandBuffer,
// then submit the whole buffer to a queue, and the GPU gets to it when it gets to it.
// Recording is cheap and happens on the CPU; submitting is the moment work crosses
// over to the GPU.
//
// One frame, in full:
//
//   1. ask the swapchain for an image        vkAcquireNextImageKHR
//   2. record: transition, clear, transition into the command buffer
//   3. submit it to the queue                vkQueueSubmit2
//   4. hand the image back to be displayed   vkQueuePresentKHR
//
// The synchronisation here is deliberately the simplest thing that is correct: after
// submitting, we wait for the GPU to go completely idle before doing anything else.
// That means the CPU and GPU take turns instead of working at the same time, which
// throws away most of the machine. Chapter 1.6 is about fixing exactly that, and it
// is easier to appreciate the fix having seen the problem.
//
// Four things now come from the scaffold. The window is 1.1, the instance is 1.2, the
// device is 1.3, and the swapchain is 1.4 -- and because every chapter from here wants
// all of the first three at once, they arrive together as vkc::Context.

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

namespace {

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

class ClearApp {
public:
    explicit ClearApp(const vkc::Args& args)
        : args_(args),
          window_("LearnVulkan - Command Buffers", args.width, args.height,
                  /*resizable=*/args.screenshot.empty()),
          context_(vkc::Context::Config{
                       .app_name = "LearnVulkan",
                       .enable_validation = args.validation,
                   },
                   window_),
          swapchain_(context_, window_) {
        create_commands();
        capture_.init(context_.device(), context_.allocator());
    }

    ~ClearApp() {
        // Wait for the GPU before tearing anything down: it may still be reading
        // the command buffer we are about to free.
        context_.wait_idle();

        capture_.destroy();
        vkDestroyFence(context_.device(), acquire_fence_, nullptr);
        vkDestroyFence(context_.device(), submit_fence_, nullptr);
        // Destroying a pool frees every command buffer allocated from it.
        vkDestroyCommandPool(context_.device(), command_pool_, nullptr);
        // swapchain_, context_ and window_ destroy themselves, in that order.
    }

    ClearApp(const ClearApp&) = delete;
    ClearApp& operator=(const ClearApp&) = delete;
    ClearApp(ClearApp&&) = delete;
    ClearApp& operator=(ClearApp&&) = delete;

    int run() {
        spdlog::info("Clearing to a cycling colour. Escape or close to quit.");

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
                    swapchain_.recreate();
                }
            }

            // Minimised windows have a zero-sized drawable, so there is nothing
            // to render into and no swapchain to build.
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
    void create_commands() {
        // A pool owns the memory that command buffers are recorded into. Pools are
        // not thread-safe, so the rule in a threaded renderer is one pool per thread;
        // we have one thread and therefore one pool.
        const VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            // We re-record the same buffer every frame, so let it be reset
            // individually rather than reallocated.
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = context_.queue_family(),
        };
        VK_CHECK(vkCreateCommandPool(context_.device(), &pool_info, nullptr,
                                     &command_pool_));

        const VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool_,
            // PRIMARY can be submitted to a queue. SECONDARY can only be called from
            // a primary one, and is a tool for recording in parallel.
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(context_.device(), &alloc_info,
                                          &command_buffer_));

        // A fence is how the CPU finds out that the GPU has finished something.
        // vkAcquireNextImageKHR insists on being given a semaphore or a fence -- it
        // will not let you ignore the question of when the image is ready -- and a
        // fence is the one the CPU can wait on directly.
        const VkFenceCreateInfo fence_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        VK_CHECK(vkCreateFence(context_.device(), &fence_info, nullptr, &acquire_fence_));
        VK_CHECK(vkCreateFence(context_.device(), &fence_info, nullptr, &submit_fence_));
    }

    void draw_frame(bool capture_this_frame) {
        const VkDevice device = context_.device();

        // 1. Ask for an image. This returns immediately with an index, but the image
        //    is not necessarily ready to use yet -- the fence tells us when it is.
        uint32_t image_index = 0;
        const VkResult acquired =
            vkAcquireNextImageKHR(device, swapchain_.handle(), UINT64_MAX,
                                  VK_NULL_HANDLE, acquire_fence_, &image_index);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain_.recreate();
            return;
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
            vkc::throw_vulkan_error(acquired, "vkAcquireNextImageKHR", __FILE__,
                                    __LINE__);
        }

        VK_CHECK(vkWaitForFences(device, 1, &acquire_fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &acquire_fence_));

        // 2. Record. Nothing executes here; we are writing a list.
        VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));
        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            // This buffer is submitted once and then reset, which lets the driver
            // optimise for that case.
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        VK_CHECK(vkBeginCommandBuffer(command_buffer_, &begin_info));

        const VkImage image = swapchain_.image(image_index);
        const VkExtent2D extent = swapchain_.extent();

        // Swapchain images come back in UNDEFINED layout: whatever was in them is not
        // ours to keep. UNDEFINED as the source layout means "discard the contents",
        // which is free -- and correct here, because we are about to clear anyway.
        transition_image(command_buffer_, image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // A slow colour cycle, so that a frozen frame is obvious at a glance.
        // Driven by the frame counter rather than the clock, so that frame N always
        // has the same colour. tools/capture.py depends on that: a screenshot taken
        // at frame 90 has to be identical every run or it cannot detect a real change.
        const float seconds = static_cast<float>(frame_number_) / 50.0F;
        const VkClearValue clear{
            .color = {{0.5F + 0.5F * std::sin(seconds),
                       0.5F + 0.5F * std::sin(seconds + 2.0F),
                       0.5F + 0.5F * std::sin(seconds + 4.0F), 1.0F}},
        };

        // Dynamic rendering: describe the attachment here, at the point of use. In
        // Vulkan 1.0 this needed a VkRenderPass and a VkFramebuffer created up front
        // and kept in step with the swapchain by hand.
        const VkRenderingAttachmentInfo colour_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = swapchain_.view(image_index),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            // CLEAR means the clear happens as part of starting the pass rather than
            // as a separate pass over every pixel. It is effectively free.
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

        vkCmdBeginRendering(command_buffer_, &rendering);
        // No draw calls yet. The triangle arrives in 1.7.
        vkCmdEndRendering(command_buffer_);

        // The presentation engine will only accept an image in this layout. When a
        // screenshot was asked for, the capture helper records the copy and performs
        // this same transition on the way out -- the image still gets presented.
        if (capture_this_frame) {
            capture_.record(command_buffer_, image, extent);
        } else {
            transition_image(command_buffer_, image,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }

        VK_CHECK(vkEndCommandBuffer(command_buffer_));

        // 3. Submit. This is where the work crosses to the GPU.
        const VkCommandBufferSubmitInfo cmd_submit{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = command_buffer_,
            .deviceMask = 0,
        };
        const VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            // No semaphores yet: we are about to block on the fence instead, which is
            // what makes this version simple and slow.
            .waitSemaphoreInfoCount = 0,
            .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_submit,
            .signalSemaphoreInfoCount = 0,
            .pSignalSemaphoreInfos = nullptr,
        };
        VK_CHECK(vkQueueSubmit2(context_.queue(), 1, &submit, submit_fence_));

        // Wait for the GPU to finish the frame entirely. This is the line that makes
        // the CPU and GPU take turns. Deleting it would be a bug, not an
        // optimisation -- the fix is semaphores, in 1.6.
        VK_CHECK(vkWaitForFences(device, 1, &submit_fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device, 1, &submit_fence_));

        // The GPU has finished, so the copy the capture helper recorded is now
        // sitting in host-visible memory and can be written out.
        if (capture_this_frame) {
            capture_.write(args_.screenshot, swapchain_.format());
        }

        ++frame_number_;

        // 4. Give the image back. Because the GPU is idle, there is nothing left to
        //    wait on, so this needs no wait semaphores either.
        const VkSwapchainKHR swapchain = swapchain_.handle();
        const VkPresentInfoKHR present{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
            .pResults = nullptr,
        };
        const VkResult presented = vkQueuePresentKHR(context_.queue(), &present);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
            swapchain_.recreate();
        } else if (presented != VK_SUCCESS) {
            vkc::throw_vulkan_error(presented, "vkQueuePresentKHR", __FILE__, __LINE__);
        }
    }

    vkc::Args args_;
    vkc::Window window_;
    vkc::Context context_;
    vkc::Swapchain swapchain_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence acquire_fence_ = VK_NULL_HANDLE;
    VkFence submit_fence_ = VK_NULL_HANDLE;
    uint64_t frame_number_ = 0;
    vkc::Capture capture_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        ClearApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
