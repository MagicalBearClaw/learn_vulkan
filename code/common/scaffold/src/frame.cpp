#include "vkc/frame.hpp"

#include "vkc/check.hpp"
#include "vkc/context.hpp"
#include "vkc/swapchain.hpp"

#include <format>

namespace vkc {

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                      VkImageLayout to) {
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        // ALL_COMMANDS is the sledgehammer: it says "everything before this point,
        // everywhere in the pipeline". Correct, and slower than it needs to be.
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        .oldLayout = from,
        .newLayout = to,
        // No queue-family ownership transfer: one queue does everything here.
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

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
                   VkImageLayout to, VkPipelineStageFlags2 src_stage,
                   VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                   VkAccessFlags2 dst_access, VkImageAspectFlags aspect) {
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = aspect,
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

void begin_rendering(VkCommandBuffer cmd, VkImageView colour_view, VkImageView depth_view,
                     VkExtent2D extent, const VkClearColorValue& clear_colour) {
    const VkRenderingAttachmentInfo colour_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = colour_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = VkClearValue{.color = clear_colour},
    };

    // 1.0 is the far plane: clearing to it means every fragment is closer than what is
    // already there, which is what makes VK_COMPARE_OP_LESS work on the first draw.
    const VkClearValue depth_clear{.depthStencil = {1.0F, 0}};
    const VkRenderingAttachmentInfo depth_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = depth_clear,
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
        .pDepthAttachment = &depth_attachment,
        .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &rendering);
}

FrameContext::FrameContext(Context& context, Swapchain& swapchain)
    : context_(context), swapchain_(swapchain) {
    create_per_frame_objects();
    create_per_image_objects();
}

FrameContext::~FrameContext() {
    // Nothing may be destroyed while the GPU might still be reading it.
    context_.wait_idle();

    destroy_per_image_objects();

    for (PerFrame& frame : frames_) {
        vkDestroyFence(context_.device(), frame.in_flight, nullptr);
        vkDestroySemaphore(context_.device(), frame.image_available, nullptr);
        // Destroying the pool frees the command buffers allocated from it.
        vkDestroyCommandPool(context_.device(), frame.pool, nullptr);
    }
    frames_.clear();
}

void FrameContext::create_per_frame_objects() {
    frames_.resize(kFramesInFlight);

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        PerFrame& frame = frames_[i];

        const VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            // We reset and re-record the whole buffer every frame, so let the pool
            // reuse its memory instead of reallocating.
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
        VK_CHECK(vkAllocateCommandBuffers(context_.device(), &alloc_info, &frame.cmd));

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
            // Created already signalled, so the very first frame does not deadlock
            // waiting for work that was never submitted.
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        VK_CHECK(vkCreateFence(context_.device(), &fence_info, nullptr,
                               &frame.in_flight));

        const std::string label = std::format("frame[{}]", i);
        context_.name(frame.pool, VK_OBJECT_TYPE_COMMAND_POOL, label.c_str());
        context_.name(frame.cmd, VK_OBJECT_TYPE_COMMAND_BUFFER, label.c_str());
    }
}

void FrameContext::create_per_image_objects() {
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

void FrameContext::destroy_per_image_objects() noexcept {
    for (VkSemaphore semaphore : render_finished_) {
        vkDestroySemaphore(context_.device(), semaphore, nullptr);
    }
    render_finished_.clear();
}

void FrameContext::on_swapchain_recreated() {
    context_.wait_idle();
    destroy_per_image_objects();
    create_per_image_objects();
}

std::optional<FrameInfo> FrameContext::begin() {
    PerFrame& frame = frames_[frame_number_ % kFramesInFlight];

    // Wait until the GPU has finished the last frame that used this slot. This is
    // the only CPU/GPU sync point in a well-behaved frame loop.
    VK_CHECK(vkWaitForFences(context_.device(), 1, &frame.in_flight, VK_TRUE,
                             UINT64_MAX));

    uint32_t image_index = 0;
    const VkResult acquired =
        vkAcquireNextImageKHR(context_.device(), swapchain_.handle(), UINT64_MAX,
                              frame.image_available, VK_NULL_HANDLE, &image_index);

    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        // The window changed size under us. Nothing was acquired and the semaphore
        // was not signalled, so we can simply bail out and rebuild.
        return std::nullopt;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        throw_vulkan_error(acquired, "vkAcquireNextImageKHR", __FILE__, __LINE__);
    }

    // Reset only now that we know we are definitely going to submit: an early reset
    // followed by an early return would leave the fence unsignalled forever.
    VK_CHECK(vkResetFences(context_.device(), 1, &frame.in_flight));
    VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(frame.cmd, &begin_info));

    return FrameInfo{
        .cmd = frame.cmd,
        .image_index = image_index,
        .image = swapchain_.image(image_index),
        .view = swapchain_.view(image_index),
        .extent = swapchain_.extent(),
        .frame_number = frame_number_,
    };
}

bool FrameContext::end(const FrameInfo& info) {
    PerFrame& frame = frames_[frame_number_ % kFramesInFlight];

    VK_CHECK(vkEndCommandBuffer(info.cmd));

    const VkSemaphoreSubmitInfo wait{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = frame.image_available,
        .value = 0,  // ignored for binary semaphores
        // Wait only at the point where we actually write colour. Vertex processing
        // can start before the image is ready.
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .deviceIndex = 0,
    };

    const VkSemaphoreSubmitInfo signal{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = render_finished_[info.image_index],
        .value = 0,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };

    const VkCommandBufferSubmitInfo cmd_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = info.cmd,
        .deviceMask = 0,
    };

    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &wait,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal,
    };

    VK_CHECK(vkQueueSubmit2(context_.queue(), 1, &submit, frame.in_flight));

    const VkSwapchainKHR swapchain = swapchain_.handle();
    const VkPresentInfoKHR present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_finished_[info.image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &info.image_index,
        .pResults = nullptr,
    };

    const VkResult presented = vkQueuePresentKHR(context_.queue(), &present);

    ++frame_number_;

    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
        return false;
    }
    if (presented != VK_SUCCESS) {
        throw_vulkan_error(presented, "vkQueuePresentKHR", __FILE__, __LINE__);
    }
    return true;
}

}  // namespace vkc
