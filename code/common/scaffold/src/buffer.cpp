#include "vkc/buffer.hpp"

#include "vkc/check.hpp"
#include "vkc/context.hpp"

#include <cstring>
#include <utility>

namespace vkc {

Buffer::Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
               VmaAllocationCreateFlags flags, VmaMemoryUsage memory_usage)
    : allocator_(allocator), size_(size) {
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    const VmaAllocationCreateInfo alloc_info{
        .flags = flags,
        .usage = memory_usage,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };

    VK_CHECK(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer_,
                             &allocation_, &info_));
}

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      info_(std::exchange(other.info_, {})),
      size_(std::exchange(other.size_, 0)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        info_ = std::exchange(other.info_, {});
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void Buffer::destroy() noexcept {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        info_ = {};
        size_ = 0;
    }
}

void immediate_submit(Context& context,
                      const std::function<void(VkCommandBuffer)>& record) {
    const VkDevice device = context.device();

    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.queue_family(),
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr, &pool));

    const VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &cmd));

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));

    record(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    const VkCommandBufferSubmitInfo cmd_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmd,
        .deviceMask = 0,
    };

    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 0,
        .pWaitSemaphoreInfos = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
        .signalSemaphoreInfoCount = 0,
        .pSignalSemaphoreInfos = nullptr,
    };

    const VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &fence));

    VK_CHECK(vkQueueSubmit2(context.queue(), 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
}

Buffer upload_to_device_local(Context& context, const void* data, VkDeviceSize size,
                              VkBufferUsageFlags usage) {
    Buffer staging(context.allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(staging.mapped(), data, static_cast<size_t>(size));

    Buffer result(context.allocator(), size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  0);

    immediate_submit(context, [&](VkCommandBuffer cmd) {
        const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size};
        vkCmdCopyBuffer(cmd, staging.handle(), result.handle(), 1, &region);
    });

    // `staging` is destroyed as this returns, after immediate_submit has waited on its
    // fence -- so the GPU has certainly finished reading it.
    return result;
}

}  // namespace vkc
