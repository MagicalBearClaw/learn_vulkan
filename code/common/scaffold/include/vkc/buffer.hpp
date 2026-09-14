#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>

namespace vkc {

class Context;

// A VkBuffer and the VMA allocation behind it, freed together.
//
// Chapter 1.8 wrote all of this out by hand and explained why a buffer starts life with
// no memory attached, why one vkAllocateMemory per buffer runs out of allocations, and
// what VMA does about it. The only thing added here is C++ ownership: the destructor
// frees, copying is forbidden, and moving transfers.
class Buffer {
public:
    Buffer() = default;

    // `flags` is where host access is requested. Pass 0 for device-local memory the
    // CPU never touches; pass HOST_ACCESS_SEQUENTIAL_WRITE_BIT | MAPPED_BIT for
    // something the CPU writes and the mapped pointer comes back from mapped().
    Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
           VmaAllocationCreateFlags flags,
           VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO);

    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

    // Non-null only if the allocation was created with VMA_ALLOCATION_CREATE_MAPPED_BIT.
    [[nodiscard]] void* mapped() const noexcept { return info_.pMappedData; }

    [[nodiscard]] explicit operator bool() const noexcept {
        return buffer_ != VK_NULL_HANDLE;
    }

    void destroy() noexcept;

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo info_{};
    VkDeviceSize size_ = 0;
};

// Records one-off GPU work and waits for it to finish. Chapter 1.8's immediate_submit.
//
// Creates a transient pool and a fence each time, which is wasteful if you call it in a
// loop and irrelevant if you call it a handful of times at load. Every use in this
// series is the second kind.
void immediate_submit(Context& context,
                      const std::function<void(VkCommandBuffer)>& record);

// Uploads CPU data into a device-local buffer through a staging buffer, exactly as
// chapter 1.8 did it. TRANSFER_DST is added to `usage` for you.
[[nodiscard]] Buffer upload_to_device_local(Context& context, const void* data,
                                            VkDeviceSize size,
                                            VkBufferUsageFlags usage);

}  // namespace vkc
