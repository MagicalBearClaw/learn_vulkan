#include "vkc/capture.hpp"

#include "vkc/check.hpp"

#include <spdlog/spdlog.h>
#include <stb_image_write.h>

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace vkc {
namespace {

uint32_t parse_uint(std::string_view text, const char* flag) {
    uint32_t value = 0;
    const auto [ptr, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || ptr != text.data() + text.size()) {
        throw std::runtime_error(
            std::format("{} expects a number, got '{}'", flag, text));
    }
    return value;
}

}  // namespace

Args parse_args(int argc, char** argv, Args defaults) {
    const std::span<char*> args(argv, static_cast<size_t>(argc));

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string_view flag = args[i];
        const auto next = [&](const char* name) -> std::string_view {
            if (i + 1 >= args.size()) {
                throw std::runtime_error(std::format("{} needs a value", name));
            }
            return args[++i];
        };

        if (flag == "--frames") {
            defaults.frame_limit = parse_uint(next("--frames"), "--frames");
        } else if (flag == "--screenshot") {
            defaults.screenshot = next("--screenshot");
            // A screenshot run that never stops would never write the file.
            if (defaults.frame_limit == 0) {
                defaults.frame_limit = 1;
            }
        } else if (flag == "--width") {
            defaults.width = parse_uint(next("--width"), "--width");
        } else if (flag == "--height") {
            defaults.height = parse_uint(next("--height"), "--height");
        } else if (flag == "--no-validation") {
            defaults.validation = false;
        } else if (flag == "--help" || flag == "-h") {
            spdlog::info(
                "options: --frames N  --screenshot PATH  --width N  --height N  "
                "--no-validation");
            std::exit(0);
        } else {
            throw std::runtime_error(std::format("unknown option '{}'", flag));
        }
    }
    return defaults;
}

Capture::~Capture() { destroy(); }

void Capture::init(VkDevice device, VmaAllocator allocator) {
    device_ = device;
    allocator_ = allocator;
}

void Capture::allocate(VkExtent2D extent) {
    destroy();
    extent_ = extent;

    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = static_cast<VkDeviceSize>(extent.width) * extent.height * 4,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    // HOST_ACCESS_RANDOM plus MAPPED gives a buffer the CPU can read straight out of
    // once the GPU is done writing it, with no second copy.
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VK_CHECK(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer_,
                             &allocation_, &allocation_info_));
}

void Capture::record(VkCommandBuffer cmd, VkImage image, VkExtent2D extent) {
    if (buffer_ == VK_NULL_HANDLE || extent_.width != extent.width ||
        extent_.height != extent.height) {
        allocate(extent);
    }

    const auto barrier = [cmd, image](VkImageLayout from, VkImageLayout to) {
        const VkImageMemoryBarrier2 image_barrier{
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
            .pImageMemoryBarriers = &image_barrier,
        };
        vkCmdPipelineBarrier2(cmd, &dependency);
    };

    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const VkBufferImageCopy2 region{
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext = nullptr,
        .bufferOffset = 0,
        .bufferRowLength = 0,  // 0 means "tightly packed to imageExtent"
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1},
    };
    const VkCopyImageToBufferInfo2 copy{
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .pNext = nullptr,
        .srcImage = image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstBuffer = buffer_,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(cmd, &copy);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    pending_ = true;
}

void Capture::write(const std::filesystem::path& path, VkFormat format) const {
    if (!pending_) {
        return;
    }

    const size_t byte_count = static_cast<size_t>(extent_.width) * extent_.height * 4;
    std::vector<uint8_t> pixels(byte_count);
    std::memcpy(pixels.data(), allocation_info_.pMappedData, byte_count);

    // Swapchains usually hand out BGRA; PNG wants RGBA, so swap red and blue.
    const bool is_bgra = format == VK_FORMAT_B8G8R8A8_SRGB ||
                         format == VK_FORMAT_B8G8R8A8_UNORM;
    if (is_bgra) {
        for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
            std::swap(pixels[i], pixels[i + 2]);
        }
    }

    const std::string filename = path.string();
    if (stbi_write_png(filename.c_str(), static_cast<int>(extent_.width),
                       static_cast<int>(extent_.height), 4, pixels.data(),
                       static_cast<int>(extent_.width) * 4) == 0) {
        spdlog::error("Could not write screenshot to {}", filename);
    } else {
        spdlog::info("Wrote {} ({}x{})", filename, extent_.width, extent_.height);
    }
}

void Capture::destroy() noexcept {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
    pending_ = false;
}

}  // namespace vkc
