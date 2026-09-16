#include "vkc/image.hpp"

#include "vkc/buffer.hpp"
#include "vkc/check.hpp"
#include "vkc/context.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <stdexcept>
#include <utility>

namespace vkc {

uint32_t mip_level_count(uint32_t width, uint32_t height) noexcept {
    return std::bit_width(std::max(width, height));
}

Image::Image(Context& context, const ImageDesc& desc)
    : device_(context.device()), allocator_(context.allocator()), desc_(desc) {
    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = desc_.format,
        .extent = {desc_.extent.width, desc_.extent.height, 1},
        .mipLevels = desc_.mip_levels,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = desc_.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo alloc_info{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };

    VK_CHECK(vmaCreateImage(allocator_, &image_info, &alloc_info, &image_, &allocation_,
                            nullptr));

    const VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = desc_.format,
        .components = {},
        .subresourceRange =
            {
                .aspectMask = desc_.aspect,
                .baseMipLevel = 0,
                .levelCount = desc_.mip_levels,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
    VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &view_));
}

Image::~Image() { destroy(); }

Image::Image(Image&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      view_(std::exchange(other.view_, VK_NULL_HANDLE)),
      desc_(std::exchange(other.desc_, {})) {}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        desc_ = std::exchange(other.desc_, {});
    }
    return *this;
}

void Image::destroy() noexcept {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

void Image::transition(VkCommandBuffer cmd, VkImageLayout from, VkImageLayout to,
                       uint32_t base_mip, uint32_t mip_count) const {
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
        .image = image_,
        .subresourceRange =
            {
                .aspectMask = desc_.aspect,
                .baseMipLevel = base_mip,
                .levelCount = mip_count,
                .baseArrayLayer = 0,
                .layerCount = 1,
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

namespace {

// Chapter 1.11's mip ladder: blit level n-1 into level n, halved, then park level n-1
// in the layout the shader wants.
void generate_mipmaps(VkCommandBuffer cmd, const Image& image) {
    int mip_width = static_cast<int>(image.extent().width);
    int mip_height = static_cast<int>(image.extent().height);

    for (uint32_t level = 1; level < image.mip_levels(); ++level) {
        image.transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, level - 1, 1);

        const int next_width = std::max(mip_width / 2, 1);
        const int next_height = std::max(mip_height / 2, 1);

        const VkImageBlit blit{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
            .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
            .dstOffsets = {{0, 0, 0}, {next_width, next_height, 1}},
        };

        vkCmdBlitImage(cmd, image.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);

        image.transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, level - 1, 1);

        mip_width = next_width;
        mip_height = next_height;
    }

    image.transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     image.mip_levels() - 1, 1);
}

}  // namespace

Image load_texture(Context& context, const std::filesystem::path& path,
                   VkFormat format) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels =
        stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (pixels == nullptr) {
        throw std::runtime_error(
            std::format("Could not load '{}': {}. Did you run tools/make_textures.py?",
                        path.string(), stbi_failure_reason()));
    }

    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device(), format, &properties);
    if ((properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
        stbi_image_free(pixels);
        throw std::runtime_error(
            "This GPU cannot linearly filter blits of this format, so mipmaps cannot "
            "be generated at load time.");
    }

    const VkExtent2D extent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

    Image image(context,
                ImageDesc{
                    .format = format,
                    .extent = extent,
                    .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT,
                    .mip_levels = mip_level_count(extent.width, extent.height),
                    .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                });

    Buffer staging(context.allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(staging.mapped(), pixels, static_cast<size_t>(size));
    stbi_image_free(pixels);

    immediate_submit(context, [&](VkCommandBuffer cmd) {
        image.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent.width, extent.height, 1},
        };
        vkCmdCopyBufferToImage(cmd, staging.handle(), image.handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        generate_mipmaps(cmd, image);
    });

    return image;
}

VkFormat choose_depth_format(VkPhysicalDevice physical_device) {
    // Most precise first. A format is only usable as a depth attachment if the GPU says
    // so, and asking is cheaper than assuming: D24_UNORM_S8_UINT is absent on some
    // desktop GPUs and D32_SFLOAT_S8_UINT on some mobile ones.
    constexpr std::array candidates{
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };

    for (const VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        if ((properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    throw std::runtime_error("No supported depth attachment format was found.");
}

Image create_depth_buffer(Context& context, VkFormat format, VkExtent2D extent) {
    // The GPU must have finished with the old depth buffer before it is replaced. A
    // resize already waits for idle, and at start-up there is nothing in flight, so this
    // costs nothing in either case.
    context.wait_idle();

    Image depth(context, ImageDesc{
                             .format = format,
                             .extent = extent,
                             .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                             .mip_levels = 1,
                             // The aspect is the easy one to forget, and it has to match
                             // in the view and in every barrier.
                             .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                         });

    // A new image is in UNDEFINED. Moving it to the attachment layout once here means no
    // frame ever has to, and loadOp CLEAR discards the contents anyway.
    immediate_submit(context, [&](VkCommandBuffer cmd) {
        depth.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    });

    context.name(depth.handle(), VK_OBJECT_TYPE_IMAGE, "depth buffer");
    return depth;
}

Sampler::Sampler(Context& context, const SamplerDesc& desc)
    : device_(context.device()) {
    const VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = desc.mag_filter,
        .minFilter = desc.min_filter,
        .mipmapMode = desc.mipmap_mode,
        .addressModeU = desc.address_mode,
        .addressModeV = desc.address_mode,
        .addressModeW = desc.address_mode,
        .mipLodBias = 0.0F,
        .anisotropyEnable = desc.anisotropy ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = context.gpu_properties().limits.maxSamplerAnisotropy,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0F,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VK_CHECK(vkCreateSampler(device_, &info, nullptr, &sampler_));
}

Sampler::~Sampler() { destroy(); }

Sampler::Sampler(Sampler&& other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      sampler_(std::exchange(other.sampler_, VK_NULL_HANDLE)) {}

Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    }
    return *this;
}

void Sampler::destroy() noexcept {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
}

}  // namespace vkc
