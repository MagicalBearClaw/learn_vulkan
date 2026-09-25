#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <array>
#include <cstdint>
#include <filesystem>

namespace vkc {

class Context;

// What an image is for. Everything an ImageDesc needs that is not a size.
struct ImageDesc {
    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
    VkExtent2D extent{};
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    uint32_t mip_levels = 1;
    // COLOR for textures and render targets, DEPTH for a depth buffer. It appears in
    // every barrier and every view, so it is part of the image's identity rather than
    // something to remember at each use.
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    // How many images are stacked inside this one. Chapter 4.4 is the first to want
    // more than one.
    uint32_t array_layers = 1;
    // Asks for VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT and makes the bundled view a
    // VK_IMAGE_VIEW_TYPE_CUBE, so the image is sampled by direction rather than by a
    // uv pair. Requires array_layers == 6, in the order the specification fixes:
    // +X, -X, +Y, -Y, +Z, -Z. Chapter 4.4 is the first to use one and explains
    // every field of it.
    bool cube = false;
};

// A VkImage, the VMA allocation behind it, and a view covering all of it.
//
// Chapter 1.11 wrote every line of this by hand and explained what a layout is, why
// tiling is OPTIMAL, and why the format ends in _SRGB. Bundling the view with the image
// is a convenience, not a rule: a second view onto the same image -- one mip level, or a
// different format -- is created separately when a chapter needs one.
class Image {
public:
    Image() = default;
    Image(Context& context, const ImageDesc& desc);

    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    [[nodiscard]] VkImage handle() const noexcept { return image_; }
    [[nodiscard]] VkImageView view() const noexcept { return view_; }
    [[nodiscard]] VkFormat format() const noexcept { return desc_.format; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return desc_.extent; }
    [[nodiscard]] uint32_t mip_levels() const noexcept { return desc_.mip_levels; }
    [[nodiscard]] VkImageAspectFlags aspect() const noexcept { return desc_.aspect; }
    [[nodiscard]] uint32_t array_layers() const noexcept { return desc_.array_layers; }

    [[nodiscard]] explicit operator bool() const noexcept {
        return image_ != VK_NULL_HANDLE;
    }

    // Moves some or all mip levels between layouts. Chapter 1.11's transition().
    void transition(VkCommandBuffer cmd, VkImageLayout from, VkImageLayout to,
                    uint32_t base_mip = 0,
                    uint32_t mip_count = VK_REMAINING_MIP_LEVELS) const;

    void destroy() noexcept;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    ImageDesc desc_{};
};

// Number of mip levels a texture of this size can have, down to 1x1.
[[nodiscard]] uint32_t mip_level_count(uint32_t width, uint32_t height) noexcept;

// Uploads four-byte-per-texel pixels, builds the mip chain with vkCmdBlitImage, and
// leaves the image in SHADER_READ_ONLY_OPTIMAL. This is the second half of chapter
// 1.11 -- everything after the decode -- split out so that a caller who already has
// pixels in memory need not write them to a file first. Chapter 3.2, whose textures
// arrive as JPEG blobs inside a .glb, is the first such caller.
[[nodiscard]] Image create_texture(Context& context, const void* rgba_pixels,
                                   VkExtent2D extent,
                                   VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);

// Loads an image file, uploads it, builds its mip chain with vkCmdBlitImage, and leaves
// it in SHADER_READ_ONLY_OPTIMAL. Chapter 1.11, start to finish.
//
// Pass VK_FORMAT_R8G8B8A8_UNORM for data textures -- normal maps, roughness, masks --
// which were never colour and must not go through the sRGB conversion.
[[nodiscard]] Image load_texture(Context& context, const std::filesystem::path& path,
                                 VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);

// Loads six square images of equal size into the six faces of one cubemap and leaves it
// in SHADER_READ_ONLY_OPTIMAL. The files must be given in the order the specification
// fixes -- +X, -X, +Y, -Y, +Z, -Z -- because the array layer *is* the face.
//
// Chapter 4.4 is the first caller, and explains why the image needs
// CUBE_COMPATIBLE, why the view type is CUBE, and why every face is uploaded from one
// staging buffer with one copy region each.
[[nodiscard]] Image load_cubemap(Context& context,
                                 const std::array<std::filesystem::path, 6>& faces,
                                 VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);

// Picks a depth format this GPU can use as a depth attachment, preferring the most
// precise. Chapter 1.13 wrote this out and explained the order: D32_SFLOAT first, the
// two stencil-carrying formats behind it as fallbacks.
[[nodiscard]] VkFormat choose_depth_format(VkPhysicalDevice physical_device);

// Creates a depth image and leaves it in DEPTH_ATTACHMENT_OPTIMAL, ready to attach.
// Chapter 1.13 again, called once at start-up and again on every resize.
[[nodiscard]] Image create_depth_buffer(Context& context, VkFormat format,
                                        VkExtent2D extent);

// How to read between texels. Kept separate from Image on purpose: a sampler is an
// independent object in Vulkan, and a handful of them serve a whole scene.
struct SamplerDesc {
    VkFilter mag_filter = VK_FILTER_LINEAR;
    VkFilter min_filter = VK_FILTER_LINEAR;
    VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    bool anisotropy = true;
};

class Sampler {
public:
    Sampler() = default;
    Sampler(Context& context, const SamplerDesc& desc);

    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;

    [[nodiscard]] VkSampler handle() const noexcept { return sampler_; }

    void destroy() noexcept;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace vkc
