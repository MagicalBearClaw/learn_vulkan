#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

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

// Loads an image file, uploads it, builds its mip chain with vkCmdBlitImage, and leaves
// it in SHADER_READ_ONLY_OPTIMAL. Chapter 1.11, start to finish.
//
// Pass VK_FORMAT_R8G8B8A8_UNORM for data textures -- normal maps, roughness, masks --
// which were never colour and must not go through the sRGB conversion.
[[nodiscard]] Image load_texture(Context& context, const std::filesystem::path& path,
                                 VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);

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
