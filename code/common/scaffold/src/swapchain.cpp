#include "vkc/swapchain.hpp"

#include "vkc/check.hpp"
#include "vkc/context.hpp"
#include "vkc/window.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <vector>

namespace vkc {
namespace {

// We want an 8-bit-per-channel sRGB target. Asking for an sRGB *format* means the
// hardware does the linear-to-sRGB conversion on write for free, which is the whole
// gamma-correction story handled correctly by default. The gamma chapter explains
// why that matters; until then it is simply the right thing to pick.
VkSurfaceFormatKHR choose_surface_format(
    const std::vector<VkSurfaceFormatKHR>& available) {
    for (const VkSurfaceFormatKHR& candidate : available) {
        const bool wanted_format = candidate.format == VK_FORMAT_B8G8R8A8_SRGB ||
                                   candidate.format == VK_FORMAT_R8G8B8A8_SRGB;
        if (wanted_format &&
            candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return candidate;
        }
    }
    // The spec guarantees at least one format exists, so falling back to the first
    // is always safe even if it is not ideal.
    return available.front();
}

// FIFO is the only present mode every implementation must support, and it is
// v-synced with no tearing. MAILBOX is the same guarantee without the latency, so
// take it when it is offered.
VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& available) {
    const bool has_mailbox = std::ranges::find(available, VK_PRESENT_MODE_MAILBOX_KHR) !=
                             available.end();
    return has_mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D window_size) {
    // A currentExtent of 0xFFFFFFFF is the window system saying "you decide";
    // anything else is it saying "this is the size, take it".
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }
    return VkExtent2D{
        .width = std::clamp(window_size.width, caps.minImageExtent.width,
                            caps.maxImageExtent.width),
        .height = std::clamp(window_size.height, caps.minImageExtent.height,
                             caps.maxImageExtent.height),
    };
}

}  // namespace

Swapchain::Swapchain(Context& context, Window& window)
    : context_(context), window_(window) {
    create();
}

Swapchain::~Swapchain() { destroy(); }

void Swapchain::recreate() {
    context_.wait_idle();
    destroy();
    create();
    spdlog::info("Swapchain rebuilt at {}x{}", extent_.width, extent_.height);
}

void Swapchain::create() {
    const VkPhysicalDevice gpu = context_.physical_device();
    const VkSurfaceKHR surface = context_.surface();

    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps));

    uint32_t format_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &format_count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &format_count,
                                                  formats.data()));

    uint32_t mode_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &mode_count,
                                                       nullptr));
    std::vector<VkPresentModeKHR> modes(mode_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &mode_count,
                                                       modes.data()));

    const VkSurfaceFormatKHR surface_format = choose_surface_format(formats);
    const VkPresentModeKHR present_mode = choose_present_mode(modes);
    extent_ = choose_extent(caps, window_.pixel_extent());
    format_ = surface_format.format;

    // One more than the minimum, so the CPU always has an image to draw into while
    // the presentation engine is busy with another. maxImageCount == 0 means
    // "no limit".
    uint32_t requested_images = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && requested_images > caps.maxImageCount) {
        requested_images = caps.maxImageCount;
    }

    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = requested_images,
        .imageFormat = format_,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent_,
        .imageArrayLayers = 1,
        // COLOR_ATTACHMENT to render into it; TRANSFER_DST so a later chapter can
        // blit a finished offscreen image onto it instead; TRANSFER_SRC so the
        // screenshot path can copy it back out to the CPU.
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        // One queue family does everything in this series, so no sharing is needed.
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        // Let the driver discard pixels hidden by another window.
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };

    VK_CHECK(vkCreateSwapchainKHR(context_.device(), &info, nullptr, &swapchain_));
    context_.name(swapchain_, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "swapchain");

    uint32_t image_count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count,
                                     nullptr));
    images_.resize(image_count);
    VK_CHECK(vkGetSwapchainImagesKHR(context_.device(), swapchain_, &image_count,
                                     images_.data()));

    views_.resize(image_count);
    for (uint32_t i = 0; i < image_count; ++i) {
        const VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = images_[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format_,
            .components = {},  // all VK_COMPONENT_SWIZZLE_IDENTITY
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        VK_CHECK(vkCreateImageView(context_.device(), &view_info, nullptr, &views_[i]));
    }
}

void Swapchain::destroy() noexcept {
    for (VkImageView view : views_) {
        vkDestroyImageView(context_.device(), view, nullptr);
    }
    views_.clear();
    // The images themselves belong to the swapchain; destroying it frees them.
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(context_.device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

}  // namespace vkc
