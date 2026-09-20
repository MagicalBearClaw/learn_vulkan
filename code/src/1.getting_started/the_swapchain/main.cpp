// 1.4 The Swapchain
//
// Adds to 1.3: a swapchain and a view onto each of its images.
//
// The swapchain is the set of images the window system will actually display, plus
// the rules for handing them back and forth. You do not create the images; the
// presentation engine owns them and lends them to you one at a time.
//
// Three decisions define it, and each one is a negotiation with what the surface
// actually supports:
//
//   format        how a pixel is stored, and whether the hardware converts to sRGB
//   present mode  what happens when you produce frames faster than the display
//                 consumes them
//   extent        how big the images are, in pixels
//
// A swapchain is also not permanent. Resize the window and it becomes invalid, so
// rebuilding it is a normal part of running, not an error path.
//
// The window, the instance and the device now come from the scaffold: 1.1, 1.2 and
// 1.3 wrote all three by hand and explained every line. Creating the surface is one
// of them too -- 1.3 wrote the SDL call out, so here it is window_.create_surface().

#include <vkc/capture.hpp>
#include <vkc/check.hpp>
#include <vkc/device.hpp>
#include <vkc/instance.hpp>
#include <vkc/window.hpp>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

namespace {

// An 8-bit-per-channel sRGB target. Asking for an sRGB *format* means the hardware
// converts from linear to sRGB as it writes, which is gamma correction done properly
// and for free. The gamma chapter explains why that matters; for now it is simply the
// right default, and picking a UNORM format here would quietly make everything later
// in the series too dark.
const char* format_name(VkFormat format) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_SRGB: return "VK_FORMAT_B8G8R8A8_SRGB";
        case VK_FORMAT_R8G8B8A8_SRGB: return "VK_FORMAT_R8G8B8A8_SRGB";
        case VK_FORMAT_B8G8R8A8_UNORM: return "VK_FORMAT_B8G8R8A8_UNORM";
        case VK_FORMAT_R8G8B8A8_UNORM: return "VK_FORMAT_R8G8B8A8_UNORM";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
        default: return "some other format";
    }
}

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& available) {
    for (const VkSurfaceFormatKHR& candidate : available) {
        const bool wanted = candidate.format == VK_FORMAT_B8G8R8A8_SRGB ||
                            candidate.format == VK_FORMAT_R8G8B8A8_SRGB;
        if (wanted && candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return candidate;
        }
    }
    // The specification guarantees at least one format, so this is always safe.
    return available.front();
}

// FIFO is a queue: finished frames wait their turn and are shown on a display refresh.
// It is the only mode every implementation must support, and it never tears.
// MAILBOX is the same guarantee except that a newer frame replaces a waiting one
// instead of queueing behind it, which trades GPU work for latency. Take it if it is
// offered.
VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& available) {
    const bool has_mailbox =
        std::ranges::find(available, VK_PRESENT_MODE_MAILBOX_KHR) != available.end();
    return has_mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D window_size) {
    // A currentExtent of 0xFFFFFFFF is the window system saying "you choose".
    // Anything else is it saying "this is the size, take it".
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

class SwapchainApp {
public:
    explicit SwapchainApp(const vkc::Args& args)
        : args_(args),
          window_("LearnVulkan - The Swapchain", args.width, args.height,
                  /*resizable=*/args.screenshot.empty()),
          instance_(vkc::Instance::Config{
              .app_name = "LearnVulkan",
              .enable_validation = args.validation,
          }) {
        surface_ = window_.create_surface(instance_.handle());
        device_ = std::make_unique<vkc::Device>(instance_, surface_);
        create_swapchain();
    }

    ~SwapchainApp() {
        // Strict reverse order. Vulkan will not warn you at run time if you get this
        // wrong, but the validation layers will, loudly.
        destroy_swapchain();
        device_.reset();
        window_.destroy_surface(instance_.handle());
        // instance_ and window_ destroy themselves, in that order, after this runs.
    }

    SwapchainApp(const SwapchainApp&) = delete;
    SwapchainApp& operator=(const SwapchainApp&) = delete;
    SwapchainApp(SwapchainApp&&) = delete;
    SwapchainApp& operator=(SwapchainApp&&) = delete;

    int run() {
        report();

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
                    // The old swapchain is now the wrong size. Rebuilding here is
                    // simple because nothing is in flight yet; from 1.6 onward there
                    // will be frames on the GPU to wait for first.
                    recreate_swapchain();
                }
            }
            ++frames;
            if (args_.frame_limit != 0 && frames >= args_.frame_limit) {
                running = false;
            }
        }
        return 0;
    }

private:
    void create_swapchain() {
        const VkPhysicalDevice gpu = device_->physical_device();

        VkSurfaceCapabilitiesKHR caps{};
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface_, &caps));

        uint32_t format_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface_, &format_count,
                                                      nullptr));
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface_, &format_count,
                                                      formats.data()));

        uint32_t mode_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface_, &mode_count,
                                                           nullptr));
        std::vector<VkPresentModeKHR> modes(mode_count);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface_, &mode_count,
                                                           modes.data()));

        const VkSurfaceFormatKHR surface_format = choose_surface_format(formats);
        present_mode_ = choose_present_mode(modes);
        format_ = surface_format.format;

        // The window's drawable size in pixels, which is not its size in points on a
        // display with a scale factor. Chapter 1.1 made that distinction; this is the
        // first place it matters.
        extent_ = choose_extent(caps, window_.pixel_extent());

        // One more than the minimum, so that the CPU always has an image to draw into
        // while the presentation engine is busy with another. A maxImageCount of 0
        // means there is no upper limit.
        uint32_t image_count = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
            image_count = caps.maxImageCount;
        }

        const VkSwapchainCreateInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = format_,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent_,
            // Always 1 unless you are rendering stereo for a headset.
            .imageArrayLayers = 1,
            // COLOR_ATTACHMENT to render into them; TRANSFER_DST so a later chapter
            // can blit a finished offscreen image on instead; TRANSFER_SRC so the
            // screenshot tooling can copy one back out.
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            // One queue family does everything here, so the images never need to be
            // shared between families.
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .preTransform = caps.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode_,
            // Lets the driver skip pixels another window is covering.
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };

        VK_CHECK(vkCreateSwapchainKHR(device_->handle(), &info, nullptr, &swapchain_));

        // We asked for a minimum; the driver may well have given us more.
        uint32_t actual_count = 0;
        VK_CHECK(vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &actual_count,
                                         nullptr));
        images_.resize(actual_count);
        VK_CHECK(vkGetSwapchainImagesKHR(device_->handle(), swapchain_, &actual_count,
                                         images_.data()));

        // A VkImage cannot be rendered into directly. A view says which part of it to
        // use and how to interpret the bits -- here, all of it, as a 2D colour image.
        views_.resize(actual_count);
        for (uint32_t i = 0; i < actual_count; ++i) {
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
            VK_CHECK(vkCreateImageView(device_->handle(), &view_info, nullptr,
                                       &views_[i]));
        }
    }

    void destroy_swapchain() noexcept {
        for (VkImageView view : views_) {
            vkDestroyImageView(device_->handle(), view, nullptr);
        }
        views_.clear();
        // The images belong to the swapchain. Destroying it frees them; destroying
        // them yourself is an error.
        images_.clear();

        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_->handle(), swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
    }

    void recreate_swapchain() {
        // Nothing may be destroyed while the GPU could still be using it. Waiting for
        // the whole device is the blunt version of this; a later chapter does it
        // without the stall.
        device_->wait_idle();
        destroy_swapchain();
        create_swapchain();
        spdlog::info("Swapchain rebuilt at {}x{}", extent_.width, extent_.height);
    }

    void report() const {
        spdlog::info("Swapchain:");
        spdlog::info("  {} images at {}x{}", images_.size(), extent_.width,
                     extent_.height);
        spdlog::info("  format: {}", format_name(format_));
        spdlog::info("  present mode: {}",
                     present_mode_ == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO");
        spdlog::info(
            "Images exist, but nothing has been recorded into them yet. Try resizing "
            "the window - the swapchain is rebuilt each time.");
    }

    vkc::Args args_;
    vkc::Window window_;
    vkc::Instance instance_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    std::unique_ptr<vkc::Device> device_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;     // owned by the swapchain
    std::vector<VkImageView> views_;  // owned by us
};

}  // namespace

int main(int argc, char** argv) {
    try {
        SwapchainApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
