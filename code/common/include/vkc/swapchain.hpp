#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

namespace vkc {

class Context;
class Window;

// The images the window system hands us to draw into, and the machinery for taking
// one and giving it back.
//
// A swapchain is not permanent: resizing the window, moving it to another monitor,
// or a driver update can all invalidate it. Every sample must be able to rebuild it
// mid-run, so recreation is built in from the very first chapter that has one.
class Swapchain {
public:
    Swapchain(Context& context, Window& window);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&&) = delete;
    Swapchain& operator=(Swapchain&&) = delete;

    // Tears down and rebuilds against the window's current size. Waits for the
    // device to be idle first, which is the blunt-but-correct approach; a later
    // chapter shows how to do it without the stall.
    void recreate();

    [[nodiscard]] VkSwapchainKHR handle() const noexcept { return swapchain_; }
    [[nodiscard]] VkFormat format() const noexcept { return format_; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
    [[nodiscard]] uint32_t image_count() const noexcept {
        return static_cast<uint32_t>(images_.size());
    }

    [[nodiscard]] VkImage image(uint32_t index) const { return images_.at(index); }
    [[nodiscard]] VkImageView view(uint32_t index) const { return views_.at(index); }

private:
    void create();
    void destroy() noexcept;

    Context& context_;
    Window& window_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;       // owned by the swapchain, not by us
    std::vector<VkImageView> views_;    // owned by us, one per image
};

}  // namespace vkc
