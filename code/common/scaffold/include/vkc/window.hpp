#pragma once

#include <volk.h>

#include <cstdint>
#include <span>
#include <string_view>

struct SDL_Window;

namespace vkc {

// A resizable, Vulkan-capable OS window, plus the VkSurfaceKHR that lets Vulkan
// present to it. SDL3 owns the platform differences; we only ever see the surface.
class Window {
public:
    // `resizable` is false only for --screenshot runs, where a tiling window manager
    // would otherwise pick the size and the reference images would not be stable.
    Window(std::string_view title, uint32_t width, uint32_t height,
           bool resizable = true);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;

    // The instance extensions SDL needs for this platform's surface (VK_KHR_surface
    // plus one of the WSI extensions). Must be enabled when the instance is created,
    // which is why this is static: it is needed before a Window can exist.
    [[nodiscard]] static std::span<const char* const> required_instance_extensions();

    [[nodiscard]] VkSurfaceKHR create_surface(VkInstance instance);
    void destroy_surface(VkInstance instance) noexcept;

    [[nodiscard]] VkSurfaceKHR surface() const noexcept { return surface_; }
    [[nodiscard]] SDL_Window* handle() const noexcept { return window_; }

    // Size of the drawable area in pixels. Not the same as the window size on
    // displays with a scale factor, and it is the pixel size the swapchain wants.
    [[nodiscard]] VkExtent2D pixel_extent() const noexcept;

    // True while the window is minimised, in which case the drawable area is 0x0
    // and there is nothing sensible to render into.
    [[nodiscard]] bool is_minimised() const noexcept;

private:
    SDL_Window* window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};

}  // namespace vkc
