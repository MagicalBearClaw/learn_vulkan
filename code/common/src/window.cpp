#include "vkc/window.hpp"

#include "vkc/check.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <format>
#include <stdexcept>

namespace vkc {
namespace {

// SDL wants to be initialised exactly once per process. The samples only ever open
// one window, but keeping the guard here means a future chapter can open a second
// one without having to think about it.
struct SdlVideoSubsystem {
    SdlVideoSubsystem() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(
                std::format("SDL_Init(SDL_INIT_VIDEO) failed: {}", SDL_GetError()));
        }
    }
    ~SdlVideoSubsystem() { SDL_Quit(); }
};

void ensure_sdl_initialised() {
    static SdlVideoSubsystem subsystem;
    (void)subsystem;
}

}  // namespace

Window::Window(std::string_view title, uint32_t width, uint32_t height) {
    ensure_sdl_initialised();

    const std::string owned_title(title);
    window_ = SDL_CreateWindow(owned_title.c_str(), static_cast<int>(width),
                               static_cast<int>(height),
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (window_ == nullptr) {
        throw std::runtime_error(
            std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
    }
}

Window::~Window() {
    // The surface must already be gone: it belongs to the VkInstance, which outlives
    // neither the window nor this destructor in any sane teardown order.
    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
    }
}

std::span<const char* const> Window::required_instance_extensions() {
    ensure_sdl_initialised();

    uint32_t count = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (extensions == nullptr) {
        throw std::runtime_error(std::format(
            "SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
    }
    return {extensions, count};
}

VkSurfaceKHR Window::create_surface(VkInstance instance) {
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface_)) {
        throw std::runtime_error(
            std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
    }
    return surface_;
}

void Window::destroy_surface(VkInstance instance) noexcept {
    if (surface_ != VK_NULL_HANDLE) {
        SDL_Vulkan_DestroySurface(instance, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
}

VkExtent2D Window::pixel_extent() const noexcept {
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(window_, &width, &height);
    return VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

bool Window::is_minimised() const noexcept {
    return (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0;
}

}  // namespace vkc
