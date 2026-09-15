// 1.1 Creating a Window
//
// No Vulkan yet. Before we can draw anything we need somewhere to draw it, and a way
// to know when the user wants to stop. That is all this chapter does: open a window,
// pump its events, and close cleanly.
//
// SDL3 handles the platform differences. On Linux it will talk to Wayland or X11, on
// Windows to Win32, and we never have to know which.

#include <vkc/capture.hpp>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <format>
#include <stdexcept>

namespace {

class WindowApp {
public:
    explicit WindowApp(const vkc::Args& args) : args_(args) {
        // SDL_Init starts the subsystems we ask for. We only need video; audio and
        // gamepads come much later in the series.
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(
                std::format("SDL_Init failed: {}", SDL_GetError()));
        }

        // SDL_WINDOW_VULKAN tells SDL to load the Vulkan loader and to set the window
        // up so that a VkSurfaceKHR can be created from it later. Nothing Vulkan
        // happens yet, but asking for it now saves recreating the window in 1.2.
        // Resizable, except on a --screenshot run: tiling window managers choose the size
        // of a resizable window themselves, which would make the book's reference images
        // depend on whatever else is open. A fixed-size window gets the size asked for.
        window_ = SDL_CreateWindow("LearnVulkan - Creating a Window",
                                   static_cast<int>(args.width),
                                   static_cast<int>(args.height),
                                   SDL_WINDOW_VULKAN |
                                   (args.screenshot.empty() ? SDL_WINDOW_RESIZABLE : 0));
        if (window_ == nullptr) {
            SDL_Quit();
            throw std::runtime_error(
                std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
        }
    }

    ~WindowApp() {
        // Destroy in the reverse of the order things were created. This is a habit
        // worth forming now, because Vulkan is far less forgiving about it than SDL.
        if (window_ != nullptr) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    WindowApp(const WindowApp&) = delete;
    WindowApp& operator=(const WindowApp&) = delete;
    WindowApp(WindowApp&&) = delete;
    WindowApp& operator=(WindowApp&&) = delete;

    int run() {
        spdlog::info("Window open. Press Escape or close it to quit.");

        uint64_t frames = 0;
        bool running = true;

        while (running) {
            // Events queue up while we are busy. Drain the whole queue every frame,
            // or the window will feel unresponsive and the compositor may decide the
            // application has hung.
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_EVENT_QUIT:
                    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                        running = false;
                        break;
                    case SDL_EVENT_KEY_DOWN:
                        if (event.key.key == SDLK_ESCAPE) {
                            running = false;
                        }
                        break;
                    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                        // From 1.4 onward this is the signal to rebuild the
                        // swapchain. For now it is only worth reporting.
                        spdlog::info("Window resized to {}x{} pixels", event.window.data1,
                                     event.window.data2);
                        break;
                    default:
                        break;
                }
            }

            // There is nothing to draw yet, so the window keeps whatever the
            // compositor last put there. It will look like an empty or garbage-filled
            // rectangle, and that is correct for this chapter.

            ++frames;
            if (args_.frame_limit != 0 && frames >= args_.frame_limit) {
                running = false;
            }
        }

        spdlog::info("Ran {} frames.", frames);
        return 0;
    }

private:
    vkc::Args args_;
    SDL_Window* window_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        WindowApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
