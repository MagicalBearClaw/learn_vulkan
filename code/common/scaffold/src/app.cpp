#include "vkc/app.hpp"

#include "vkc/check.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <optional>
#include <utility>

namespace vkc {
App::Options App::parse_args(int argc, char** argv, Options defaults) {
    // The flags themselves are shared with every hand-written chapter, so they are
    // parsed in one place. Only the title is App's own.
    static_cast<Args&>(defaults) = vkc::parse_args(argc, argv, defaults);
    return defaults;
}

App::App(Options options) : options_(std::move(options)) {
    window_ = std::make_unique<Window>(options_.title, options_.width, options_.height,
                                       /*resizable=*/options_.screenshot.empty());

    const Context::Config config{
        .app_name = options_.title,
        .enable_validation = options_.validation,
    };
    context_ = std::make_unique<Context>(config, *window_);
    swapchain_ = std::make_unique<Swapchain>(*context_, *window_);
    frames_ = std::make_unique<FrameContext>(*context_, *swapchain_);

    capture_.init(context_->device(), context_->allocator());
}

App::~App() = default;

int App::run() {
    on_start();

    last_tick_ = SDL_GetPerformanceCounter();
    const double tick_frequency =
        static_cast<double>(SDL_GetPerformanceFrequency());

    while (running_) {
        if (!poll_events()) {
            break;
        }

        // A minimised window has a zero-sized drawable. There is nothing to render
        // into and the swapchain cannot be built, so idle rather than spin.
        if (window_->is_minimised()) {
            SDL_WaitEvent(nullptr);
            continue;
        }

        if (swapchain_dirty_) {
            rebuild_swapchain();
        }

        const uint64_t now = SDL_GetPerformanceCounter();
        delta_time_ = static_cast<float>(
            static_cast<double>(now - last_tick_) / tick_frequency);
        last_tick_ = now;
        elapsed_ += delta_time_;

        std::optional<FrameInfo> frame = frames_->begin();
        if (!frame.has_value()) {
            swapchain_dirty_ = true;
            continue;
        }

        // Swapchain images come back in UNDEFINED layout (their previous contents are
        // not ours to keep) and must be in COLOR_ATTACHMENT_OPTIMAL before anything
        // renders into them.
        transition_image(frame->cmd, frame->image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        on_render(*frame);

        // A requested screenshot is taken on the final frame, while this image is
        // still ours. Once it has been handed to vkQueuePresentKHR it may not be
        // read, copied or transitioned again.
        const bool capture_this_frame =
            !options_.screenshot.empty() && options_.frame_limit != 0 &&
            frames_rendered_ + 1 == options_.frame_limit;

        if (capture_this_frame) {
            capture_.record(frame->cmd, frame->image, frame->extent);
        } else {
            // ...and in PRESENT_SRC_KHR before the presentation engine will take them.
            transition_image(frame->cmd, frame->image,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }

        if (!frames_->end(*frame)) {
            swapchain_dirty_ = true;
        }

        ++frames_rendered_;

        if (options_.frame_limit != 0 && frames_rendered_ >= options_.frame_limit) {
            running_ = false;
        }
    }

    context_->wait_idle();

    if (capture_.has_pending_capture()) {
        capture_.write(options_.screenshot, swapchain_->format());
    }
    capture_.destroy();

    on_shutdown();
    return 0;
}

bool App::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        // Samples see every event first. Nothing here can suppress the window
        // management below, which a chapter should never have to think about.
        on_event(event);

        switch (event.type) {
            case SDL_EVENT_QUIT:
                return false;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                return false;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                swapchain_dirty_ = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_ESCAPE) {
                    return false;
                }
                break;
            default:
                break;
        }
    }
    return true;
}

void App::rebuild_swapchain() {
    swapchain_->recreate();
    frames_->on_swapchain_recreated();
    swapchain_dirty_ = false;
    on_resize(swapchain_->extent());
}

}  // namespace vkc
