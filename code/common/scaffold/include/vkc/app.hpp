#pragma once

#include "vkc/capture.hpp"
#include "vkc/context.hpp"
#include "vkc/frame.hpp"
#include "vkc/swapchain.hpp"
#include "vkc/window.hpp"

#include <SDL3/SDL_events.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace vkc {

// Base class for every sample in the series.
//
// It owns the four objects that never change between chapters -- window, context,
// swapchain, frame loop -- and hands a derived sample exactly one thing: an open
// command buffer with a swapchain image already in COLOR_ATTACHMENT_OPTIMAL layout.
// What a chapter does with that command buffer is the entire subject of the chapter.
class App {
public:
    // Everything in Args (size, frame limit, screenshot path, validation) plus the
    // window title. Args is shared with the hand-written chapters so that
    // tools/capture.py can drive every sample in the book identically.
    struct Options : Args {
        std::string title = "LearnVulkan";
    };

    explicit App(Options options);
    virtual ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;
    App(App&&) = delete;
    App& operator=(App&&) = delete;

    // Parses the flags every sample understands: --frames N, --screenshot PATH,
    // --no-validation, --width N, --height N.
    [[nodiscard]] static Options parse_args(int argc, char** argv, Options defaults);

    // Runs the loop. Returns a process exit code.
    int run();

protected:
    // Called once after the context and swapchain exist, before the first frame.
    virtual void on_start() {}
    // Called with an open command buffer, once per frame.
    virtual void on_render(const FrameInfo& frame) = 0;
    // Called before anything is destroyed, with the GPU already idle.
    virtual void on_shutdown() {}
    // Called after the swapchain has been rebuilt at a new size.
    virtual void on_resize(VkExtent2D /*extent*/) {}

    // Called for every SDL event, before App itself looks at it. Quit, Escape and
    // resize are still handled by App regardless of what a sample does here.
    virtual void on_event(const SDL_Event& /*event*/) {}

    [[nodiscard]] Context& context() noexcept { return *context_; }
    [[nodiscard]] Swapchain& swapchain() noexcept { return *swapchain_; }
    [[nodiscard]] Window& window() noexcept { return *window_; }

    // Seconds since the first frame, and since the previous one.
    [[nodiscard]] float elapsed() const noexcept { return elapsed_; }
    [[nodiscard]] float delta_time() const noexcept { return delta_time_; }

private:
    bool poll_events();
    void rebuild_swapchain();

    Options options_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Context> context_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<FrameContext> frames_;

    Capture capture_;

    bool swapchain_dirty_ = false;
    bool running_ = true;
    uint64_t frames_rendered_ = 0;
    float elapsed_ = 0.0F;
    float delta_time_ = 0.0F;
    uint64_t last_tick_ = 0;
};

}  // namespace vkc
