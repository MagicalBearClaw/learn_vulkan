#pragma once

#include "vkc/context.hpp"
#include "vkc/frame.hpp"
#include "vkc/swapchain.hpp"
#include "vkc/window.hpp"

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
    struct Options {
        std::string title = "LearnVulkan";
        uint32_t width = 1280;
        uint32_t height = 720;
        bool validation = true;

        // Run this many frames then exit. Zero means "until the user closes the
        // window". tools/capture.py uses this to render deterministic screenshots.
        uint64_t frame_limit = 0;
        // Write the contents of the last rendered frame here as a PNG, then exit.
        std::filesystem::path screenshot;
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

    [[nodiscard]] Context& context() noexcept { return *context_; }
    [[nodiscard]] Swapchain& swapchain() noexcept { return *swapchain_; }
    [[nodiscard]] Window& window() noexcept { return *window_; }

    // Seconds since the first frame, and since the previous one.
    [[nodiscard]] float elapsed() const noexcept { return elapsed_; }
    [[nodiscard]] float delta_time() const noexcept { return delta_time_; }

private:
    bool poll_events();
    void rebuild_swapchain();

    // Screenshots are taken inside the frame, not after it. A swapchain image may
    // only be touched between vkAcquireNextImageKHR and vkQueuePresentKHR, so the
    // copy has to be recorded into the same command buffer that just drew it.
    void create_capture_buffer(VkExtent2D extent);
    void record_capture(const FrameInfo& frame);
    void write_capture();
    void destroy_capture_buffer() noexcept;

    Options options_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Context> context_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<FrameContext> frames_;

    VkBuffer capture_buffer_ = VK_NULL_HANDLE;
    VmaAllocation capture_allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo capture_allocation_info_{};
    VkExtent2D capture_extent_{};
    VkFormat capture_format_ = VK_FORMAT_UNDEFINED;
    bool capture_ready_ = false;

    bool swapchain_dirty_ = false;
    bool running_ = true;
    uint64_t frames_rendered_ = 0;
    float elapsed_ = 0.0F;
    float delta_time_ = 0.0F;
    uint64_t last_tick_ = 0;
};

}  // namespace vkc
