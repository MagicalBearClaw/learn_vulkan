// Smoke test: opens a window and cycles the clear colour.
//
// This is not one of the tutorial chapters. It exists so that a fresh clone can
// prove the whole toolchain works -- compiler, vcpkg, Vulkan loader, driver,
// validation layers, swapchain, present -- before chapter 1 asks you to learn
// anything. Chapter 1.6 builds the same thing from nothing, explaining every call.

#include <vkc/app.hpp>

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <numbers>

namespace {

class ClearColourApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        spdlog::info("Smoke test running. Press Escape or close the window to quit.");
    }

    void on_render(const vkc::FrameInfo& frame) override {
        // Two sine waves a third of a cycle apart give a slow, obvious colour
        // cycle -- obvious enough that a frozen frame is easy to spot.
        constexpr float kTwoThirdsPi = 2.0F * std::numbers::pi_v<float> / 3.0F;
        // Driven by the frame counter rather than the clock, so frame N always has
        // the same colour and tools/capture.py can diff screenshots meaningfully.
        const float t = static_cast<float>(frame.frame_number) / 50.0F;

        const VkClearValue clear{
            .color = {{0.5F + 0.5F * std::sin(t),
                       0.5F + 0.5F * std::sin(t + kTwoThirdsPi),
                       0.5F + 0.5F * std::sin(t + 2.0F * kTwoThirdsPi), 1.0F}},
        };

        // Vulkan 1.3 dynamic rendering: describe the attachment here and now,
        // instead of baking a VkRenderPass and VkFramebuffer up front. LOAD_OP_CLEAR
        // means the clear is free -- it happens as part of starting the pass rather
        // than as a separate write over the whole image.
        const VkRenderingAttachmentInfo colour_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = frame.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clear,
        };

        const VkRenderingInfo rendering{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {{0, 0}, frame.extent},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour_attachment,
            .pDepthAttachment = nullptr,
            .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(frame.cmd, &rendering);
        // Nothing is drawn yet. The triangle arrives in chapter 1.8.
        vkCmdEndRendering(frame.cmd);
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan - smoke test";
        ClearColourApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
