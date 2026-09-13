#include "vkc/app.hpp"

#include "vkc/check.hpp"
#include "vkc/paths.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <stb_image_write.h>

#include <charconv>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace vkc {
namespace {

uint32_t parse_uint(std::string_view text, const char* flag) {
    uint32_t value = 0;
    const auto [ptr, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || ptr != text.data() + text.size()) {
        throw std::runtime_error(
            std::format("{} expects a number, got '{}'", flag, text));
    }
    return value;
}

}  // namespace

App::Options App::parse_args(int argc, char** argv, Options defaults) {
    const std::span<char*> args(argv, static_cast<size_t>(argc));

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string_view flag = args[i];
        const auto next = [&](const char* name) -> std::string_view {
            if (i + 1 >= args.size()) {
                throw std::runtime_error(std::format("{} needs a value", name));
            }
            return args[++i];
        };

        if (flag == "--frames") {
            defaults.frame_limit = parse_uint(next("--frames"), "--frames");
        } else if (flag == "--screenshot") {
            defaults.screenshot = next("--screenshot");
            // A screenshot run that never stops would never write the file.
            if (defaults.frame_limit == 0) {
                defaults.frame_limit = 1;
            }
        } else if (flag == "--width") {
            defaults.width = parse_uint(next("--width"), "--width");
        } else if (flag == "--height") {
            defaults.height = parse_uint(next("--height"), "--height");
        } else if (flag == "--no-validation") {
            defaults.validation = false;
        } else if (flag == "--help" || flag == "-h") {
            spdlog::info(
                "options: --frames N  --screenshot PATH  --width N  --height N  "
                "--no-validation");
            std::exit(0);
        } else {
            throw std::runtime_error(std::format("unknown option '{}'", flag));
        }
    }
    return defaults;
}

App::App(Options options) : options_(std::move(options)) {
    window_ = std::make_unique<Window>(options_.title, options_.width, options_.height);

    const Context::Config config{
        .app_name = options_.title,
        .enable_validation = options_.validation,
    };
    context_ = std::make_unique<Context>(config, *window_);
    swapchain_ = std::make_unique<Swapchain>(*context_, *window_);
    frames_ = std::make_unique<FrameContext>(*context_, *swapchain_);
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
            record_capture(*frame);
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

    if (capture_ready_) {
        write_capture();
    }
    destroy_capture_buffer();

    on_shutdown();
    return 0;
}

bool App::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
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

void App::create_capture_buffer(VkExtent2D extent) {
    destroy_capture_buffer();

    capture_extent_ = extent;
    capture_format_ = swapchain_->format();
    const VkDeviceSize byte_count =
        static_cast<VkDeviceSize>(extent.width) * extent.height * 4;

    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = byte_count,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    // HOST_ACCESS_RANDOM plus MAPPED gives a buffer the CPU can read straight out
    // of once the GPU is done writing it, with no second copy.
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VK_CHECK(vmaCreateBuffer(context_->allocator(), &buffer_info, &alloc_info,
                             &capture_buffer_, &capture_allocation_,
                             &capture_allocation_info_));
    context_->name(capture_buffer_, VK_OBJECT_TYPE_BUFFER, "screenshot staging");
}

void App::record_capture(const FrameInfo& frame) {
    if (capture_buffer_ == VK_NULL_HANDLE ||
        capture_extent_.width != frame.extent.width ||
        capture_extent_.height != frame.extent.height) {
        create_capture_buffer(frame.extent);
    }

    transition_image(frame.cmd, frame.image,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const VkBufferImageCopy2 region{
        .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext = nullptr,
        .bufferOffset = 0,
        .bufferRowLength = 0,    // 0 means "tightly packed to imageExtent"
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {frame.extent.width, frame.extent.height, 1},
    };
    const VkCopyImageToBufferInfo2 copy{
        .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .pNext = nullptr,
        .srcImage = frame.image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstBuffer = capture_buffer_,
        .regionCount = 1,
        .pRegions = &region,
    };
    vkCmdCopyImageToBuffer2(frame.cmd, &copy);

    // Still has to reach PRESENT_SRC_KHR: the image is presented as normal, the
    // screenshot is just a copy taken on the way past.
    transition_image(frame.cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    capture_ready_ = true;
}

void App::write_capture() {
    const size_t byte_count =
        static_cast<size_t>(capture_extent_.width) * capture_extent_.height * 4;

    std::vector<uint8_t> pixels(byte_count);
    std::memcpy(pixels.data(), capture_allocation_info_.pMappedData, byte_count);

    // Swapchains usually hand out BGRA; PNG wants RGBA, so swap red and blue.
    const bool is_bgra = capture_format_ == VK_FORMAT_B8G8R8A8_SRGB ||
                         capture_format_ == VK_FORMAT_B8G8R8A8_UNORM;
    if (is_bgra) {
        for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
            std::swap(pixels[i], pixels[i + 2]);
        }
    }

    const std::string path = options_.screenshot.string();
    if (stbi_write_png(path.c_str(), static_cast<int>(capture_extent_.width),
                       static_cast<int>(capture_extent_.height), 4, pixels.data(),
                       static_cast<int>(capture_extent_.width) * 4) == 0) {
        spdlog::error("Could not write screenshot to {}", path);
    } else {
        spdlog::info("Wrote {} ({}x{})", path, capture_extent_.width,
                     capture_extent_.height);
    }
}

void App::destroy_capture_buffer() noexcept {
    if (capture_buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context_->allocator(), capture_buffer_, capture_allocation_);
        capture_buffer_ = VK_NULL_HANDLE;
        capture_allocation_ = VK_NULL_HANDLE;
    }
    capture_ready_ = false;
}

}  // namespace vkc
