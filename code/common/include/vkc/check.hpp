#pragma once

#include <volk.h>

#include <stdexcept>
#include <string>

namespace vkc {

// Almost every Vulkan entry point returns a VkResult, and almost every one of those
// is a code you are expected to check. Ignoring them is the single most common way
// a first Vulkan program ends up as a black window with no explanation, so the
// samples in this series check every one.
class VulkanError : public std::runtime_error {
public:
    VulkanError(VkResult result, std::string message)
        : std::runtime_error(std::move(message)), result_(result) {}

    [[nodiscard]] VkResult result() const noexcept { return result_; }

private:
    VkResult result_;
};

// Human-readable name for a VkResult, e.g. VK_ERROR_OUT_OF_DEVICE_MEMORY.
[[nodiscard]] const char* to_string(VkResult result) noexcept;

[[noreturn]] void throw_vulkan_error(VkResult result, const char* expression,
                                     const char* file, int line);

}  // namespace vkc

// VK_CHECK(vkCreateInstance(...)) — evaluates once, throws on anything but VK_SUCCESS.
#define VK_CHECK(expression)                                                        \
    do {                                                                            \
        const VkResult vkc_result_ = (expression);                                  \
        if (vkc_result_ != VK_SUCCESS) {                                            \
            ::vkc::throw_vulkan_error(vkc_result_, #expression, __FILE__, __LINE__); \
        }                                                                           \
    } while (false)
