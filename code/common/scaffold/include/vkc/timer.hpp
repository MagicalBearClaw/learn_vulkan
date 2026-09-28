#pragma once

#include "vkc/frame.hpp"

#include <volk.h>

#include <array>
#include <cstdint>

namespace vkc {

class Context;

// How long a stretch of a command buffer took on the GPU, from a pair of timestamp
// queries per frame in flight.
//
// Chapter 4.6 wrote every line of this by hand: why the answer arrives a frame late,
// why each frame slot gets its own pair of queries, why the reset has to be recorded
// outside a rendering pass, and what timestampPeriod and timestampValidBits are for. From 4.7 it
// lives here, unchanged except for being split into a header and a source file.
class GpuTimer {
public:
    void create(Context& context);
    void destroy() noexcept;

    // What this slot measured the last time it was used, in milliseconds, or a negative
    // number if it has never been used. Must be called before reset() reuses the slot.
    [[nodiscard]] double read(uint32_t slot) const;

    // Outside a rendering pass, before begin().
    void reset(VkCommandBuffer cmd, uint32_t slot);
    void begin(VkCommandBuffer cmd, uint32_t slot);
    void end(VkCommandBuffer cmd, uint32_t slot);

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    float period_ns_ = 0.0F;
    uint64_t mask_ = 0;
    std::array<bool, kFramesInFlight> written_{};
};

}  // namespace vkc
