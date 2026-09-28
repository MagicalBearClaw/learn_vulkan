#include "vkc/timer.hpp"

#include "vkc/check.hpp"
#include "vkc/context.hpp"

#include <stdexcept>
#include <vector>

namespace vkc {

void GpuTimer::create(Context& context) {
    device_ = context.device();

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    period_ns_ = properties.limits.timestampPeriod;

    // A queue that cannot write timestamps reports zero valid bits.
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(),
                                             &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(),
                                             &family_count, families.data());
    const uint32_t valid_bits =
        families[context.queue_family()].timestampValidBits;
    if (valid_bits == 0) {
        throw std::runtime_error("This queue cannot write timestamps.");
    }
    mask_ = valid_bits == 64 ? ~uint64_t{0} : (uint64_t{1} << valid_bits) - 1;

    const VkQueryPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2 * kFramesInFlight,
        .pipelineStatistics = 0,
    };
    VK_CHECK(vkCreateQueryPool(device_, &pool_info, nullptr, &pool_));
}

void GpuTimer::destroy() noexcept { vkDestroyQueryPool(device_, pool_, nullptr); }

double GpuTimer::read(uint32_t slot) const {
    if (!written_[slot]) {
        return -1.0;
    }
    std::array<uint64_t, 2> ticks{};
    VK_CHECK(vkGetQueryPoolResults(device_, pool_, 2 * slot, 2, sizeof(ticks),
                                   ticks.data(), sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT));
    const uint64_t elapsed = (ticks[1] - ticks[0]) & mask_;
    return static_cast<double>(elapsed) * period_ns_ * 1e-6;
}

void GpuTimer::reset(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdResetQueryPool(cmd, pool_, 2 * slot, 2);
}

void GpuTimer::begin(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * slot);
}

void GpuTimer::end(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_,
                         2 * slot + 1);
    written_[slot] = true;
}

}  // namespace vkc
