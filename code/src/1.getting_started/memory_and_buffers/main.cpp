// 1.8 Memory and buffers
//
// Adds to 1.7: real geometry. The vertices stop living inside the shader and move into
// GPU memory, where every mesh you will ever draw belongs.
//
// This is the first chapter that links vkcommon. Everything chapters 1.1 to 1.7 built
// by hand -- window, instance, device, swapchain, command buffers, synchronisation,
// the pipeline builder -- now lives in code/common/scaffold/ and is used rather than
// retyped. Nothing new is hidden there.
//
// The new idea is that a VkBuffer is not memory. It is a description of how some
// memory will be used, and it starts out with none attached. Allocating the memory,
// choosing which heap it comes from, and getting data into it are three separate
// steps, and Vulkan makes you take all three.

#include <vkc/app.hpp>
#include <vkc/check.hpp>
#include <vkc/pipeline.hpp>

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>

namespace {

// One vertex of the quad. Position in clip space, colour to interpolate.
//
// This is a plain C++ struct with no Vulkan in it at all. What makes it a vertex is
// the binding and attribute descriptions below, which tell the pipeline how to walk
// over an array of these and where each field is.
struct Vertex {
    glm::vec2 position;
    glm::vec3 colour;
};

// Four corners, drawn as two triangles.
//
// Remember y points down in Vulkan clip space, so -0.5 is the top of the screen.
constexpr std::array<Vertex, 4> kVertices{{
    {{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},  // top left,     red
    {{0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},   // top right,    green
    {{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},    // bottom right, blue
    {{-0.5F, 0.5F}, {1.0F, 1.0F, 0.0F}},   // bottom left,  yellow
}};

// Two triangles sharing an edge. Without an index buffer this would be six vertices
// with two of them duplicated; with one it is four vertices and six small integers.
//
// On a real mesh the saving is large. A cube has 8 corners and 36 triangle vertices,
// and a typical model reuses each vertex six times or more. The GPU also caches
// recently transformed vertices by index, so a shared vertex may only run the vertex
// shader once.
constexpr std::array<uint16_t, 6> kIndices{0, 1, 2, 2, 3, 0};

// A buffer and the allocation backing it.
//
// VMA hands back two things from one call: the VkBuffer, and a VmaAllocation, which is
// its record of where in which VkDeviceMemory block the buffer actually lives. Both
// have to be kept, and both have to be freed together.
struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
};

class QuadApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        create_geometry();
        create_pipeline();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        destroy_buffer(index_buffer_);
        destroy_buffer(vertex_buffer_);
    }

    void on_render(const vkc::FrameInfo& frame) override {
        const VkClearValue clear{.color = {{0.02F, 0.02F, 0.04F, 1.0F}}};

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

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        const VkViewport viewport{
            .x = 0.0F,
            .y = 0.0F,
            .width = static_cast<float>(frame.extent.width),
            .height = static_cast<float>(frame.extent.height),
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        const VkRect2D scissor{.offset = {0, 0}, .extent = frame.extent};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        // Binding a vertex buffer says "binding 0 reads from this buffer, starting at
        // this byte offset". The pipeline already knows the stride and the attribute
        // layout; this only supplies the memory.
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer_.handle, &offset);

        // Index type is a property of the bind, not of the pipeline. uint16 halves the
        // bandwidth of uint32 and is enough for any mesh under 65536 vertices.
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle, 0, VK_INDEX_TYPE_UINT16);

        // Six indices rather than six vertices. The vertex shader still runs per
        // vertex, but which vertex it runs on comes from the index buffer.
        vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Buffers
    // ---------------------------------------------------------------------------

    // Creates a buffer and the memory behind it in one call.
    //
    // Done by hand this is four steps: vkCreateBuffer, vkGetBufferMemoryRequirements,
    // find a memory type whose properties match what you asked for, vkAllocateMemory,
    // vkBindBufferMemory. And you would not want one VkDeviceMemory per buffer -- real
    // drivers cap the number of allocations at a few thousand, so you have to
    // sub-allocate out of large blocks yourself.
    //
    // That is the whole reason VMA exists, and why this series uses it from the start.
    // vmaCreateBuffer does all of it, and suballocates from a block it already owns.
    [[nodiscard]] Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VmaAllocationCreateFlags flags,
                                       VmaMemoryUsage memory_usage) {
        const VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            // Usage is a promise. The driver picks memory and alignment to suit it,
            // and the validation layers hold you to it: binding a buffer as a vertex
            // buffer without VK_BUFFER_USAGE_VERTEX_BUFFER_BIT is an error.
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };

        const VmaAllocationCreateInfo alloc_info{
            .flags = flags,
            .usage = memory_usage,
            .requiredFlags = 0,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.0F,
        };

        Buffer buffer{};
        VK_CHECK(vmaCreateBuffer(context().allocator(), &buffer_info, &alloc_info,
                                 &buffer.handle, &buffer.allocation, &buffer.info));
        return buffer;
    }

    void destroy_buffer(Buffer& buffer) noexcept {
        if (buffer.handle != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context().allocator(), buffer.handle, buffer.allocation);
            buffer = {};
        }
    }

    // Uploads CPU data into a device-local buffer via a staging buffer.
    //
    // The short version of why this is two buffers rather than one: the memory the CPU
    // can write to and the memory the GPU reads fastest from are usually not the same
    // memory. Device-local memory lives on the card and is not mapped into the CPU's
    // address space at all on a discrete GPU. So we write into a buffer the CPU can
    // reach, then ask the GPU to copy it across the bus into memory it likes.
    //
    // For data that changes every frame the copy is not worth it and a permanently
    // mapped host-visible buffer wins. Vertex data that never changes is the opposite
    // case, and this is the right trade.
    [[nodiscard]] Buffer upload_to_device_local(const void* data, VkDeviceSize size,
                                                VkBufferUsageFlags usage) {
        // TRANSFER_SRC: this buffer is only ever the source of a copy.
        // HOST_ACCESS_SEQUENTIAL_WRITE tells VMA we will write it once, front to back,
        // and never read it -- which lets it pick write-combined memory.
        const Buffer staging =
            create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                              VMA_ALLOCATION_CREATE_MAPPED_BIT,
                          VMA_MEMORY_USAGE_AUTO);

        // MAPPED_BIT above means VMA kept it mapped and handed us the pointer, so
        // there is no vkMapMemory/vkUnmapMemory pair to get wrong.
        std::memcpy(staging.info.pMappedData, data, static_cast<size_t>(size));

        // TRANSFER_DST plus whatever the buffer is actually for. Without
        // TRANSFER_DST the copy below is invalid.
        const Buffer result = create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            0, VMA_MEMORY_USAGE_AUTO);

        immediate_submit([&](VkCommandBuffer cmd) {
            const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size};
            vkCmdCopyBuffer(cmd, staging.handle, result.handle, 1, &region);
        });

        // immediate_submit waited for the GPU, so the staging buffer has done its job
        // and can go. Destroying it any earlier would free memory the copy is reading.
        vmaDestroyBuffer(context().allocator(), staging.handle, staging.allocation);

        return result;
    }

    // Runs one-off GPU work and waits for it to finish.
    //
    // Uploads happen at load time, outside the frame loop, so there is no pipelining
    // to preserve and a plain fence wait is the honest thing to do. The pool is
    // TRANSIENT because these command buffers are recorded once and thrown away, which
    // lets the driver allocate them differently.
    //
    // A real engine batches its uploads and overlaps them with the first frames.
    // That is a later chapter; correctness first.
    template <typename Recorder>
    void immediate_submit(Recorder&& record) {
        const VkDevice device = context().device();

        const VkCommandPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = context().queue_family(),
        };
        VkCommandPool pool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr, &pool));

        const VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &cmd));

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));

        record(cmd);

        VK_CHECK(vkEndCommandBuffer(cmd));

        const VkCommandBufferSubmitInfo cmd_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0,
        };

        const VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = 0,
            .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_info,
            .signalSemaphoreInfoCount = 0,
            .pSignalSemaphoreInfos = nullptr,
        };

        const VkFenceCreateInfo fence_info{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        VkFence fence = VK_NULL_HANDLE;
        VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &fence));

        VK_CHECK(vkQueueSubmit2(context().queue(), 1, &submit, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
    }

    void create_geometry() {
        vertex_buffer_ =
            upload_to_device_local(kVertices.data(), sizeof(kVertices),
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer_ = upload_to_device_local(kIndices.data(), sizeof(kIndices),
                                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        context().name(vertex_buffer_.handle, VK_OBJECT_TYPE_BUFFER, "quad vertices");
        context().name(index_buffer_.handle, VK_OBJECT_TYPE_BUFFER, "quad indices");

        spdlog::info("Uploaded {} vertices ({} bytes) and {} indices ({} bytes).",
                     kVertices.size(), sizeof(kVertices), kIndices.size(),
                     sizeof(kIndices));
    }

    // ---------------------------------------------------------------------------
    // Pipeline
    // ---------------------------------------------------------------------------

    void create_pipeline() {
        const VkDevice device = context().device();

        // How to step through the vertex buffer. One binding, one Vertex per step.
        //
        // VERTEX means "advance once per vertex". The other option, INSTANCE, advances
        // once per instance instead, and is how per-instance data gets in. That is
        // chapter 4.8's subject.
        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        // Where each shader input lives inside one Vertex.
        //
        // `location` matches `[[vk::location(N)]]` in the shader's VSInput; `format`
        // describes the data as if it were an image format, which reads oddly at first
        // but is exactly the same vocabulary: R32G32_SFLOAT is two 32-bit floats.
        // `offset` is where the field starts, and offsetof is the right way to get it
        // rather than adding up sizes by hand.
        const std::array<VkVertexInputAttributeDescription, 2> attributes{{
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, colour),
            },
        }};

        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr,
                                        &pipeline_layout_));

        // vkc::PipelineBuilder is chapter 1.7's pipeline creation, unchanged, with the
        // parts a chapter needs to vary turned into methods. Read it at
        // code/common/scaffold/src/pipeline.cpp.
        // One module, both stages: quad.slang compiles to a single SPIR-V
        // binary with a vertexMain and a fragmentMain entry point in it.
        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "quad.slang");

        pipeline_ = vkc::PipelineBuilder(device)
                        .shaders(shader)
                        .vertex_input(std::span(&binding, 1), attributes)
                        .colour_attachment(swapchain().format())
                        .layout(pipeline_layout_)
                        .build();

        vkDestroyShaderModule(device, shader, nullptr);
    }

    Buffer vertex_buffer_{};
    Buffer index_buffer_{};

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.8 - Memory and buffers";
        QuadApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
