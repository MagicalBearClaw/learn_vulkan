// 1.10 Descriptors and push constants
//
// Adds to 1.9: the answer to "where did glUniform go".
//
// It went to two places, and which one you want depends entirely on how often the data
// changes and how big it is.
//
//   Push constants    A few dozen bytes, written straight into the command buffer.
//                     No buffer, no descriptor, no synchronisation. Per-draw data.
//
//   Uniform buffers   Ordinary GPU memory, reached through a descriptor set. Any size
//                     up to at least 16 KiB. Per-frame or per-material data.
//
// This chapter draws the same quad four times in one frame. What is shared between the
// four draws lives in a uniform buffer; what differs between them arrives as push
// constants.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
#include <vkc/pipeline.hpp>

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numbers>
#include <span>

namespace {

struct Vertex {
    glm::vec2 position;
    glm::vec3 colour;
};

constexpr std::array<Vertex, 4> kVertices{{
    {{-1.0F, -1.0F}, {1.0F, 0.0F, 0.0F}},
    {{1.0F, -1.0F}, {0.0F, 1.0F, 0.0F}},
    {{1.0F, 1.0F}, {0.0F, 0.0F, 1.0F}},
    {{-1.0F, 1.0F}, {1.0F, 1.0F, 0.0F}},
}};

constexpr std::array<uint16_t, 6> kIndices{0, 1, 2, 2, 3, 0};

// Data shared by every draw in the frame, in a uniform buffer.
//
// The padding is not decoration. A uniform buffer is laid out by the std140 rules,
// which round a struct's size up to a multiple of 16 bytes and align a four-component
// vector to 16, so this struct has to match what the shader expects or the values
// arrive shifted. Slang lays its ConstantBuffer<Globals> out by exactly those rules --
// the SPIR-V even names the type Globals_std140 -- and it will not warn you when the
// C++ struct disagrees. See the article: this is one of the most common and most
// baffling bugs in early Vulkan code.
struct Globals {
    glm::vec4 ambient;  // offset 0, 16 bytes
    float time;         // offset 16
    float padding[3];   // to 32, the next multiple of 16
};
static_assert(sizeof(Globals) == 32);

// Data that differs per draw, pushed straight into the command buffer.
//
// 128 bytes is the minimum guaranteed size of the whole push-constant block on any
// Vulkan implementation, and plenty of hardware offers no more. Treat it as the budget.
struct PushConstants {
    glm::vec2 offset;  // where this copy of the quad goes, in clip space
    float scale;       // how big
    float phase;       // where in the pulse it starts
};
static_assert(sizeof(PushConstants) == 16);

// The four quads. Only the push constants differ between them.
constexpr std::array<PushConstants, 4> kQuads{{
    {{-0.45F, -0.45F}, 0.35F, 0.0F},
    {{0.45F, -0.45F}, 0.35F, 1.57F},
    {{0.45F, 0.45F}, 0.35F, 3.14F},
    {{-0.45F, 0.45F}, 0.35F, 4.71F},
}};

class DescriptorApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        vertex_buffer_ = vkc::upload_to_device_local(
            context(), kVertices.data(), sizeof(kVertices),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer_ = vkc::upload_to_device_local(
            context(), kIndices.data(), sizeof(kIndices),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        create_uniform_buffers();
        create_descriptors();
        create_pipeline();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        // Sets allocated from the pool are freed with it; the layout is separate.
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        index_buffer_.destroy();
        vertex_buffer_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        // Which per-frame copy of the uniform buffer and descriptor set to use.
        //
        // One of each is not enough. The CPU is up to kFramesInFlight frames ahead of
        // the GPU, so writing into the buffer the GPU is still reading would corrupt
        // the frame already in flight. This is the same reasoning as 1.6's per-frame
        // command buffers, applied to data.
        const size_t slot = frame.frame_number % vkc::kFramesInFlight;

        update_globals(slot, frame.frame_number);

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

        // Bound once for the whole frame, because nothing in it changes between draws.
        // A descriptor set binding is relatively expensive; a push constant is not.
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &descriptor_sets_[slot], 0,
                                nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        for (const PushConstants& quad : kQuads) {
            // No buffer, no descriptor, no synchronisation: the bytes are copied into
            // the command buffer here and now, so each draw sees its own values with
            // nothing to keep alive afterwards.
            vkCmdPushConstants(frame.cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT |
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &quad);

            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0,
                             0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    void create_uniform_buffers() {
        for (vkc::Buffer& buffer : uniform_buffers_) {
            // Host-visible and permanently mapped. This is the opposite trade from the
            // vertex buffer: the data changes every frame, so a staging copy each time
            // would cost more than the slower reads do.
            buffer = vkc::Buffer(context().allocator(), sizeof(Globals),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    void create_descriptors() {
        const VkDevice device = context().device();

        // What the shader expects to find in set 0. This is a description only: no
        // memory, no buffer, nothing bound. It is the type of the set.
        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            // Both stages read it, so both are named. Naming only the stages that
            // actually use a binding lets some hardware skip work.
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };

        const VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                             &set_layout_));

        // Descriptor sets are not allocated individually; they come out of a pool, and
        // the pool is told up front how many of each descriptor type it must serve.
        // Run out and allocation fails -- it does not grow.
        const VkDescriptorPoolSize pool_size{
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = vkc::kFramesInFlight,
        };

        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const std::array<VkDescriptorSetLayout, vkc::kFramesInFlight> layouts{
            set_layout_, set_layout_};

        const VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = vkc::kFramesInFlight,
            .pSetLayouts = layouts.data(),
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets_.data()));

        // Now point each set at its buffer. A descriptor set starts out empty; writing
        // into it is what makes it refer to anything.
        for (size_t i = 0; i < vkc::kFramesInFlight; ++i) {
            const VkDescriptorBufferInfo buffer_info{
                .buffer = uniform_buffers_[i].handle(),
                .offset = 0,
                .range = sizeof(Globals),
            };

            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = descriptor_sets_[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &buffer_info,
                .pTexelBufferView = nullptr,
            };

            // Done once, at start-up. The set keeps pointing at the same buffer for the
            // life of the program, and only the buffer's *contents* change per frame.
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    void update_globals(size_t slot, uint64_t frame_number) {
        // Frame-counter driven rather than clock driven, so frame N always looks the
        // same and tools/capture.py can diff screenshots.
        const Globals globals{
            .ambient = glm::vec4(0.12F, 0.10F, 0.18F, 1.0F),
            .time = static_cast<float>(frame_number) / 30.0F,
            .padding = {},
        };

        // A plain memcpy into memory the GPU can see. No flush is needed because VMA
        // gave us coherent memory; on a device without it, vmaFlushAllocation would go
        // here.
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    void create_pipeline() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

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

        // The push constant range. Offset and size are in bytes, and the stages listed
        // here must cover every stage that reads the block.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        // The pipeline layout, finally doing something. Every chapter until now created
        // an empty one: it is the declaration of what the shaders can see.
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr,
                                        &pipeline_layout_));

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

        spdlog::info("maxPushConstantsSize on this GPU: {} bytes",
                     context().gpu_properties().limits.maxPushConstantsSize);
    }

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> descriptor_sets_{};

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.10 - Descriptors and push constants";
        DescriptorApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
