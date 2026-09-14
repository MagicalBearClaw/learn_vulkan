// 1.9 Shaders
//
// Adds to 1.8: nothing on the C++ side that is new Vulkan. The buffers this chapter
// needs are the ones 1.8 wrote, now living in vkcommon as vkc::Buffer and
// vkc::upload_to_device_local, so main.cpp is half the length it was.
//
// The subject this time is the shading language itself -- Slang, and the SPIR-V it
// compiles to -- plus the one piece of Vulkan that goes with it: specialisation
// constants, which let the same SPIR-V module be compiled into different pipelines
// with different constant values baked in.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
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

struct Vertex {
    glm::vec2 position;
    glm::vec3 colour;
};

constexpr std::array<Vertex, 4> kVertices{{
    {{-0.6F, -0.6F}, {1.0F, 0.0F, 0.0F}},
    {{0.6F, -0.6F}, {0.0F, 1.0F, 0.0F}},
    {{0.6F, 0.6F}, {0.0F, 0.0F, 1.0F}},
    {{-0.6F, 0.6F}, {1.0F, 1.0F, 0.0F}},
}};

constexpr std::array<uint16_t, 6> kIndices{0, 1, 2, 2, 3, 0};

// The values baked into the fragment shader when the pipeline is built.
//
// This is a plain struct whose layout we control, because VkSpecializationInfo takes a
// flat block of bytes plus a table saying which constant id lives at which offset.
struct ShaderConstants {
    // Matches `[[vk::constant_id(0)]] const int kShadingMode` in quad.slang.
    int32_t shading_mode = 0;  // 0 = smooth, 1 = flat
    // Matches `[[vk::constant_id(1)]] const float kCheckerSize`.
    float checker_size = 48.0F;
};

class ShaderApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        // Two lines, where 1.8 needed a hundred. Same staging upload, same
        // device-local memory; see code/common/scaffold/src/buffer.cpp.
        vertex_buffer_ = vkc::upload_to_device_local(
            context(), kVertices.data(), sizeof(kVertices),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer_ = vkc::upload_to_device_local(
            context(), kIndices.data(), sizeof(kIndices),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        create_pipeline();
    }

    void on_shutdown() override {
        vkDestroyPipeline(context().device(), pipeline_, nullptr);
        vkDestroyPipelineLayout(context().device(), pipeline_layout_, nullptr);

        // vkc::Buffer frees itself, but only when it is destroyed -- and these members
        // outlive the device unless they are told to let go now.
        index_buffer_.destroy();
        vertex_buffer_.destroy();
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

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);

        vkCmdEndRendering(frame.cmd);
    }

private:
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

        // The specialisation table. Each entry says: constant id N is `size` bytes at
        // `offset` in the data block. The ids match `constant_id` in the shader, and
        // the offsets match the struct above.
        //
        // This is the only place in Vulkan where a shader's constants are supplied by
        // the host without a buffer and without a descriptor, because the values are
        // consumed by the compiler rather than read at run time.
        const std::array<VkSpecializationMapEntry, 2> entries{{
            {
                .constantID = 0,
                .offset = offsetof(ShaderConstants, shading_mode),
                .size = sizeof(ShaderConstants::shading_mode),
            },
            {
                .constantID = 1,
                .offset = offsetof(ShaderConstants, checker_size),
                .size = sizeof(ShaderConstants::checker_size),
            },
        }};

        const ShaderConstants constants{};

        // One module, both stages: quad.slang compiles to a single SPIR-V
        // binary with a vertexMain and a fragmentMain entry point in it.
        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "quad.slang");

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

        pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(shader)
                .vertex_input(std::span(&binding, 1), attributes)
                .fragment_specialisation(
                    entries, std::span(reinterpret_cast<const std::byte*>(&constants),
                                       sizeof(constants)))
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_)
                .build();

        vkDestroyShaderModule(device, shader, nullptr);

        spdlog::info("Pipeline built with shading_mode={} checker_size={}",
                     constants.shading_mode, constants.checker_size);
    }

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.9 - Shaders";
        ShaderApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
