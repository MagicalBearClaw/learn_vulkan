// 1.12 Transformations
//
// Adds to 1.11: matrices. The quad stops being nailed to the coordinates in its vertex
// buffer and starts being placed, turned and scaled by a matrix computed per draw.
//
// There is very little new Vulkan here -- a mat4 in a push constant and nothing else.
// Chapter 1.11's texture loading has moved into vkcommon as vkc::load_texture and
// vkc::Sampler, which is why this file is half the length of the last one.
// The chapter is about the mathematics, because everything from here to the end of the
// series is built on it: a camera is a matrix, a bone in a skeleton is a matrix, a
// shadow map's light is a matrix.
//
// The one thing worth watching for is *order*. Matrix multiplication does not commute,
// and the two quads on screen differ only in the order two matrices were multiplied.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <span>

namespace {

struct Vertex {
    glm::vec2 position;
    glm::vec2 uv;
};

// A unit square centred on the origin. Its own coordinates never change again: from
// here on, where it appears is entirely the matrix's business.
constexpr std::array<Vertex, 4> kVertices{{
    {{-0.5F, -0.5F}, {0.0F, 0.0F}},
    {{0.5F, -0.5F}, {1.0F, 0.0F}},
    {{0.5F, 0.5F}, {1.0F, 1.0F}},
    {{-0.5F, 0.5F}, {0.0F, 1.0F}},
}};

constexpr std::array<uint16_t, 6> kIndices{0, 1, 2, 2, 3, 0};

// 64 bytes of the 128-byte guaranteed push constant budget. A model matrix in push
// constants is the standard arrangement, and it is the reason to be careful about what
// else goes in there.
struct PushConstants {
    glm::mat4 model;
};
static_assert(sizeof(PushConstants) == 64);

class TransformApp : public vkc::App {
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

        // Chapter 1.11's texture loading, now in vkcommon: the same staging copy, the
        // same vkCmdBlitImage mip ladder, the same final SHADER_READ_ONLY_OPTIMAL.
        // The sampler stays a separate object, because in Vulkan it is one.
        texture_ = vkc::load_texture(context(), vkc::asset_path("textures/lvk_grid.png"));
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{
                                               .address_mode =
                                                   VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                           });
        create_descriptors();
        create_pipeline();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);

        sampler_.destroy();
        texture_.destroy();
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

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        // Frame-counter driven so frame N always looks the same.
        const float angle = static_cast<float>(frame.frame_number) * 0.015F;

        // Clip space is stretched to whatever shape the window is, so a square drawn
        // at equal x and y extents comes out as a rectangle and a rotation comes out
        // sheared. Squeezing x by the aspect ratio cancels it.
        //
        // This is a stopgap, and naming it honestly: it is a two-by-two orthographic
        // projection wearing a disguise. Chapter 1.13 replaces it with a real
        // projection matrix, which is the same idea done properly and in three
        // dimensions.
        const float aspect = static_cast<float>(frame.extent.width) /
                             static_cast<float>(frame.extent.height);
        const glm::mat4 aspect_fix =
            glm::scale(glm::mat4(1.0F), glm::vec3(1.0F / aspect, 1.0F, 1.0F));

        // Read every chain below right to left: the rightmost matrix is applied to the
        // vertex first. That is a consequence of column-vector convention -- the vertex
        // is a column on the right of the product, so it meets the nearest matrix
        // first.

        // Left: rotate, then translate. The quad spins about its own centre, and the
        // translation carries the already-rotated quad sideways. Spin in place.
        const glm::mat4 spin_in_place =
            aspect_fix *
            glm::translate(glm::mat4(1.0F), glm::vec3(-0.5F, 0.0F, 0.0F)) *
            glm::rotate(glm::mat4(1.0F), angle, glm::vec3(0.0F, 0.0F, 1.0F)) *
            glm::scale(glm::mat4(1.0F), glm::vec3(0.6F, 0.6F, 1.0F));

        // Right: translate, then rotate. The translation moves the quad off the origin
        // first, and the rotation then turns the whole thing about the origin, taking
        // the displaced quad with it. Orbit.
        const glm::mat4 orbit =
            aspect_fix *
            glm::rotate(glm::mat4(1.0F), angle, glm::vec3(0.0F, 0.0F, 1.0F)) *
            glm::translate(glm::mat4(1.0F), glm::vec3(0.5F, 0.0F, 0.0F)) *
            glm::scale(glm::mat4(1.0F), glm::vec3(0.35F, 0.35F, 1.0F));

        for (const glm::mat4& model : {spin_in_place, orbit}) {
            const PushConstants push{.model = model};
            vkCmdPushConstants(frame.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(push), &push);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0,
                             0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    void create_descriptors() {
        const VkDevice device = context().device();

        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
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

        const VkDescriptorPoolSize pool_size{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
        };

        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout_,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_));

        const VkDescriptorImageInfo image_info{
            .sampler = sampler_.handle(),
            .imageView = texture_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptor_set_,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
        }};

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

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

        const VkShaderModule vertex_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "transform.vert");
        const VkShaderModule fragment_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "transform.frag");

        pipeline_ = vkc::PipelineBuilder(device)
                        .shaders(vertex_shader, fragment_shader)
                        .vertex_input(std::span(&binding, 1), attributes)
                        .colour_attachment(swapchain().format())
                        .layout(pipeline_layout_)
                        .build();

        vkDestroyShaderModule(device, fragment_shader, nullptr);
        vkDestroyShaderModule(device, vertex_shader, nullptr);
    }

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    vkc::Image texture_;
    vkc::Sampler sampler_;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.12 - Transformations";
        TransformApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
