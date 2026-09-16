// 2.1 Colours
//
// The first chapter of Part 2, and the first that is about how something *looks* rather
// than about machinery.
//
// Adds to 1.14: a second pipeline, and the idea that an object's colour and a light's
// colour combine by multiplication. Removes rather more than it adds -- the texture and
// the sampler are gone, and the hand-written camera from 1.14 now lives in vkcommon as
// vkc::Camera, so the whole of the input handling is two lines.
//
// There is no lighting model here yet. Every face of the lit cube comes out the same
// flat shade, because nothing in this chapter knows which way a surface faces. Chapter
// 2.2 adds normals and the scene stops looking like a silhouette.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/pipeline.hpp>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>

namespace {

// Position only.
//
// 1.14's vertex carried a texture coordinate as well; there is no texture in this
// chapter, so it goes. The normal that 2.2 needs is not here yet either -- adding it is
// most of what 2.2 does to this file.
struct Vertex {
    glm::vec3 position;
};

// The same unit cube as 1.13 and 1.14, minus the texture coordinates.
//
// Still 24 vertices rather than 8. With no uv and no normal a cube really could be 8
// vertices and 36 indices, but it goes back to 24 in the very next chapter when each
// face needs its own normal, and keeping the layout stable makes the diff between the
// two chapters readable.
constexpr float kH = 0.5F;
constexpr std::array<Vertex, 24> kVertices{{
    // front (+z)
    {{-kH, -kH, kH}}, {{kH, -kH, kH}}, {{kH, kH, kH}}, {{-kH, kH, kH}},
    // back (-z)
    {{kH, -kH, -kH}}, {{-kH, -kH, -kH}}, {{-kH, kH, -kH}}, {{kH, kH, -kH}},
    // left (-x)
    {{-kH, -kH, -kH}}, {{-kH, -kH, kH}}, {{-kH, kH, kH}}, {{-kH, kH, -kH}},
    // right (+x)
    {{kH, -kH, kH}}, {{kH, -kH, -kH}}, {{kH, kH, -kH}}, {{kH, kH, kH}},
    // top (+y)
    {{-kH, kH, kH}}, {{kH, kH, kH}}, {{kH, kH, -kH}}, {{-kH, kH, -kH}},
    // bottom (-y)
    {{-kH, -kH, -kH}}, {{kH, -kH, -kH}}, {{kH, -kH, kH}}, {{-kH, -kH, kH}},
}};

constexpr std::array<uint16_t, 36> kIndices{
    0,  1,  2,  2,  3,  0,   // front
    4,  5,  6,  6,  7,  4,   // back
    8,  9,  10, 10, 11, 8,   // left
    12, 13, 14, 14, 15, 12,  // right
    16, 17, 18, 18, 19, 16,  // top
    20, 21, 22, 22, 23, 20,  // bottom
};

// Set 0, binding 0. Matches `Globals` in colours.slang exactly.
//
// std140 again: two matrices at 64 bytes each, then a vec4 that is already 16-byte
// aligned at offset 128. Nothing needs padding here, which is what ordering members
// largest-first buys you.
struct Globals {
    glm::mat4 view;        // offset 0
    glm::mat4 projection;  // offset 64
    glm::vec4 light_colour;  // offset 128
};
static_assert(sizeof(Globals) == 144);

// Per-object, pushed rather than bound. 80 of the 128 guaranteed bytes.
struct PushConstants {
    glm::mat4 model;
    glm::vec4 object_colour;
};
static_assert(sizeof(PushConstants) == 80);

// Coral, the colour LearnOpenGL uses for this scene, and white for the light.
constexpr glm::vec4 kObjectColour{1.0F, 0.5F, 0.31F, 1.0F};
constexpr glm::vec4 kLightColour{1.0F, 1.0F, 1.0F, 1.0F};

// Where the lamp orbits. Driven by the frame counter rather than the wall clock so that
// tools/capture.py gets the same image every run.
constexpr float kLightOrbitRadius = 1.8F;
constexpr float kLightHeight = 1.2F;

class ColoursApp : public vkc::App {
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

        depth_format_ = vkc::choose_depth_format(context().physical_device());
        depth_ = vkc::create_depth_buffer(context(), depth_format_,
                                          swapchain().extent());

        create_uniform_buffers();
        create_descriptors();
        create_pipelines();

        // Straight back from the origin, looking down -z, so the lit cube is centred
        // and the lamp's orbit passes above and around it.
        camera_.position = {0.0F, 0.5F, 4.0F};

        spdlog::info("Right-click to capture the mouse, WASD to fly, Escape to quit.");
    }

    void on_resize(VkExtent2D extent) override {
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
    }

    // The whole of 1.14's on_event and move_camera, now that vkc::Camera owns both.
    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, lamp_pipeline_, nullptr);
        vkDestroyPipeline(device, object_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        depth_.destroy();
        index_buffer_.destroy();
        vertex_buffer_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const float time = static_cast<float>(frame.frame_number) / 60.0F;
        const glm::vec3 light_position = orbit_position(time);

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.02F, 0.02F, 0.04F, 1.0F}});

        set_viewport(frame);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &descriptor_sets_[slot], 0,
                                nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        // The lit cube. Sits at the origin and turns slowly, so that the flatness of
        // the shading is unmistakable: the silhouette changes and the colour does not.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object_pipeline_);
        draw_cube(frame.cmd,
                  glm::rotate(glm::mat4(1.0F), time * 0.3F,
                              glm::normalize(glm::vec3(0.3F, 1.0F, 0.15F))),
                  kObjectColour);

        // The lamp. Same geometry, same vertex stage, different fragment entry point
        // and therefore a different pipeline. Scaled right down so it reads as a marker
        // rather than as a second object in the scene.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        draw_cube(frame.cmd,
                  glm::scale(glm::translate(glm::mat4(1.0F), light_position),
                             glm::vec3(0.2F)),
                  kLightColour);

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

    // A circle in the xz plane, lifted to a constant height. 2.2 uses this same
    // position to work out a light direction, which is when it starts to matter.
    [[nodiscard]] static glm::vec3 orbit_position(float time) {
        return {kLightOrbitRadius * std::sin(time * 0.35F), kLightHeight,
                kLightOrbitRadius * std::cos(time * 0.35F)};
    }

    void draw_cube(VkCommandBuffer cmd, const glm::mat4& model,
                   const glm::vec4& colour) {
        const PushConstants push{.model = model, .object_colour = colour};
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing. choose_depth_format, create_depth_buffer and begin_rendering
    // now live in vkcommon -- 1.13 wrote all three out in full -- which leaves the
    // viewport and the projection matrix here.
    // ---------------------------------------------------------------------------

    static void set_viewport(const vkc::FrameInfo& frame) {
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
    }

    [[nodiscard]] static glm::mat4 build_projection(VkExtent2D extent) {
        const float aspect =
            static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 projection =
            glm::perspective(glm::radians(45.0F), aspect, 0.1F, 100.0F);
        projection[1][1] *= -1.0F;  // Vulkan's clip-space y points down.
        return projection;
    }

    void create_uniform_buffers() {
        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer = vkc::Buffer(context().allocator(), sizeof(Globals),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    void update_globals(size_t slot, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = build_projection(extent),
            .light_colour = kLightColour,
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors and pipelines
    // ---------------------------------------------------------------------------

    void create_descriptors() {
        const VkDevice device = context().device();

        // One binding: the per-frame block. The texture 1.14 had at binding 1 is gone.
        //
        // The stage flags now name both stages, because the fragment shader reads
        // light_colour out of the same block the vertex shader reads the matrices from.
        // Getting this wrong is a validation error rather than a silent one, which is a
        // mercy.
        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
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
        VK_CHECK(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout_));

        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                             vkc::kFramesInFlight};

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
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    // Two pipelines, one shader module, one pipeline layout.
    //
    // Everything about these two is identical except the fragment entry point, which is
    // the cheapest possible illustration of what a VkPipeline actually is: a compiled
    // combination of state. Two combinations, two objects, chosen with one
    // vkCmdBindPipeline each.
    void create_pipelines() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        const VkVertexInputAttributeDescription attribute{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        };

        // Visible to both stages: the vertex shader reads `model`, the fragment shader
        // reads `object_colour`, and a push constant range covers whole stages rather
        // than individual members.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
        VK_CHECK(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "colours.slang");

        const auto build = [&](const char* fragment_entry) {
            return vkc::PipelineBuilder(device)
                .shaders(shader, "vertexMain", fragment_entry)
                .vertex_input(std::span(&binding, 1), std::span(&attribute, 1))
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true, VK_COMPARE_OP_LESS)
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_)
                .build();
        };

        object_pipeline_ = build("fragmentMain");
        lamp_pipeline_ = build("lampFragmentMain");

        vkDestroyShaderModule(device, shader, nullptr);

        context().name(object_pipeline_, VK_OBJECT_TYPE_PIPELINE, "object pipeline");
        context().name(lamp_pipeline_, VK_OBJECT_TYPE_PIPELINE, "lamp pipeline");
    }

    vkc::Camera camera_;

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> descriptor_sets_{};

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline object_pipeline_ = VK_NULL_HANDLE;
    VkPipeline lamp_pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 2.1 - Colours";
        ColoursApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
