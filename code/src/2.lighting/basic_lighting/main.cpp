// 2.2 Basic Lighting
//
// Adds to 2.1: normals, and the Phong reflection model -- ambient, diffuse and specular
// light, computed for every fragment. The lit cube stops being a silhouette.
//
// On the Vulkan side very little changes. The vertex gains a second attribute, the
// per-frame block gains two positions, and the push constants gain a normal matrix,
// which brings them to exactly 128 bytes: the size every Vulkan device guarantees. The
// normal matrix is also where this chapter's one real trap lives, in how a 3x3 matrix
// is laid out in memory.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/pipeline.hpp>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>

namespace {

// A normal per vertex, alongside the position.
//
// This is why the cube has been 24 vertices all along. A corner of a cube belongs to
// three faces pointing three different ways, so it needs three normals, and a vertex is
// the whole bundle of attributes -- three normals means three vertices.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

// The same unit cube, each face now carrying the direction it faces. The winding and
// vertex order are unchanged from 2.1.
constexpr float kH = 0.5F;
constexpr std::array<Vertex, 24> kVertices{{
    // front (+z)
    {{-kH, -kH, kH}, {0, 0, 1}}, {{kH, -kH, kH}, {0, 0, 1}},
    {{kH, kH, kH}, {0, 0, 1}},   {{-kH, kH, kH}, {0, 0, 1}},
    // back (-z)
    {{kH, -kH, -kH}, {0, 0, -1}}, {{-kH, -kH, -kH}, {0, 0, -1}},
    {{-kH, kH, -kH}, {0, 0, -1}}, {{kH, kH, -kH}, {0, 0, -1}},
    // left (-x)
    {{-kH, -kH, -kH}, {-1, 0, 0}}, {{-kH, -kH, kH}, {-1, 0, 0}},
    {{-kH, kH, kH}, {-1, 0, 0}},   {{-kH, kH, -kH}, {-1, 0, 0}},
    // right (+x)
    {{kH, -kH, kH}, {1, 0, 0}},  {{kH, -kH, -kH}, {1, 0, 0}},
    {{kH, kH, -kH}, {1, 0, 0}},  {{kH, kH, kH}, {1, 0, 0}},
    // top (+y)
    {{-kH, kH, kH}, {0, 1, 0}},  {{kH, kH, kH}, {0, 1, 0}},
    {{kH, kH, -kH}, {0, 1, 0}},  {{-kH, kH, -kH}, {0, 1, 0}},
    // bottom (-y)
    {{-kH, -kH, -kH}, {0, -1, 0}}, {{kH, -kH, -kH}, {0, -1, 0}},
    {{kH, -kH, kH}, {0, -1, 0}},   {{-kH, -kH, kH}, {0, -1, 0}},
}};

constexpr std::array<uint16_t, 36> kIndices{
    0,  1,  2,  2,  3,  0,   // front
    4,  5,  6,  6,  7,  4,   // back
    8,  9,  10, 10, 11, 8,   // left
    12, 13, 14, 14, 15, 12,  // right
    16, 17, 18, 18, 19, 16,  // top
    20, 21, 22, 22, 23, 20,  // bottom
};

// Set 0, binding 0. Matches `Globals` in lighting.slang.
//
// The two positions are vec4 rather than vec3. A vec3 would work -- std140 aligns it to
// 16 and the next member would land in the same place -- but only by a rule you have to
// remember, and the w component costs nothing.
struct Globals {
    glm::mat4 view;            // offset 0
    glm::mat4 projection;      // offset 64
    glm::vec4 light_colour;    // offset 128
    glm::vec4 light_position;  // offset 144
    glm::vec4 view_position;   // offset 160
};
static_assert(sizeof(Globals) == 176);

// Per object: exactly the 128 bytes Vulkan guarantees, with nothing to spare.
//
// normal_matrix is three vec4s and not a glm::mat3. The shader's float3x3 is laid out
// with each column padded to 16 bytes -- 48 in all -- while glm::mat3 packs its columns
// into 36. Send a glm::mat3 and the second and third columns arrive shifted by 4 and 8
// bytes: the lighting comes out wrong, and nothing -- compiler, driver or validation
// layer -- reports an error.
struct PushConstants {
    glm::mat4 model;                           // offset 0
    std::array<glm::vec4, 3> normal_matrix;    // offset 64, one column each
    glm::vec4 object_colour;                   // offset 112
};
static_assert(offsetof(PushConstants, normal_matrix) == 64);
static_assert(offsetof(PushConstants, object_colour) == 112);
static_assert(sizeof(PushConstants) == 128);

// The inverse transpose of the model matrix's upper 3x3, packed the way the shader
// expects it. For a pure rotation this equals the rotation itself; it only differs once
// there is a non-uniform scale, which is exactly when getting it wrong shows.
[[nodiscard]] std::array<glm::vec4, 3> normal_matrix_of(const glm::mat4& model) {
    const glm::mat3 m = glm::inverseTranspose(glm::mat3(model));
    return {glm::vec4(m[0], 0.0F), glm::vec4(m[1], 0.0F), glm::vec4(m[2], 0.0F)};
}

// Coral, the colour LearnOpenGL uses for this scene, and white for the light.
constexpr glm::vec4 kObjectColour{1.0F, 0.5F, 0.31F, 1.0F};
constexpr glm::vec4 kLightColour{1.0F, 1.0F, 1.0F, 1.0F};

// Where the lamp orbits. Driven by the frame counter rather than the wall clock so that
// tools/capture.py gets the same image every run.
constexpr float kLightOrbitRadius = 1.8F;
constexpr float kLightHeight = 1.2F;

class BasicLightingApp : public vkc::App {
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
        update_globals(slot, frame.extent, light_position);

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

        // The lit cube, turning exactly as in 2.1. Now each face brightens as it turns
        // toward the lamp and darkens as it turns away.
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

    // A circle in the xz plane, lifted to a constant height. Unchanged from 2.1, but
    // now the position matters: it is where every light direction points to.
    [[nodiscard]] static glm::vec3 orbit_position(float time) {
        return {kLightOrbitRadius * std::sin(time * 0.35F), kLightHeight,
                kLightOrbitRadius * std::cos(time * 0.35F)};
    }

    void draw_cube(VkCommandBuffer cmd, const glm::mat4& model,
                   const glm::vec4& colour) {
        const PushConstants push{
            .model = model,
            .normal_matrix = normal_matrix_of(model),
            .object_colour = colour,
        };
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

    void update_globals(size_t slot, VkExtent2D extent, const glm::vec3& light_position) {
        const Globals globals{
            .view = camera_.view(),
            .projection = build_projection(extent),
            .light_colour = kLightColour,
            .light_position = glm::vec4(light_position, 1.0F),
            // The specular term depends on where the eye is. Diffuse does not, which is
            // why a matte surface looks the same from anywhere and a shiny one does not.
            .view_position = glm::vec4(camera_.position, 1.0F),
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors and pipelines
    // ---------------------------------------------------------------------------

    void create_descriptors() {
        const VkDevice device = context().device();

        // One binding, the per-frame block, visible to both stages exactly as in 2.1.
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

    // Two pipelines from one module, exactly as 2.1 built them.
    void create_pipelines() {
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
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
        }};

        // Visible to both stages: the vertex shader reads `model` and `normal_matrix`,
        // the fragment shader reads `object_colour`.
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
            vkc::load_shader(device, LVK_CHAPTER_ID, "lighting.slang");

        const auto build = [&](const char* fragment_entry) {
            return vkc::PipelineBuilder(device)
                .shaders(shader, "vertexMain", fragment_entry)
                .vertex_input(std::span(&binding, 1), attributes)
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
        options.title = "LearnVulkan 2.2 - Basic Lighting";
        BasicLightingApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
