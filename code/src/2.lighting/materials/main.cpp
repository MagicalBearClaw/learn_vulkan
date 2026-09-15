// 2.3 Materials
//
// Adds to 2.2: materials, and the second descriptor set.
//
// The lighting model is unchanged. What changes is where its inputs live. 2.2 baked a
// surface's response to light into shader constants and pushed a single colour per
// object; that filled the push constants to the last byte. Here a material becomes four
// values in a uniform buffer, each material gets a descriptor set of its own at set 1,
// and set 0 keeps the per-frame data it has held since 2.1.
//
// The Vulkan lesson is the split. Set 0 is bound once per frame. Set 1 is rebound once
// per material. Push constants change once per object. Three lifetimes, three
// mechanisms, and binding the frequently changing one never disturbs the others.

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

// Set 0, binding 0. Matches `Globals` in materials.slang.
struct Light {
    glm::vec4 position;  // offset 0 within Light
    glm::vec4 ambient;   // 16
    glm::vec4 diffuse;   // 32
    glm::vec4 specular;  // 48
};

struct Globals {
    glm::mat4 view;           // offset 0
    glm::mat4 projection;     // offset 64
    glm::vec4 view_position;  // offset 128
    Light light;              // offset 144
};
static_assert(offsetof(Globals, light) == 144);
static_assert(sizeof(Globals) == 208);

// Set 1, binding 0. One of these per material.
//
// The shader's block is 52 bytes -- shininess is a lone float at offset 48 -- and this
// struct pads it to 64. The padding is not required by the shader, but a whole number of
// 16-byte slots is what std140 would round a nested copy of this struct to, and it means
// the struct stays correct if a later chapter puts an array of materials in one buffer.
struct Material {
    glm::vec4 ambient;   // offset 0
    glm::vec4 diffuse;   // offset 16
    glm::vec4 specular;  // offset 32
    float shininess;     // offset 48
    float padding[3];
};
static_assert(offsetof(Material, shininess) == 48);
static_assert(sizeof(Material) == 64);

// Per object: the object colour has moved into the material, so 112 of 128 bytes.
struct PushConstants {
    glm::mat4 model;                         // offset 0
    std::array<glm::vec4, 3> normal_matrix;  // offset 64
};
static_assert(offsetof(PushConstants, normal_matrix) == 64);
static_assert(sizeof(PushConstants) == 112);

[[nodiscard]] std::array<glm::vec4, 3> normal_matrix_of(const glm::mat4& model) {
    const glm::mat3 m = glm::inverseTranspose(glm::mat3(model));
    return {glm::vec4(m[0], 0.0F), glm::vec4(m[1], 0.0F), glm::vec4(m[2], 0.0F)};
}

// Five materials from the classic table in McReynolds and Blythe, "Advanced Graphics
// Programming Using OpenGL" (2005), with shininess scaled from 0..1 to an exponent. The
// values assume a white light at full strength in all three components, which is what
// kLight below supplies.
struct NamedMaterial {
    const char* name;
    Material material;
};

constexpr std::array<NamedMaterial, 5> kMaterials{{
    {"emerald",
     {{0.0215F, 0.1745F, 0.0215F, 1}, {0.07568F, 0.61424F, 0.07568F, 1},
      {0.633F, 0.727811F, 0.633F, 1}, 76.8F, {}}},
    {"gold",
     {{0.24725F, 0.1995F, 0.0745F, 1}, {0.75164F, 0.60648F, 0.22648F, 1},
      {0.628281F, 0.555802F, 0.366065F, 1}, 51.2F, {}}},
    {"ruby",
     {{0.1745F, 0.01175F, 0.01175F, 1}, {0.61424F, 0.04136F, 0.04136F, 1},
      {0.727811F, 0.626959F, 0.626959F, 1}, 76.8F, {}}},
    {"obsidian",
     {{0.05375F, 0.05F, 0.06625F, 1}, {0.18275F, 0.17F, 0.22525F, 1},
      {0.332741F, 0.328634F, 0.346435F, 1}, 38.4F, {}}},
    {"cyan plastic",
     {{0.0F, 0.1F, 0.06F, 1}, {0.0F, 0.50980392F, 0.50980392F, 1},
      {0.50196078F, 0.50196078F, 0.50196078F, 1}, 32.0F, {}}},
}};

// White, at full strength in every component.
constexpr glm::vec4 kLightAmbient{1.0F, 1.0F, 1.0F, 1.0F};
constexpr glm::vec4 kLightDiffuse{1.0F, 1.0F, 1.0F, 1.0F};
constexpr glm::vec4 kLightSpecular{1.0F, 1.0F, 1.0F, 1.0F};

// The cubes sit in a row along x, one per material.
constexpr float kCubeSpacing = 1.6F;

// Where the lamp orbits. Driven by the frame counter rather than the wall clock so that
// tools/capture.py gets the same image every run.
constexpr float kLightOrbitRadius = 3.0F;
constexpr float kLightHeight = 2.0F;

class MaterialsApp : public vkc::App {
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

        depth_format_ = choose_depth_format();
        create_depth_buffer(swapchain().extent());

        create_uniform_buffers();
        create_material_buffers();
        create_descriptors();
        create_pipelines();

        // Straight back from the origin, looking down -z, so the lit cube is centred
        // and the lamp's orbit passes above and around it.
        camera_.position = {0.0F, 1.2F, 7.0F};
        camera_.pitch = -8.0F;

        spdlog::info("Right-click to capture the mouse, WASD to fly, Escape to quit.");
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

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
        vkDestroyDescriptorSetLayout(device, material_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        for (vkc::Buffer& buffer : material_buffers_) {
            buffer.destroy();
        }
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

        begin_rendering(frame);

        set_viewport(frame);

        // Set 0, once for the whole frame. Nothing below rebinds it.
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object_pipeline_);

        for (size_t i = 0; i < kMaterials.size(); ++i) {
            // Set 1 only. The `1` is firstSet: this call writes set 1 and leaves set 0
            // bound, because both were bound through the same pipeline layout.
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout_, 1, 1, &material_sets_[i], 0,
                                    nullptr);

            const float x = (static_cast<float>(i) - 2.0F) * kCubeSpacing;
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), glm::vec3(x, 0.0F, 0.0F)) *
                glm::rotate(glm::mat4(1.0F), time * 0.3F + static_cast<float>(i),
                            glm::normalize(glm::vec3(0.3F, 1.0F, 0.15F)));
            draw_cube(frame.cmd, model);
        }

        // The lamp. A different pipeline, but built from the same pipeline layout, so
        // switching to it keeps both descriptor sets bound. Its fragment stage reads
        // only set 0.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        draw_cube(frame.cmd, glm::scale(glm::translate(glm::mat4(1.0F), light_position),
                                        glm::vec3(0.2F)));

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

    void draw_cube(VkCommandBuffer cmd, const glm::mat4& model) {
        const PushConstants push{
            .model = model,
            .normal_matrix = normal_matrix_of(model),
        };
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(push), &push);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing, unchanged from 1.14
    // ---------------------------------------------------------------------------

    void begin_rendering(const vkc::FrameInfo& frame) {
        const VkClearValue colour_clear{.color = {{0.02F, 0.02F, 0.04F, 1.0F}}};
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
            .clearValue = colour_clear,
        };

        const VkClearValue depth_clear{.depthStencil = {1.0F, 0}};
        const VkRenderingAttachmentInfo depth_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = depth_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue = depth_clear,
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
            .pDepthAttachment = &depth_attachment,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.cmd, &rendering);
    }

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

    [[nodiscard]] VkFormat choose_depth_format() {
        constexpr std::array candidates{
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        };

        for (const VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(context().physical_device(), format,
                                                &properties);
            if ((properties.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth attachment format was found.");
    }

    void create_depth_buffer(VkExtent2D extent) {
        context().wait_idle();
        depth_ = vkc::Image(
            context(), vkc::ImageDesc{
                           .format = depth_format_,
                           .extent = extent,
                           .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           .mip_levels = 1,
                           .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                       });

        vkc::immediate_submit(context(), [&](VkCommandBuffer cmd) {
            depth_.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        });

        context().name(depth_.handle(), VK_OBJECT_TYPE_IMAGE, "depth buffer");
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
            .view_position = glm::vec4(camera_.position, 1.0F),
            .light =
                {
                    .position = glm::vec4(light_position, 1.0F),
                    .ambient = kLightAmbient,
                    .diffuse = kLightDiffuse,
                    .specular = kLightSpecular,
                },
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors and pipelines
    // ---------------------------------------------------------------------------

    // Materials never change, so each gets one small buffer written once at start-up
    // and never touched again -- no copy per frame in flight, for the same reason 1.11's
    // texture needed none. Only data that changes while the GPU might still be reading
    // the previous frame's copy needs one per frame.
    void create_material_buffers() {
        for (size_t i = 0; i < kMaterials.size(); ++i) {
            material_buffers_[i] = vkc::Buffer(
                context().allocator(), sizeof(Material), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            std::memcpy(material_buffers_[i].mapped(), &kMaterials[i].material,
                        sizeof(Material));
        }
    }

    [[nodiscard]] VkDescriptorSetLayout create_set_layout(VkShaderStageFlags stages) {
        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorSetLayout(context().device(), &info, nullptr, &layout));
        return layout;
    }

    void write_uniform(VkDescriptorSet set, VkBuffer buffer, VkDeviceSize size) {
        const VkDescriptorBufferInfo buffer_info{.buffer = buffer, .offset = 0, .range = size};
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_info,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(context().device(), 1, &write, 0, nullptr);
    }

    void create_descriptors() {
        const VkDevice device = context().device();

        // Two layouts that happen to look identical -- one uniform buffer at binding 0 --
        // but mean different things. The frame block is read by both stages; the
        // material only by the fragment stage.
        frame_set_layout_ =
            create_set_layout(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        material_set_layout_ = create_set_layout(VK_SHADER_STAGE_FRAGMENT_BIT);

        // One pool for everything: a set per frame in flight for set 0, and a set per
        // material for set 1. maxSets and the descriptor count both have to cover the
        // total, and running out is VK_ERROR_OUT_OF_POOL_MEMORY at allocation time.
        const auto set_count = static_cast<uint32_t>(vkc::kFramesInFlight + kMaterials.size());
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set_count};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = set_count,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const std::array<VkDescriptorSetLayout, vkc::kFramesInFlight> frame_layouts{
            frame_set_layout_, frame_set_layout_};
        const VkDescriptorSetAllocateInfo frame_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = vkc::kFramesInFlight,
            .pSetLayouts = frame_layouts.data(),
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &frame_alloc, frame_sets_.data()));

        std::array<VkDescriptorSetLayout, kMaterials.size()> material_layouts{};
        material_layouts.fill(material_set_layout_);
        const VkDescriptorSetAllocateInfo material_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = static_cast<uint32_t>(kMaterials.size()),
            .pSetLayouts = material_layouts.data(),
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &material_alloc, material_sets_.data()));

        for (size_t i = 0; i < vkc::kFramesInFlight; ++i) {
            write_uniform(frame_sets_[i], uniform_buffers_[i].handle(), sizeof(Globals));
        }
        for (size_t i = 0; i < kMaterials.size(); ++i) {
            write_uniform(material_sets_[i], material_buffers_[i].handle(), sizeof(Material));
            context().name(material_sets_[i], VK_OBJECT_TYPE_DESCRIPTOR_SET,
                           kMaterials[i].name);
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

        // Only the vertex stage reads the push constants now that the object colour has
        // become part of the material.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        // Set layouts in set order: index 0 is set 0, index 1 is set 1. The order is the
        // numbering -- there is no other place a set number is declared on this side.
        const std::array<VkDescriptorSetLayout, 2> set_layouts{frame_set_layout_,
                                                               material_set_layout_};

        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = static_cast<uint32_t>(set_layouts.size()),
            .pSetLayouts = set_layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "materials.slang");

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
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    std::array<vkc::Buffer, kMaterials.size()> material_buffers_;
    std::array<VkDescriptorSet, kMaterials.size()> material_sets_{};

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout material_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline object_pipeline_ = VK_NULL_HANDLE;
    VkPipeline lamp_pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 2.3 - Materials";
        MaterialsApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
