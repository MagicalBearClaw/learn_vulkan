// 2.4 Lighting Maps
//
// Adds to 2.3: textures in the material. The diffuse and specular colours stop being
// one value for the whole surface and become per-texel lookups, so a single crate can be
// matte wood and shiny steel at once.
//
// On the Vulkan side, set 1 becomes a set with two descriptor types in it -- two
// combined image samplers and a uniform buffer -- which is the first time one set has
// mixed types, and the pool has to be sized per type to match. The two textures also
// differ in format for a reason that has nothing to do with Vulkan and everything to do
// with what their numbers mean: one is colour, the other is data.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
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

// Position, normal and texture coordinate -- 1.14's uv is back.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// The cube from 2.2 with 1.14's texture coordinates restored: each face is listed
// bottom-left, bottom-right, top-right, top-left seen from outside, and maps the whole
// texture onto itself the right way up.
constexpr float kH = 0.5F;
constexpr std::array<Vertex, 24> kVertices{{
    // front (+z)
    {{-kH, -kH, kH}, {0, 0, 1}, {0, 1}},  {{kH, -kH, kH}, {0, 0, 1}, {1, 1}},
    {{kH, kH, kH}, {0, 0, 1}, {1, 0}},    {{-kH, kH, kH}, {0, 0, 1}, {0, 0}},
    // back (-z)
    {{kH, -kH, -kH}, {0, 0, -1}, {0, 1}}, {{-kH, -kH, -kH}, {0, 0, -1}, {1, 1}},
    {{-kH, kH, -kH}, {0, 0, -1}, {1, 0}}, {{kH, kH, -kH}, {0, 0, -1}, {0, 0}},
    // left (-x)
    {{-kH, -kH, -kH}, {-1, 0, 0}, {0, 1}}, {{-kH, -kH, kH}, {-1, 0, 0}, {1, 1}},
    {{-kH, kH, kH}, {-1, 0, 0}, {1, 0}},   {{-kH, kH, -kH}, {-1, 0, 0}, {0, 0}},
    // right (+x)
    {{kH, -kH, kH}, {1, 0, 0}, {0, 1}},  {{kH, -kH, -kH}, {1, 0, 0}, {1, 1}},
    {{kH, kH, -kH}, {1, 0, 0}, {1, 0}},  {{kH, kH, kH}, {1, 0, 0}, {0, 0}},
    // top (+y)
    {{-kH, kH, kH}, {0, 1, 0}, {0, 1}},  {{kH, kH, kH}, {0, 1, 0}, {1, 1}},
    {{kH, kH, -kH}, {0, 1, 0}, {1, 0}},  {{-kH, kH, -kH}, {0, 1, 0}, {0, 0}},
    // bottom (-y)
    {{-kH, -kH, -kH}, {0, -1, 0}, {0, 1}}, {{kH, -kH, -kH}, {0, -1, 0}, {1, 1}},
    {{kH, -kH, kH}, {0, -1, 0}, {1, 0}},   {{-kH, -kH, kH}, {0, -1, 0}, {0, 0}},
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

// Set 1, binding 2. All that is left of 2.3's Material once its colours have become
// textures. The shader's block is 4 bytes; this is padded to 16 so it stays correct if
// a later chapter packs several into one buffer.
struct MaterialParams {
    float shininess;
    float padding[3];
};
static_assert(sizeof(MaterialParams) == 16);

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

// Steel wants a tighter highlight than 2.2's plastic-ish 32.
constexpr float kShininess = 32.0F;

// A white light with the three components now doing distinct jobs. The ambient and
// diffuse parts multiply a texture that already carries the surface's colour, so they
// are well below 1.0; the specular part multiplies the specular map, which decides on its
// own how much of it any texel reflects.
constexpr glm::vec4 kLightAmbient{0.2F, 0.2F, 0.2F, 1.0F};
constexpr glm::vec4 kLightDiffuse{0.6F, 0.6F, 0.6F, 1.0F};
constexpr glm::vec4 kLightSpecular{1.0F, 1.0F, 1.0F, 1.0F};

// Three crates sharing one material.
constexpr std::array<glm::vec3, 3> kCrates{{
    {-1.8F, 0.0F, -0.4F},
    {0.0F, 0.0F, 0.4F},
    {1.8F, 0.0F, -0.4F},
}};

// The lamp sweeps side to side in front of the crates, low down, rather than circling
// above them as in 2.3.
//
// A specular highlight appears where the surface is a mirror between the light and the
// eye. With both the camera and the lamp above the crates, that mirror point lies above
// every front face and no highlight lands on a crate at all -- which is exactly what a
// first version of this sample did, and a lighting-maps demo with no highlights shows
// nothing. Low and in front, the highlights slide across the steel as the lamp moves.
constexpr float kLightSweep = 1.2F;     // half-width of the sweep along x
constexpr float kLightHeight = 0.25F;
constexpr float kLightDistance = 3.0F;  // z, in front of the crates

class LightingMapsApp : public vkc::App {
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

        load_textures();
        create_uniform_buffers();
        create_material_buffer();
        create_descriptors();
        create_pipelines();

        // Straight back from the origin, looking down -z, so the lit cube is centred
        // and the lamp's orbit passes above and around it.
        camera_.position = {0.0F, 0.35F, 5.0F};
        camera_.pitch = -3.0F;

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

        material_buffer_.destroy();
        sampler_.destroy();
        specular_map_.destroy();
        diffuse_map_.destroy();
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

        // One material for all three crates, so set 1 is bound once, outside the loop.
        // Per-object data still changes per draw, and that is what the push constants
        // are for.
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 1, 1, &material_set_, 0, nullptr);

        for (size_t i = 0; i < kCrates.size(); ++i) {
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), kCrates[i]) *
                glm::rotate(glm::mat4(1.0F), (time * 0.25F - 0.375F) + (static_cast<float>(i) - 1.0F) * 0.65F,
                            glm::normalize(glm::vec3(0.2F, 1.0F, 0.1F)));
            draw_cube(frame.cmd, model);
        }

        // The lamp. A different pipeline, but built from the same pipeline layout, so
        // switching to it keeps both descriptor sets bound. Its fragment stage reads
        // only set 0.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        // Smaller than 2.3's lamp: it now sits between the camera and the crates, and at
        // 0.2 it hid most of a crate.
        draw_cube(frame.cmd, glm::scale(glm::translate(glm::mat4(1.0F), light_position),
                                        glm::vec3(0.08F)));

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

    // A side-to-side sweep at a fixed height and distance, as the constants above
    // explain. The phase puts the lamp in the gap between the middle and right crates on
    // the captured frame.
    [[nodiscard]] static glm::vec3 orbit_position(float time) {
        return {kLightSweep * std::sin(time * 0.5F - 0.48F), kLightHeight, kLightDistance};
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

    // The same loader chapter 1.11 wrote, now vkc::load_texture, called twice with two
    // different formats.
    void load_textures() {
        // The diffuse map is colour. It was painted to look right on an sRGB display,
        // so its bytes are sRGB-encoded, and _SRGB tells the GPU to decode them to linear
        // light when sampling -- which is what the lighting arithmetic needs.
        diffuse_map_ = vkc::load_texture(context(),
                                         vkc::asset_path("textures/lvk_crate_diffuse.png"),
                                         VK_FORMAT_R8G8B8A8_SRGB);

        // The specular map is data: 128 means "reflect half", not "a mid-grey that looks
        // half as bright". Loaded as _SRGB, every value would be run through the decode
        // curve on the way in and 128 would arrive as about 0.22. UNORM leaves the numbers
        // alone.
        specular_map_ = vkc::load_texture(context(),
                                          vkc::asset_path("textures/lvk_crate_specular.png"),
                                          VK_FORMAT_R8G8B8A8_UNORM);

        // One sampler serves both. A sampler describes *how* to read a texture, not
        // which one, so any number of textures that want the same filtering can share it.
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

        context().name(diffuse_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate diffuse map");
        context().name(specular_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate specular map");
    }

    // Written once, like 2.3's materials.
    void create_material_buffer() {
        material_buffer_ = vkc::Buffer(context().allocator(), sizeof(MaterialParams),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT);
        const MaterialParams params{.shininess = kShininess, .padding = {}};
        std::memcpy(material_buffer_.mapped(), &params, sizeof(params));
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

        frame_set_layout_ =
            create_set_layout(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        // The material layout: three bindings, two types, fragment stage only. Binding
        // numbers match `[[vk::binding(n, 1)]]` in the shader; the types have to match
        // too, and a sampler declared where a uniform buffer is expected is a validation
        // error at pipeline creation.
        const std::array<VkDescriptorSetLayoutBinding, 3> material_bindings{{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
        }};
        const VkDescriptorSetLayoutCreateInfo material_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(material_bindings.size()),
            .pBindings = material_bindings.data(),
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &material_layout_info, nullptr,
                                             &material_set_layout_));

        // Pool sizes are per descriptor *type*, summed over every set the pool will hold.
        // Uniform buffers: one per frame set, plus the material's. Combined image
        // samplers: the material's two. maxSets counts sets, not descriptors.
        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight + 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + 1,
            .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
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

        const VkDescriptorSetAllocateInfo material_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &material_set_layout_,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &material_alloc, &material_set_));

        for (size_t i = 0; i < vkc::kFramesInFlight; ++i) {
            write_uniform(frame_sets_[i], uniform_buffers_[i].handle(), sizeof(Globals));
        }

        // The material set takes three writes in one call. Each image write names the
        // layout the image will be in *when it is sampled*, not the one it is in now --
        // load_texture has already left both in SHADER_READ_ONLY_OPTIMAL.
        const VkDescriptorImageInfo diffuse_info{
            .sampler = sampler_.handle(),
            .imageView = diffuse_map_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo specular_info{
            .sampler = sampler_.handle(),
            .imageView = specular_map_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorBufferInfo params_info{
            .buffer = material_buffer_.handle(),
            .offset = 0,
            .range = sizeof(MaterialParams),
        };

        const auto image_write = [&](uint32_t binding, const VkDescriptorImageInfo& info) {
            return VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = material_set_,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &info,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            };
        };

        const std::array<VkWriteDescriptorSet, 3> material_writes{{
            image_write(0, diffuse_info),
            image_write(1, specular_info),
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = material_set_,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &params_info,
                .pTexelBufferView = nullptr,
            },
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(material_writes.size()),
                               material_writes.data(), 0, nullptr);

        context().name(material_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "crate material");
    }

    // Two pipelines from one module, exactly as 2.1 built them.
    void create_pipelines() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        const std::array<VkVertexInputAttributeDescription, 3> attributes{{
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
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
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
            vkc::load_shader(device, LVK_CHAPTER_ID, "lighting_maps.slang");

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

    vkc::Image diffuse_map_;
    vkc::Image specular_map_;
    vkc::Sampler sampler_;

    vkc::Buffer material_buffer_;
    VkDescriptorSet material_set_ = VK_NULL_HANDLE;

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
        options.title = "LearnVulkan 2.4 - Lighting Maps";
        LightingMapsApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
