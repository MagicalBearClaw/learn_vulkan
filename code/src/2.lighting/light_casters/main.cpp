// 2.5 Light Casters
//
// Adds to 2.4: three kinds of light -- directional, point and spot -- and a light struct
// that can describe any of them.
//
// The Vulkan lesson is a return visit. Which kind of light the scene uses is fixed for
// the life of the pipeline, which is exactly the job specialisation constants were
// introduced for in chapter 1.9: the value is baked in when the pipeline is created and
// the shader's branch on it costs nothing per fragment. Change kLightType below and
// rebuild to see each kind.

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

// Set 0, binding 0. Matches `Globals` in light_casters.slang.
//
// The light has grown from four vec4s to five, plus five floats. In the shader's layout
// those floats pack tightly from offset 80 -- scalars align to 4, not 16 -- so the block
// is 100 bytes, and the padding rounds it to 112 the way std140 rounds a nested struct.
// The static_asserts are what make that a fact rather than a hope.
struct Light {
    glm::vec4 position;   // 0
    glm::vec4 direction;  // 16
    glm::vec4 ambient;    // 32
    glm::vec4 diffuse;    // 48
    glm::vec4 specular;   // 64
    float constant_term;  // 80
    float linear_term;    // 84
    float quadratic_term; // 88
    float inner_cutoff;   // 92, a cosine
    float outer_cutoff;   // 96, a cosine
    float padding[3];
};
static_assert(offsetof(Light, constant_term) == 80);
static_assert(offsetof(Light, outer_cutoff) == 96);
static_assert(sizeof(Light) == 112);

struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::vec4 view_position;  // 128
    Light light;              // 144
};
static_assert(offsetof(Globals, light) == 144);
static_assert(sizeof(Globals) == 256);

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

constexpr float kShininess = 32.0F;

// Which light the scene uses. Matches kLightType in the shader: change it, rebuild, run.
enum class LightType : int32_t {
    directional = 0,
    point = 1,
    spot = 2,
};
constexpr LightType kLightType = LightType::spot;

// The specialisation data block: one int, at offset 0, for constant id 0.
struct ShaderConstants {
    int32_t light_type = static_cast<int32_t>(kLightType);
};

constexpr glm::vec4 kLightAmbient{0.08F, 0.08F, 0.08F, 1.0F};
constexpr glm::vec4 kLightDiffuse{0.9F, 0.9F, 0.9F, 1.0F};
constexpr glm::vec4 kLightSpecular{1.0F, 1.0F, 1.0F, 1.0F};

// Attenuation. With these three, intensity is under half by 5 units away and about a
// tenth by 16 -- a light that covers this scene but visibly fades across it. The
// constant term is 1.0 so the formula never divides by less than 1: without it, a surface
// closer than one unit would receive more light than the light emits.
constexpr float kConstant = 1.0F;
constexpr float kLinear = 0.09F;
constexpr float kQuadratic = 0.032F;

// The spotlight's cone, as angles from its axis. Inside 12.5 degrees full strength,
// outside 17.5 none, a smooth ramp between.
constexpr float kInnerCutoffDegrees = 12.5F;
constexpr float kOuterCutoffDegrees = 17.5F;

// Where the directional light comes from: high, from the front right. The vector is the
// way the light travels, so it points down and away.
constexpr glm::vec3 kSunDirection{-0.4F, -1.0F, -0.6F};

// Ten crates at different distances, placed as 1.14 placed its cubes, so that distance
// falloff and the edge of the spotlight's cone both have something to fall across.
struct CratePlacement {
    glm::vec3 position;
    glm::vec3 axis;
    float angle;
};

constexpr std::array<CratePlacement, 10> kCrates{{
    {{0.0F, 0.0F, 0.0F}, {0.35F, 1.0F, 0.2F}, 0.4F},
    {{2.4F, 0.8F, -3.0F}, {1.0F, 0.3F, 0.0F}, 1.1F},
    {{-2.0F, -0.6F, -2.2F}, {0.2F, 0.4F, 1.0F}, 2.0F},
    {{1.2F, -1.4F, -5.5F}, {0.0F, 1.0F, 0.0F}, 0.7F},
    {{-3.1F, 1.6F, -6.0F}, {1.0F, 1.0F, 0.0F}, 2.6F},
    {{0.6F, 2.2F, -4.1F}, {0.5F, 0.0F, 1.0F}, 1.5F},
    {{-1.4F, 0.2F, -8.0F}, {1.0F, 0.0F, 0.3F}, 0.2F},
    {{3.4F, -1.0F, -7.2F}, {0.3F, 1.0F, 0.6F}, 3.0F},
    {{-4.0F, -1.8F, -4.6F}, {0.0F, 0.4F, 1.0F}, 1.8F},
    {{2.0F, 2.6F, -9.5F}, {1.0F, 0.6F, 0.2F}, 0.9F},
}};

// Where the point light sits: in the middle of the cluster, a little above it. Fixed
// rather than moving, so the falloff pattern holds still while you fly around it.
constexpr glm::vec3 kPointLightPosition{0.0F, 0.8F, -3.2F};

class LightCastersApp : public vkc::App {
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

        load_textures();
        create_uniform_buffers();
        create_material_buffer();
        create_descriptors();
        create_pipelines();

        // Straight back from the origin, looking down -z, so the lit cube is centred
        // and the lamp's orbit passes above and around it.
        // 1.14's starting view onto the same ten positions.
        camera_.position = {0.0F, 0.0F, 4.0F};

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

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.02F, 0.02F, 0.04F, 1.0F}});

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

        for (const CratePlacement& crate : kCrates) {
            // Still, not spinning: with a torch moving through the scene, spinning crates
            // make it hard to tell what the light is doing and what the geometry is.
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), crate.position) *
                glm::rotate(glm::mat4(1.0F), crate.angle, glm::normalize(crate.axis));
            draw_cube(frame.cmd, model);
        }

        // Only a point light has a place to put a lamp. A directional light has no
        // position, and the spotlight here is held by the camera, so a marker would sit
        // inside your own eye.
        if (kLightType == LightType::point) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
            draw_cube(frame.cmd,
                      glm::scale(glm::translate(glm::mat4(1.0F), kPointLightPosition),
                                 glm::vec3(0.15F)));
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

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

    // Fills in the fields this kind of light uses. The rest are left at zero and the
    // shader never reads them.
    [[nodiscard]] Light make_light() const {
        Light light{
            .position = {},
            .direction = {},
            .ambient = kLightAmbient,
            .diffuse = kLightDiffuse,
            .specular = kLightSpecular,
            .constant_term = kConstant,
            .linear_term = kLinear,
            .quadratic_term = kQuadratic,
            // Cosines, computed once here rather than per fragment in the shader.
            .inner_cutoff = std::cos(glm::radians(kInnerCutoffDegrees)),
            .outer_cutoff = std::cos(glm::radians(kOuterCutoffDegrees)),
            .padding = {},
        };

        switch (kLightType) {
        case LightType::directional:
            light.direction = glm::vec4(glm::normalize(kSunDirection), 0.0F);
            break;
        case LightType::point:
            light.position = glm::vec4(kPointLightPosition, 1.0F);
            break;
        case LightType::spot:
            // A torch in your hand: at the eye, pointing where you look. It moves with
            // the camera because it is rebuilt from the camera every frame.
            light.position = glm::vec4(camera_.position, 1.0F);
            light.direction = glm::vec4(camera_.front(), 0.0F);
            break;
        }
        return light;
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
            .view_position = glm::vec4(camera_.position, 1.0F),
            .light = make_light(),
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
            vkc::load_shader(device, LVK_CHAPTER_ID, "light_casters.slang");

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

        // The object pipeline takes the light type as specialisation constant 0. The lamp
        // pipeline is built from the same module and needs no specialisation: its
        // fragment entry point never reads kLightType.
        const ShaderConstants constants{};
        const std::array<VkSpecializationMapEntry, 1> entries{{
            {
                .constantID = 0,
                .offset = offsetof(ShaderConstants, light_type),
                .size = sizeof(ShaderConstants::light_type),
            },
        }};
        object_pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(shader, "vertexMain", "fragmentMain")
                .vertex_input(std::span(&binding, 1), attributes)
                .fragment_specialisation(
                    entries, std::span(reinterpret_cast<const std::byte*>(&constants),
                                       sizeof(constants)))
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true, VK_COMPARE_OP_LESS)
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_)
                .build();
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
        options.title = "LearnVulkan 2.5 - Light Casters";
        LightCastersApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
