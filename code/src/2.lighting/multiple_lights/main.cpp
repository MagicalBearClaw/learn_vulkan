// 2.6 Multiple Lights
//
// Adds to 2.5: six lights of three kinds, shading the same scene at once.
//
// The Vulkan lesson is what happens to a uniform block when the data it carries stops
// being one of something. The light becomes an array sized for the worst case, the number
// of live entries becomes a specialisation constant, and the kind of each light -- which
// 2.5 specialised on -- moves back into the data, because an array holding a sun and a
// torch has no single kind to bake in.

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

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// 2.4's cube, unchanged.
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

enum class LightType : int32_t {
    directional = 0,
    point = 1,
    spot = 2,
};

// 2.5's Light with `type` appended. The five floats packed from offset 80 and left the
// block at 100 bytes inside a 112-byte slot; `type` lands in that padding at 100, so the
// struct that can describe three kinds of light costs no more than the one that could
// describe one. The array below inherits the same 112-byte stride.
struct Light {
    glm::vec4 position;    // 0
    glm::vec4 direction;   // 16
    glm::vec4 ambient;     // 32
    glm::vec4 diffuse;     // 48
    glm::vec4 specular;    // 64
    float constant_term;   // 80
    float linear_term;     // 84
    float quadratic_term;  // 88
    float inner_cutoff;    // 92, a cosine
    float outer_cutoff;    // 96, a cosine
    int32_t type;          // 100
    float padding[2];      // 104
};
static_assert(offsetof(Light, type) == 100);
static_assert(sizeof(Light) == 112);

// Must match kMaxLights in the shader: it is the array's declared length on both sides,
// and nothing checks it for us.
constexpr size_t kMaxLights = 8;

struct Globals {
    glm::mat4 view;                        // 0
    glm::mat4 projection;                  // 64
    glm::vec4 view_position;               // 128
    std::array<Light, kMaxLights> lights;  // 144
};
static_assert(offsetof(Globals, lights) == 144);
static_assert(sizeof(Globals) == 144 + kMaxLights * sizeof(Light));

struct MaterialParams {
    float shininess;
    float padding[3];
};
static_assert(sizeof(MaterialParams) == 16);

// The lamp colour brings the push block to exactly 128 bytes -- the whole of the range
// Vulkan guarantees on every implementation, and the reason chapter 1.11 said to treat
// push constants as a handful of values rather than a place to put data.
struct PushConstants {
    glm::mat4 model;                         // 0
    std::array<glm::vec4, 3> normal_matrix;  // 64
    glm::vec4 colour;                        // 112
};
static_assert(offsetof(PushConstants, colour) == 112);
static_assert(sizeof(PushConstants) == 128);

[[nodiscard]] std::array<glm::vec4, 3> normal_matrix_of(const glm::mat4& model) {
    const glm::mat3 m = glm::inverseTranspose(glm::mat3(model));
    return {glm::vec4(m[0], 0.0F), glm::vec4(m[1], 0.0F), glm::vec4(m[2], 0.0F)};
}

constexpr float kShininess = 32.0F;

// Attenuation, 2.5's values. Tighter ones were tried first, on the reasoning that four
// bulbs in one scene would wash it out; they did the opposite. With a steep falloff each
// bulb only reaches the crate it is sitting next to, and its colour shows up as a thin rim
// on one edge rather than as a pool of light. The scene has no floor and no walls, so the
// only thing a bulb has to light is a crate several units away, and the light has to carry
// that far to show at all.
constexpr float kConstant = 1.0F;
constexpr float kLinear = 0.09F;
constexpr float kQuadratic = 0.032F;

constexpr float kInnerCutoffDegrees = 12.5F;
constexpr float kOuterCutoffDegrees = 17.5F;

// The sun: cool, dim, and from the upper front right. It is the only light whose ambient
// survives at full strength, so it also sets the floor the rest of the scene sits on.
constexpr glm::vec3 kSunDirection{-0.4F, -1.0F, -0.6F};
constexpr glm::vec3 kSunColour{0.24F, 0.26F, 0.30F};

// Four bulbs, placed among the crates so their pools overlap in a few places and nowhere
// else. Distinct hues, because the point of the chapter is seeing which light did what.
struct PointLight {
    glm::vec3 position;
    glm::vec3 colour;
};

// How bright a bulb is at the source, before distance takes its cut. The colours below are
// hues, normalised so the brightest channel is 1.0; without this multiplier a bulb four
// units from a crate lands at about a third of that, which the sun and the torch -- both
// white, both reaching everything -- simply drown out.
constexpr float kBulbStrength = 2.2F;

constexpr std::array<PointLight, 4> kPointLights{{
    {{1.6F, 1.2F, -1.5F}, {1.00F, 0.55F, 0.18F}},    // warm
    {{-2.6F, 0.4F, -3.4F}, {0.16F, 0.75F, 0.95F}},   // cyan
    {{2.6F, -1.2F, -6.0F}, {0.25F, 0.90F, 0.35F}},   // green
    {{-1.0F, 2.0F, -7.5F}, {0.85F, 0.25F, 0.80F}},   // magenta
}};

// The torch is deliberately dimmer than 2.5's. There it was the only light and had to
// carry the scene; here it competes with four coloured bulbs, and at full strength its
// white simply erases them wherever it points.
constexpr glm::vec3 kTorchColour{0.55F, 0.55F, 0.52F};

// One sun + four bulbs + one torch.
constexpr int32_t kLightCount = 1 + static_cast<int32_t>(kPointLights.size()) + 1;
static_assert(kLightCount <= static_cast<int32_t>(kMaxLights));

// Specialisation constant 0 is now the light *count*, not the light kind.
struct ShaderConstants {
    int32_t light_count = kLightCount;
};

// 1.14's ten positions, as in 2.5.
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

class MultipleLightsApp : public vkc::App {
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

        camera_.position = {0.0F, 0.0F, 4.0F};

        spdlog::info("Right-click to capture the mouse, WASD to fly, Escape to quit.");
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

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

        begin_rendering(frame);

        set_viewport(frame);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object_pipeline_);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 1, 1, &material_set_, 0, nullptr);

        for (const CratePlacement& crate : kCrates) {
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), crate.position) *
                glm::rotate(glm::mat4(1.0F), crate.angle, glm::normalize(crate.axis));
            draw_cube(frame.cmd, model, glm::vec4(1.0F));
        }

        // One lamp per bulb, each in its own colour. The sun has no position to mark and
        // the torch is at the camera, so four of the six lights are visible as objects.
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        for (const PointLight& light : kPointLights) {
            const glm::mat4 model = glm::scale(
                glm::translate(glm::mat4(1.0F), light.position), glm::vec3(0.12F));
            draw_cube(frame.cmd, model, glm::vec4(light.colour, 1.0F));
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

    void draw_cube(VkCommandBuffer cmd, const glm::mat4& model, const glm::vec4& colour) {
        const PushConstants push{
            .model = model,
            .normal_matrix = normal_matrix_of(model),
            .colour = colour,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0, 0);
    }

    // Unused fields are left at zero, as in 2.5: the shader never reads the ones this
    // kind of light does not have.
    [[nodiscard]] static Light make_directional() {
        return Light{
            .position = {},
            .direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
            .ambient = glm::vec4(kSunColour * 0.22F, 1.0F),
            .diffuse = glm::vec4(kSunColour, 1.0F),
            .specular = glm::vec4(kSunColour * 1.3F, 1.0F),
            .constant_term = 0.0F,
            .linear_term = 0.0F,
            .quadratic_term = 0.0F,
            .inner_cutoff = 0.0F,
            .outer_cutoff = 0.0F,
            .type = static_cast<int32_t>(LightType::directional),
            .padding = {},
        };
    }

    [[nodiscard]] static Light make_point(const PointLight& bulb) {
        return Light{
            .position = glm::vec4(bulb.position, 1.0F),
            .direction = {},
            // A bulb's ambient has to be small: four of them add up, and each one is
            // already carrying a diffuse term at full colour.
            .ambient = glm::vec4(bulb.colour * 0.04F, 1.0F),
            .diffuse = glm::vec4(bulb.colour * kBulbStrength, 1.0F),
            .specular = glm::vec4(bulb.colour * kBulbStrength, 1.0F),
            .constant_term = kConstant,
            .linear_term = kLinear,
            .quadratic_term = kQuadratic,
            .inner_cutoff = 0.0F,
            .outer_cutoff = 0.0F,
            .type = static_cast<int32_t>(LightType::point),
            .padding = {},
        };
    }

    [[nodiscard]] Light make_spot() const {
        return Light{
            .position = glm::vec4(camera_.position, 1.0F),
            .direction = glm::vec4(camera_.front(), 0.0F),
            // No ambient at all. A torch lights what it points at; the sun is what keeps
            // the rest of the scene out of pure black.
            .ambient = {},
            .diffuse = glm::vec4(kTorchColour, 1.0F),
            .specular = glm::vec4(1.0F),
            .constant_term = kConstant,
            .linear_term = kLinear,
            .quadratic_term = kQuadratic,
            .inner_cutoff = std::cos(glm::radians(kInnerCutoffDegrees)),
            .outer_cutoff = std::cos(glm::radians(kOuterCutoffDegrees)),
            .type = static_cast<int32_t>(LightType::spot),
            .padding = {},
        };
    }

    // The order the lights go into the array does not matter -- addition commutes -- but
    // keeping it fixed makes the shader easy to reason about when something looks wrong.
    [[nodiscard]] std::array<Light, kMaxLights> make_lights() const {
        std::array<Light, kMaxLights> lights{};
        size_t next = 0;
        lights[next++] = make_directional();
        for (const PointLight& bulb : kPointLights) {
            lights[next++] = make_point(bulb);
        }
        lights[next++] = make_spot();
        return lights;
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

    void update_globals(size_t slot, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = build_projection(extent),
            .view_position = glm::vec4(camera_.position, 1.0F),
            .lights = make_lights(),
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors and pipelines
    // ---------------------------------------------------------------------------

    void load_textures() {
        diffuse_map_ = vkc::load_texture(context(),
                                         vkc::asset_path("textures/lvk_crate_diffuse.png"),
                                         VK_FORMAT_R8G8B8A8_SRGB);
        specular_map_ = vkc::load_texture(context(),
                                          vkc::asset_path("textures/lvk_crate_specular.png"),
                                          VK_FORMAT_R8G8B8A8_UNORM);
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

        context().name(diffuse_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate diffuse map");
        context().name(specular_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate specular map");
    }

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

        // Both stages read the block now: the vertex shader takes the matrices, the lamp's
        // fragment shader takes the colour. One range covering both is the simplest thing
        // that works, and the stage flags here must cover every stage that reads any of it.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

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
            vkc::load_shader(device, LVK_CHAPTER_ID, "multiple_lights.slang");

        const ShaderConstants constants{};
        const std::array<VkSpecializationMapEntry, 1> entries{{
            {
                .constantID = 0,
                .offset = offsetof(ShaderConstants, light_count),
                .size = sizeof(ShaderConstants::light_count),
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

        lamp_pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(shader, "vertexMain", "lampFragmentMain")
                .vertex_input(std::span(&binding, 1), attributes)
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true, VK_COMPARE_OP_LESS)
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_)
                .build();

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
        options.title = "LearnVulkan 2.6 - Multiple Lights";
        MultipleLightsApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
