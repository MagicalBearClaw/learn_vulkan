// 4.2 Pipeline state: blending & culling
//
// The other half of the fixed-function tail. 4.1 covered what happens to a fragment's
// depth; this covers what happens to its colour, plus the one piece of fixed-function
// state that runs *before* the fragment shader rather than after it.
//
//   * face culling, which throws away triangles by their winding before they are
//     rasterised at all, and which has a Vulkan-specific trap in it
//   * alpha testing, which is `discard` in the fragment shader and nothing else
//   * blending, the full VkPipelineColorBlendAttachmentState, and the reason
//     depthWriteEnable -- introduced in passing in 4.1 -- finally matters
//
// The scene is 4.1's floor and crates, with a stand of ferns cut out of their quads and
// three panes of coloured glass in front of everything.
//
//   1  cycles the opaque geometry's cull mode: back, none, front
//   2  toggles back-to-front sorting of the glass panes
//   3  toggles depth writes on the glass panes
//   4  switches the ferns between alpha testing and alpha blending

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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// 2.4's cube, unchanged, for the crates.
constexpr float kH = 0.5F;
constexpr std::array<Vertex, 24> kCubeVertices{{
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

constexpr std::array<uint16_t, 36> kCubeIndices{
    0,  1,  2,  2,  3,  0,   // front
    4,  5,  6,  6,  7,  4,   // back
    8,  9,  10, 10, 11, 8,   // left
    12, 13, 14, 14, 15, 12,  // right
    16, 17, 18, 18, 19, 16,  // top
    20, 21, 22, 22, 23, 20,  // bottom
};

constexpr float kFloorY = -1.0F;
// Wider than 4.1's floor, so that its far edge falls outside the frame rather than
// drawing a hard line across the horizon.
constexpr float kFloorHalf = 22.0F;
constexpr float kFloorTiles = 19.0F;

// Two quads in one buffer: 4.1's floor, in world space, and a unit upright quad that
// every fern and every pane of glass is an instance of. The upright quad stands on
// y = 0 and is one unit tall, so translating it to the floor's y puts its foot on the
// ground with no arithmetic.
constexpr std::array<Vertex, 8> kQuadVertices{{
    // floor, indices 0..5
    {{-kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {0, 0}},
    {{-kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {0, kFloorTiles}},
    {{kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {kFloorTiles, kFloorTiles}},
    {{kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {kFloorTiles, 0}},
    // upright unit quad facing +z, indices 6..11
    {{-0.5F, 0.0F, 0.0F}, {0, 0, 1}, {0, 1}},
    {{0.5F, 0.0F, 0.0F}, {0, 0, 1}, {1, 1}},
    {{0.5F, 1.0F, 0.0F}, {0, 0, 1}, {1, 0}},
    {{-0.5F, 1.0F, 0.0F}, {0, 0, 1}, {0, 0}},
}};

constexpr std::array<uint16_t, 12> kQuadIndices{
    0, 1, 2, 2, 3, 0,  // floor
    4, 5, 6, 6, 7, 4,  // upright
};

constexpr uint32_t kFloorFirstIndex = 0;
constexpr uint32_t kUprightFirstIndex = 6;
constexpr uint32_t kQuadIndexCount = 6;

struct Placement {
    glm::vec3 position;  // where its foot sits
    float angle;         // radians about y
    glm::vec2 size;      // width, height
};

constexpr std::array<Placement, 2> kCrates{{
    {{-1.35F, kFloorY, 0.5F}, 0.36F, {1.0F, 1.0F}},
    {{1.30F, kFloorY, -1.20F}, -0.72F, {1.0F, 1.0F}},
}};

// A stand of ferns, deliberately overlapping each other from this camera. Overlap is
// what makes the difference between the cutout and the blended version obvious.
constexpr std::array<Placement, 6> kFerns{{
    {{-3.30F, kFloorY, 1.70F}, 0.30F, {2.0F, 2.0F}},
    {{-2.10F, kFloorY, 2.90F}, -0.55F, {1.7F, 1.7F}},
    {{-2.70F, kFloorY, 2.30F}, 0.95F, {1.5F, 1.5F}},
    {{2.55F, kFloorY, 2.10F}, 0.80F, {2.1F, 2.1F}},
    {{3.40F, kFloorY, 1.30F}, -0.20F, {1.6F, 1.6F}},
    {{0.35F, kFloorY, 3.30F}, 0.15F, {1.8F, 1.8F}},
}};

// Three panes of coloured glass at three different distances, placed so that all three
// overlap on screen. The overlap is the whole experiment: two transparent surfaces that
// never cover the same pixel look identical whatever order they are drawn in.
//
// The order they appear in this array is *not* the order they should be drawn in --
// blue is the farthest away and comes last -- which is what the sort exists to fix.
constexpr std::array<Placement, 3> kPanes{{
    {{-1.15F, kFloorY, 5.10F}, 0.12F, {1.9F, 1.5F}},
    {{0.80F, kFloorY, 6.00F}, -0.20F, {1.9F, 1.5F}},
    {{-0.30F, kFloorY, 4.10F}, 0.04F, {1.9F, 1.5F}},
}};

constexpr std::array<glm::vec4, 3> kPaneColours{{
    {0.88F, 0.16F, 0.20F, 0.55F},
    {0.14F, 0.76F, 0.38F, 0.55F},
    {0.18F, 0.36F, 0.96F, 0.55F},
}};

constexpr glm::vec4 kWhite{1.0F};

constexpr std::array<VkCullModeFlags, 3> kCullModes{
    VK_CULL_MODE_BACK_BIT,
    VK_CULL_MODE_NONE,
    VK_CULL_MODE_FRONT_BIT,
};
constexpr std::array<const char*, 3> kCullModeNames{"back", "none", "front"};

constexpr glm::vec3 kSunDirection{-0.45F, -1.0F, -0.38F};
constexpr glm::vec3 kSunColour{0.92F, 0.88F, 0.80F};
constexpr glm::vec3 kAmbient{0.24F, 0.26F, 0.31F};
constexpr float kShininess = 48.0F;

struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::vec4 view_position;  // 128
    glm::vec4 sun_direction;  // 144
    glm::vec4 sun_colour;     // 160
    glm::vec4 ambient;        // 176, w is the specular exponent
};
static_assert(offsetof(Globals, ambient) == 176);
static_assert(sizeof(Globals) == 192);

// 2.6's push block, still exactly the 128 bytes Vulkan guarantees.
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

[[nodiscard]] glm::mat4 model_of(const Placement& placement) {
    return glm::scale(
        glm::rotate(glm::translate(glm::mat4(1.0F), placement.position), placement.angle,
                    glm::vec3(0.0F, 1.0F, 0.0F)),
        glm::vec3(placement.size.x, placement.size.y, 1.0F));
}

// The middle of the quad, for sorting. Only the horizontal position really matters at
// this camera angle, but the centre is the honest answer and costs nothing.
[[nodiscard]] glm::vec3 centre_of(const Placement& placement) {
    return placement.position + glm::vec3(0.0F, 0.5F * placement.size.y, 0.0F);
}

class BlendingAndCullingApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        cube_vertices_ = vkc::upload_to_device_local(context(), kCubeVertices.data(),
                                                     sizeof(kCubeVertices),
                                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cube_indices_ = vkc::upload_to_device_local(context(), kCubeIndices.data(),
                                                    sizeof(kCubeIndices),
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        quad_vertices_ = vkc::upload_to_device_local(context(), kQuadVertices.data(),
                                                     sizeof(kQuadVertices),
                                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        quad_indices_ = vkc::upload_to_device_local(context(), kQuadIndices.data(),
                                                    sizeof(kQuadIndices),
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        // Back to 1.13's depth-only buffer. 4.1 needed a stencil aspect and wrote its
        // own chooser; nothing in this chapter tests the stencil, so vkcommon's helpers
        // are the right ones again.
        depth_format_ = vkc::choose_depth_format(context().physical_device());
        depth_ = vkc::create_depth_buffer(context(), depth_format_, swapchain().extent());

        load_textures();
        create_uniform_buffers();
        create_descriptors();
        create_pipelines();

        camera_.position = {0.0F, 2.60F, 12.60F};
        camera_.pitch = -14.0F;

        spdlog::info("1 cull mode, 2 pane sorting, 3 pane depth writes, 4 fern mode.");
    }

    void on_resize(VkExtent2D extent) override {
        context().wait_idle();
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
    }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                cull_index_ = (cull_index_ + 1) % kCullModes.size();
                spdlog::info("Cull mode: {}", kCullModeNames[cull_index_]);
                break;
            case SDL_SCANCODE_2:
                sort_panes_ = !sort_panes_;
                spdlog::info("Pane sorting {}", sort_panes_ ? "on" : "off");
                break;
            case SDL_SCANCODE_3:
                pane_depth_writes_ = !pane_depth_writes_;
                spdlog::info("Pane depth writes {}", pane_depth_writes_ ? "on" : "off");
                break;
            case SDL_SCANCODE_4:
                fern_cutout_ = !fern_cutout_;
                spdlog::info("Ferns: {}", fern_cutout_ ? "alpha tested" : "alpha blended");
                break;
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pane_depth_write_pipeline_, nullptr);
        vkDestroyPipeline(device, pane_pipeline_, nullptr);
        vkDestroyPipeline(device, fern_blend_pipeline_, nullptr);
        vkDestroyPipeline(device, fern_cutout_pipeline_, nullptr);
        for (const VkPipeline pipeline : opaque_pipelines_) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, material_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        clamp_sampler_.destroy();
        sampler_.destroy();
        fern_map_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();
        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        depth_.destroy();
        quad_indices_.destroy();
        quad_vertices_.destroy();
        cube_indices_.destroy();
        cube_vertices_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.05F, 0.06F, 0.09F, 1.0F}});

        set_viewport(frame);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        // --- opaque, front to back would be ideal and any order is correct ---------
        //
        // Nothing here blends, so the depth test settles every overlap however these
        // are ordered. That freedom is exactly what the transparent passes below lose.
        const VkPipeline opaque = opaque_pipelines_[cull_index_];

        bind_geometry(frame.cmd, quad_vertices_, quad_indices_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque);
        bind_material(frame.cmd, floor_set_);
        push(frame.cmd, glm::mat4(1.0F), kWhite);
        vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kFloorFirstIndex, 0, 0);

        bind_geometry(frame.cmd, cube_vertices_, cube_indices_);
        bind_material(frame.cmd, crate_set_);
        for (const Placement& crate : kCrates) {
            // The cube is centred on its origin, so it has to be lifted half its height
            // to stand on the floor rather than be buried to the waist in it.
            push(frame.cmd,
                 glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F)), kWhite);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kCubeIndices.size()), 1, 0,
                             0, 0);
        }

        // --- the ferns ------------------------------------------------------------
        bind_geometry(frame.cmd, quad_vertices_, quad_indices_);
        bind_material(frame.cmd, fern_set_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          fern_cutout_ ? fern_cutout_pipeline_ : fern_blend_pipeline_);
        for (const Placement& fern : kFerns) {
            push(frame.cmd, model_of(fern), kWhite);
            vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }

        // --- the glass ------------------------------------------------------------
        //
        // Blending is not commutative: "draw red over green" and "draw green over red"
        // are different pictures. The depth test cannot rescue the order the way it does
        // for opaque geometry, because a pane that is behind another pane still has to
        // be drawn -- and drawn first.
        std::array<size_t, kPanes.size()> order{};
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        if (sort_panes_) {
            const glm::vec3 eye = camera_.position;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                // Squared distance is enough: nothing here needs the actual length, and
                // a sort only ever compares.
                return glm::dot(centre_of(kPanes[a]) - eye, centre_of(kPanes[a]) - eye) >
                       glm::dot(centre_of(kPanes[b]) - eye, centre_of(kPanes[b]) - eye);
            });
        }

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pane_depth_writes_ ? pane_depth_write_pipeline_ : pane_pipeline_);
        for (const size_t index : order) {
            push(frame.cmd, model_of(kPanes[index]), kPaneColours[index]);
            vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

    void push(VkCommandBuffer cmd, const glm::mat4& model, const glm::vec4& colour) {
        const PushConstants constants{
            .model = model,
            .normal_matrix = normal_matrix_of(model),
            .colour = colour,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
    }

    static void bind_geometry(VkCommandBuffer cmd, const vkc::Buffer& vertices,
                              const vkc::Buffer& indices) {
        const VkDeviceSize offset = 0;
        const VkBuffer handle = vertices.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &handle, &offset);
        vkCmdBindIndexBuffer(cmd, indices.handle(), 0, VK_INDEX_TYPE_UINT16);
    }

    void bind_material(VkCommandBuffer cmd, VkDescriptorSet set) const {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 1,
                                1, &set, 0, nullptr);
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
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
            .view_position = glm::vec4(camera_.position, 1.0F),
            .sun_direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
            .sun_colour = glm::vec4(kSunColour, 1.0F),
            .ambient = glm::vec4(kAmbient, kShininess),
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors
    // ---------------------------------------------------------------------------

    void load_textures() {
        crate_map_ = vkc::load_texture(
            context(), vkc::asset_path("textures/lvk_crate_diffuse.png"),
            VK_FORMAT_R8G8B8A8_SRGB);
        // Not 1.11's grid, which every chapter up to here has used. That texture is
        // loud by design -- it exists to make filtering artefacts obvious -- and this
        // chapter needs the eye on the glass and the ferns, not on the floor.
        floor_map_ = vkc::load_texture(context(),
                                       vkc::asset_path("textures/lvk_ground.png"),
                                       VK_FORMAT_R8G8B8A8_SRGB);
        // The fern is the first texture in the series whose alpha carries information.
        // Loading it as _SRGB is still right: the sRGB transfer function applies to the
        // three colour channels only, and alpha is left linear by definition.
        fern_map_ = vkc::load_texture(context(),
                                      vkc::asset_path("textures/lvk_foliage.png"),
                                      VK_FORMAT_R8G8B8A8_SRGB);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});
        // The fern needs CLAMP_TO_EDGE. Its quad's uvs stay inside 0..1, but a sampler
        // filtering near the edge reaches past it, and REPEAT would fetch the opposite
        // side of the leaf and smear a sliver of it around the border.
        clamp_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE});

        context().name(crate_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate diffuse map");
        context().name(floor_map_.handle(), VK_OBJECT_TYPE_IMAGE, "floor grid map");
        context().name(fern_map_.handle(), VK_OBJECT_TYPE_IMAGE, "fern cutout map");
    }

    void create_descriptors() {
        const VkDevice device = context().device();

        const VkDescriptorSetLayoutBinding globals_binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo globals_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &globals_binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &globals_layout_info, nullptr,
                                             &frame_set_layout_));

        const VkDescriptorSetLayoutBinding texture_binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo texture_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &texture_binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &texture_layout_info, nullptr,
                                             &material_set_layout_));

        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + 3,
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

        for (size_t i = 0; i < vkc::kFramesInFlight; ++i) {
            const VkDescriptorBufferInfo buffer_info{
                .buffer = uniform_buffers_[i].handle(),
                .offset = 0,
                .range = sizeof(Globals),
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame_sets_[i],
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

        crate_set_ = allocate_material_set(crate_map_, sampler_);
        floor_set_ = allocate_material_set(floor_map_, sampler_);
        fern_set_ = allocate_material_set(fern_map_, clamp_sampler_);

        context().name(crate_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "crate material");
        context().name(floor_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "floor material");
        context().name(fern_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "fern material");
    }

    [[nodiscard]] VkDescriptorSet allocate_material_set(const vkc::Image& texture,
                                                        const vkc::Sampler& sampler) {
        const VkDevice device = context().device();

        const VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &material_set_layout_,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc, &set));

        const VkDescriptorImageInfo image_info{
            .sampler = sampler.handle(),
            .imageView = texture.view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        return set;
    }

    // ---------------------------------------------------------------------------
    // Pipelines -- where this chapter actually happens
    // ---------------------------------------------------------------------------

    void create_pipelines() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };
        const std::array<VkVertexInputAttributeDescription, 3> attributes{{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        }};

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
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "blending_and_culling.slang");

        // Spelled-out return type, for 4.1's reason: the setters hand back a reference to
        // the builder, and `auto` here would deduce a reference to a dead temporary.
        const auto base = [&]() -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(shader)
                .vertex_input(std::span(&binding, 1), attributes)
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_);
        };

        // Source-over blending, written out in full because this is the chapter that
        // owes an explanation for every field.
        //
        //   result.rgb = src.rgb * src.a  +  dst.rgb * (1 - src.a)
        //   result.a   = src.a   * 0      +  dst.a   * 1
        //
        // The colour line is a linear interpolation: at src.a = 1 the new fragment wins
        // outright, at 0 it contributes nothing.
        //
        // The alpha line is a *separate* equation with its own two factors, and the
        // obvious-looking choice for it -- ONE and ZERO, "store the incoming alpha" --
        // is wrong here. It would stamp each fern's own transparency into the
        // attachment, leaving the rendered image itself partly transparent: a screenshot
        // of this scene came back with alpha 0 across every empty texel of every fern
        // quad, even though the colours were exactly right. ZERO and ONE keep whatever
        // alpha the attachment already had, which for an opaque target cleared to 1 is
        // what is wanted.
        const VkPipelineColorBlendAttachmentState source_over{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            // Which channels the result is allowed to touch. All four, here. Turning one
            // off is how a depth-only or stencil-only pass keeps a pipeline that still
            // has a colour attachment from writing to it.
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        // The opaque geometry, three times over: one pipeline per cull mode.
        //
        // Culling is decided from the triangle's winding *after* projection, in
        // framebuffer coordinates -- which is why frontFace and the projection matrix
        // are entangled. This series' build_projection() negates projection[1][1] to
        // put Vulkan's clip-space y where glm expects it, and negating one axis mirrors
        // the image, which reverses the apparent winding of every triangle on screen.
        for (size_t i = 0; i < kCullModes.size(); ++i) {
            opaque_pipelines_[i] =
                base()
                    .cull(kCullModes[i], VK_FRONT_FACE_COUNTER_CLOCKWISE)
                    .depth(depth_format_, /*test=*/true, /*write=*/true)
                    .build();
        }

        // The ferns as a cutout. Opaque in every respect the pipeline can see: depth
        // test on, depth writes on, blending off. The transparency is entirely the
        // fragment shader's `discard`, and the pipeline does not know it is happening.
        //
        // Culling is off, and has to be: a quad has one triangle pair and no inside, so
        // whichever way it is wound, half the ferns in this scene are seen from behind.
        const auto quad_base = [&]() -> vkc::PipelineBuilder {
            return base().cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        };

        fern_cutout_pipeline_ =
            quad_base()
                .shaders(shader, "vertexMain", "cutoutFragmentMain")
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .build();

        // The same ferns, blended. Two changes, and they travel together:
        //
        //   blending on, so the sampled alpha is used rather than thresholded
        //   depth *writes* off, so a fern does not stamp its whole rectangle -- empty
        //   texels included -- into the depth buffer and hide the ferns behind it
        //
        // Depth *testing* stays on: a transparent surface is still occluded by the
        // opaque floor and crates in front of it.
        fern_blend_pipeline_ =
            quad_base()
                .shaders(shader, "vertexMain", "foliageBlendFragmentMain")
                .depth(depth_format_, /*test=*/true, /*write=*/false)
                .colour_blend(source_over)
                .build();

        // The glass. Same state as the blended ferns, a different fragment shader.
        pane_pipeline_ = quad_base()
                             .shaders(shader, "vertexMain", "flatFragmentMain")
                             .depth(depth_format_, /*test=*/true, /*write=*/false)
                             .colour_blend(source_over)
                             .build();

        // And the same thing with depth writes back on, which is the mistake this
        // chapter wants to be able to show rather than only describe.
        pane_depth_write_pipeline_ =
            quad_base()
                .shaders(shader, "vertexMain", "flatFragmentMain")
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .colour_blend(source_over)
                .build();

        vkDestroyShaderModule(device, shader, nullptr);

        for (size_t i = 0; i < opaque_pipelines_.size(); ++i) {
            context().name(opaque_pipelines_[i], VK_OBJECT_TYPE_PIPELINE,
                           kCullModeNames[i]);
        }
        context().name(fern_cutout_pipeline_, VK_OBJECT_TYPE_PIPELINE, "fern (cutout)");
        context().name(fern_blend_pipeline_, VK_OBJECT_TYPE_PIPELINE, "fern (blended)");
        context().name(pane_pipeline_, VK_OBJECT_TYPE_PIPELINE, "glass");
        context().name(pane_depth_write_pipeline_, VK_OBJECT_TYPE_PIPELINE,
                       "glass (depth writes)");
    }

    vkc::Camera camera_;

    vkc::Buffer cube_vertices_;
    vkc::Buffer cube_indices_;
    vkc::Buffer quad_vertices_;
    vkc::Buffer quad_indices_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    vkc::Image crate_map_;
    vkc::Image floor_map_;
    vkc::Image fern_map_;
    vkc::Sampler sampler_;
    vkc::Sampler clamp_sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fern_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout material_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    std::array<VkPipeline, 3> opaque_pipelines_{};
    VkPipeline fern_cutout_pipeline_ = VK_NULL_HANDLE;
    VkPipeline fern_blend_pipeline_ = VK_NULL_HANDLE;
    VkPipeline pane_pipeline_ = VK_NULL_HANDLE;
    VkPipeline pane_depth_write_pipeline_ = VK_NULL_HANDLE;

    size_t cull_index_ = 0;
    bool sort_panes_ = true;
    bool pane_depth_writes_ = false;
    bool fern_cutout_ = true;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.2 - Blending & Culling";
        BlendingAndCullingApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
