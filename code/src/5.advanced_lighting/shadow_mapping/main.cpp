// 5.3 Shadow mapping
//
// 5.2's room by daylight, lit by a low sun, and the crates casting shadows for the first
// time in the series. Each frame now renders the scene twice: once from the sun, keeping
// nothing but depth, and once from the camera, where every fragment asks that depth
// image whether anything stood between it and the sun.
//
//   1  shadows on or off
//   2  the shadow pass's slope-scaled depth bias: 0, 1, 2, 4 or 8
//   3  how the shadow map is read: one texel, bilinear, or 3x3 bilinear taps
//   4  show the shadow map itself in the corner of the window

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 4.4's vertex, unchanged -- and now read by the shader rather than fed to it. The
// layout matters in a way it never did before: a vertex attribute description used to
// translate between these bytes and the shader's idea of them, and there is no
// translation layer left. See the note on `struct Vertex` in scene.slang.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32);

// 2.4's cube, which is the crates.
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
constexpr float kFloorHalf = 22.0F;
constexpr float kFloorTiles = 19.0F;

constexpr std::array<Vertex, 4> kFloorVertices{{
    {{-kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {0, 0}},
    {{-kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {0, kFloorTiles}},
    {{kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {kFloorTiles, kFloorTiles}},
    {{kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {kFloorTiles, 0}},
}};

constexpr std::array<uint16_t, 6> kFloorIndices{0, 1, 2, 2, 3, 0};

struct Placement {
    glm::vec3 position;
    float angle;
    glm::vec2 size;  // width in x and z, height in y
};

// 5.2's four crates, a small one stacked on the first, and a tall thin post, which
// casts the longest shadow in the room.
constexpr std::array<Placement, 6> kCrates{{
    {{-2.20F, kFloorY, 0.75F}, 0.36F, {1.0F, 1.0F}},
    {{2.05F, kFloorY, -1.05F}, -0.72F, {1.0F, 1.0F}},
    {{-1.35F, kFloorY, -3.40F}, 0.18F, {1.0F, 1.0F}},
    {{3.10F, kFloorY, 1.60F}, 0.95F, {1.0F, 1.0F}},
    {{-2.15F, kFloorY + 1.0F, 0.70F}, 1.10F, {0.55F, 0.55F}},
    {{0.35F, kFloorY, -1.90F}, 0.30F, {0.30F, 2.60F}},
}};

constexpr glm::vec4 kWhite{1.0F};

// The sun. A direction rather than a position: it is far enough away that every ray from
// it into this room is parallel to every other. This vector points *towards* the sun,
// which is what the lighting wants; it sits 30 degrees above the horizon, behind the
// room and to its left, so shadows fall long, to the right and towards the camera.
const glm::vec3 kSunDirection = glm::normalize(glm::vec3(-0.80F, 0.50F, -0.33F));
constexpr glm::vec3 kSunColour{1.0F, 0.92F, 0.80F};

// Daylight from the rest of the sky: the only light a shadowed surface gets.
constexpr glm::vec3 kAmbient{0.10F, 0.12F, 0.16F};

// ---------------------------------------------------------------------------
// The shadow map
// ---------------------------------------------------------------------------

// Texels along each side. With the box fitted around the crates, as below, one texel
// covers about 6 mm of the room.
constexpr uint32_t kShadowMapSize = 1024;

// The box the sun's camera sees: looked at from kSunDistance away along kSunDirection,
// towards kShadowCentre, keeping depth between kShadowNear and kShadowFar. Its width is
// not a constant -- sun_view_projection() fits it around the crates -- and the article
// works out why holding every crate is enough.
constexpr glm::vec3 kShadowCentre{0.30F, kFloorY, -0.80F};
constexpr float kSunDistance = 15.0F;
constexpr float kShadowNear = 1.0F;
constexpr float kShadowFar = 30.0F;

// A little more than the crates need, so that the 3x3 reads around a crate's outermost
// edge stay inside the map.
constexpr float kShadowMargin = 1.02F;

// The slope-scaled depth bias the shadow pass can use, in units the article derives:
// 1 moves each surface back by the depth it changes across one shadow-map texel.
constexpr std::array<float, 5> kSlopeBiasSteps{0.0F, 1.0F, 2.0F, 4.0F, 8.0F};
constexpr size_t kDefaultSlopeBias = 3;  // 4

// How the fragment shader reads the shadow map. The values are Globals::filter.
enum class ShadowFilter : uint32_t { OneTexel = 0, Bilinear = 1, Pcf3x3 = 2 };

constexpr std::array<const char*, 3> kFilterNames{"one texel", "bilinear",
                                                  "3x3 bilinear taps"};

// ---------------------------------------------------------------------------
// What the shaders read through a pointer
// ---------------------------------------------------------------------------

// The bits of Globals::flags.
constexpr uint32_t kShadowsOn = 1U << 0;

// 5.2's Globals with the lamps replaced by the sun, and what the fragment shader needs to
// find itself in the shadow map: the matrix the shadow pass drew with, the size of one
// texel, and which way to read it.
struct Globals {
    glm::mat4 view;                    // 0
    glm::mat4 projection;              // 64
    glm::mat4 light_view_projection;   // 128
    glm::vec4 view_position;           // 192
    glm::vec4 ambient;                 // 208
    glm::vec4 sun_direction;           // 224, towards the sun
    glm::vec4 sun_colour;              // 240
    float shadow_texel;                // 256, one texel in uv units
    uint32_t flags;                    // 260
    uint32_t filter;                   // 264
};
static_assert(offsetof(Globals, light_view_projection) == 128);
static_assert(offsetof(Globals, shadow_texel) == 256);
static_assert(sizeof(Globals) == 268);

// 4.5's per-object data, unchanged. The article there explains the 116 bytes and why
// the normal matrix needs no padding behind a pointer.
struct DrawData {
    glm::mat4 model;           // 0
    glm::mat3 normal_matrix;   // 64
    glm::vec4 colour;          // 100
};
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, colour) == 100);
static_assert(sizeof(DrawData) == 116);

// 4.5's push constants, unchanged: three addresses and an index.
struct PushConstants {
    VkDeviceAddress globals;   // 0
    VkDeviceAddress draws;     // 8
    VkDeviceAddress vertices;  // 16
    uint32_t draw_index;       // 24
};
static_assert(offsetof(PushConstants, draw_index) == 24);
static_assert(sizeof(PushConstants) == 32);

// Where each object's DrawData sits in the array the frame writes. The order is fixed by
// write_draws() below and nothing else may reorder it.
constexpr uint32_t kFloorDraw = 0;
constexpr uint32_t kFirstCrateDraw = kFloorDraw + 1;
constexpr uint32_t kDrawCount = kFirstCrateDraw + static_cast<uint32_t>(kCrates.size());

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

// 4.5's, as are FrameArena and upload_mesh below.
//
// The address is a property of the buffer and does not change for as long as the buffer
// lives, so this is worth calling once and keeping rather than calling per frame. It is
// also legal to call only because the buffer asked for
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT at creation and the device was created with
// VkPhysicalDeviceVulkan12Features::bufferDeviceAddress -- the validation layers say so
// plainly if either is missing, which is the best possible outcome for a feature whose
// failure mode is otherwise a wild pointer.
[[nodiscard]] VkDeviceAddress buffer_address(VkDevice device, VkBuffer buffer) {
    const VkBufferDeviceAddressInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &info);
}

// One frame's scratch memory: a mapped buffer and a cursor into it.
//
// Every frame writes its globals and its per-object data here from offset zero and hands
// the shaders addresses inside it. There is one of these per frame in flight and they
// are never synchronised against each other, because the frame that owns a slot has
// already been waited on before it is reset -- the same argument that lets each frame
// reuse its command buffer.
//
// Host-visible rather than device-local-plus-staging on purpose. This data is written
// once by the CPU and read a handful of times by the GPU in the same frame; a copy would
// cost a transfer, a barrier and a second allocation to save reads that are not the
// bottleneck. Data that is read thousands of times, like the vertices below, is worth
// staging. Data that changes every frame is usually not.
class FrameArena {
public:
    void create(vkc::Context& context, VkDeviceSize bytes) {
        buffer_ = vkc::Buffer(context.allocator(), bytes,
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT);
        base_ = buffer_address(context.device(), buffer_.handle());
        capacity_ = bytes;
        used_ = 0;
    }

    void destroy() noexcept { buffer_.destroy(); }

    void reset() noexcept { used_ = 0; }

    // Copies `bytes` into the arena and returns the address they landed at.
    //
    // Alignment is 16 because that is the alignment of the widest thing written here and
    // costs nothing at this scale. The requirement is much weaker -- Slang emits loads
    // decorated Aligned 4 for these structs -- but "the alignment of the type" is the
    // rule that keeps being true when the types change.
    [[nodiscard]] VkDeviceAddress write(const void* data, VkDeviceSize bytes) {
        constexpr VkDeviceSize kAlignment = 16;
        const VkDeviceSize offset = (used_ + kAlignment - 1) & ~(kAlignment - 1);
        if (offset + bytes > capacity_) {
            throw std::runtime_error(
                std::format("Frame arena is full: {} bytes used, {} more wanted, {} "
                            "total. Raise kArenaBytes.",
                            offset, bytes, capacity_));
        }
        std::memcpy(static_cast<std::byte*>(buffer_.mapped()) + offset, data,
                    static_cast<size_t>(bytes));
        used_ = offset + bytes;
        // Pointer arithmetic on the GPU's address space, done on the CPU. This is the
        // entire suballocator: there is no Vulkan object for a range within a buffer,
        // and there does not need to be one.
        return base_ + offset;
    }

private:
    vkc::Buffer buffer_;
    VkDeviceAddress base_ = 0;
    VkDeviceSize used_ = 0;
    VkDeviceSize capacity_ = 0;
};

// 64 KiB, against the under 1 KiB a frame actually uses. Arena sizing is a guess that
// wants to be generous: too small throws above, too large costs address space nobody is
// competing for.
constexpr VkDeviceSize kArenaBytes = 64 * 1024;

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

// A mesh is now a pointer and an index buffer.
//
// The index buffer is still bound with vkCmdBindIndexBuffer, and that is not an
// oversight: index fetch is fixed-function hardware that reads the buffer before any
// shader runs, so it has to be told through the command buffer. Vertex fetch used to be
// fixed-function too -- that is what a vertex attribute description configured -- and
// here it is not fixed-function any more. It is four lines of shader.
struct Mesh {
    vkc::Buffer vertices;
    vkc::Buffer indices;
    VkDeviceAddress vertex_address = 0;
    uint32_t index_count = 0;

    void destroy() noexcept {
        indices.destroy();
        vertices.destroy();
    }
};

[[nodiscard]] Mesh upload_mesh(vkc::Context& context, std::span<const Vertex> vertices,
                               std::span<const uint16_t> indices) {
    Mesh mesh;
    // No VK_BUFFER_USAGE_VERTEX_BUFFER_BIT. This buffer is never bound as a vertex
    // buffer, so claiming it can be is a promise the sample does not keep; the usage
    // flags are what the driver picks memory and alignment from, and they should say
    // what is true. SHADER_DEVICE_ADDRESS is what it is for now.
    mesh.vertices =
        vkc::upload_to_device_local(context, vertices.data(), vertices.size_bytes(),
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    mesh.indices = vkc::upload_to_device_local(context, indices.data(),
                                               indices.size_bytes(),
                                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.vertex_address = buffer_address(context.device(), mesh.vertices.handle());
    mesh.index_count = static_cast<uint32_t>(indices.size());
    return mesh;
}

[[nodiscard]] glm::mat4 model_of(const Placement& placement) {
    return glm::scale(
        glm::rotate(glm::translate(glm::mat4(1.0F), placement.position), placement.angle,
                    glm::vec3(0.0F, 1.0F, 0.0F)),
        glm::vec3(placement.size.x, placement.size.y, placement.size.x));
}

[[nodiscard]] DrawData draw_data_of(const glm::mat4& model, const glm::vec4& colour) {
    return DrawData{
        .model = model,
        .normal_matrix = glm::inverseTranspose(glm::mat3(model)),
        .colour = colour,
    };
}

// Each crate's model matrix: the placement, lifted by half a unit so that the cube's
// base, not its centre, sits at the placement's height.
[[nodiscard]] glm::mat4 crate_model(const Placement& crate) {
    return glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F));
}

// The sun's camera: a view matrix that looks along the sun's rays at the middle of the
// room, and an orthographic projection, because parallel rays need no perspective.
[[nodiscard]] glm::mat4 sun_view_projection() {
    const glm::vec3 eye = kShadowCentre + kSunDirection * kSunDistance;
    const glm::mat4 view = glm::lookAt(eye, kShadowCentre, glm::vec3(0.0F, 1.0F, 0.0F));

    // The smallest square around the sun's line of sight that holds every corner of
    // every crate. In view space the sun looks down -z, so x and y are the two
    // directions across the map; depth plays no part in the fit.
    float half_width = 0.0F;
    for (const Placement& crate : kCrates) {
        const glm::mat4 model_view = view * crate_model(crate);
        for (const Vertex& vertex : kCubeVertices) {
            const glm::vec4 corner = model_view * glm::vec4(vertex.position, 1.0F);
            half_width = std::max({half_width, std::abs(corner.x), std::abs(corner.y)});
        }
    }
    half_width *= kShadowMargin;

    glm::mat4 projection = glm::ortho(-half_width, half_width, -half_width, half_width,
                                      kShadowNear, kShadowFar);
    // The y flip is the same one every camera in this series has made, and here it is
    // not needed for the lookup to agree with the pass: both use this one matrix,
    // whichever way up the image is stored. It is there so that triangles keep the
    // winding the cull mode expects, and so that key 4 shows the map the right way up.
    projection[1][1] *= -1.0F;
    return projection * view;
}

class ShadowMappingApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        cube_ = upload_mesh(context(), kCubeVertices, kCubeIndices);
        floor_ = upload_mesh(context(), kFloorVertices, kFloorIndices);

        depth_format_ = vkc::choose_depth_format(context().physical_device());

        load_textures();
        create_shadow_map();
        create_arenas();
        create_descriptors();
        create_depth_buffer(swapchain().extent());
        create_pipelines();

        draws_.reserve(kDrawCount);

        camera_.position = {0.0F, 0.35F, 6.40F};
        camera_.pitch = -9.0F;

        spdlog::info("1 shadows, 2 slope bias, 3 filter, 4 show the shadow map.");
        log_mode();
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                shadows_ = !shadows_;
                log_mode();
                break;
            case SDL_SCANCODE_2:
                slope_bias_ = (slope_bias_ + 1) % kSlopeBiasSteps.size();
                log_mode();
                break;
            case SDL_SCANCODE_3:
                filter_ = static_cast<ShadowFilter>(
                    (static_cast<uint32_t>(filter_) + 1) % kFilterNames.size());
                log_mode();
                break;
            case SDL_SCANCODE_4:
                show_map_ = !show_map_;
                log_mode();
                break;
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, inset_pipeline_, nullptr);
        for (VkPipeline pipeline : shadow_pipelines_) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyPipeline(device, opaque_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, shadow_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, texture_set_layout_, nullptr);

        vkDestroySampler(device, shadow_linear_sampler_, nullptr);
        vkDestroySampler(device, shadow_nearest_sampler_, nullptr);
        depth_sampler_.destroy();
        shadow_map_.destroy();

        sampler_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();

        for (FrameArena& arena : arenas_) {
            arena.destroy();
        }
        depth_.destroy();
        floor_.destroy();
        cube_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        // Everything the shaders will read this frame is written here, into this frame
        // slot's arena, in the order the frame happens to want it. The arena is emptied
        // first: the GPU finished with the last frame that used this slot before the
        // command buffer was handed over, so there is nothing left in it to protect.
        FrameArena& arena = arenas_[frame.frame_number % vkc::kFramesInFlight];
        arena.reset();

        write_globals(arena, frame.extent);
        write_draws(arena);

        render_shadow_map(frame.cmd);
        render_scene(frame);
    }

private:
    // ---------------------------------------------------------------------------
    // Pass one: depth from the sun
    // ---------------------------------------------------------------------------

    void render_shadow_map(VkCommandBuffer cmd) {
        // Into DEPTH_ATTACHMENT_OPTIMAL, from nothing: the pass clears the whole map, so
        // last frame's depths are not worth keeping. The source scope is last frame's
        // reader, as in 4.3 -- there is one shadow map and two frames in flight, and the
        // previous frame's fragment shaders may still be sampling it.
        vkc::image_barrier(cmd, shadow_map_.handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);

        // A rendering instance with a depth attachment and no colour attachment at all.
        // The depth is STOREd this time: unlike the camera's depth buffer, which nothing
        // reads once the frame is drawn, this one is the whole point of the pass.
        const VkRenderingAttachmentInfo depth_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = shadow_map_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.depthStencil = {.depth = 1.0F, .stencil = 0}},
        };
        const VkExtent2D extent{kShadowMapSize, kShadowMapSize};
        const VkRenderingInfo rendering{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {.offset = {0, 0}, .extent = extent},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthAttachment = &depth_attachment,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &rendering);
        set_viewport(cmd, extent);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          shadow_pipelines_[slope_bias_]);
        draw_geometry(cmd);

        vkCmdEndRendering(cmd);

        // From attachment to texture. Depth is written by the fragment tests, early or
        // late, and the last of those writes has landed once LATE_FRAGMENT_TESTS is
        // done; the reader is the camera pass's fragment shader.
        vkc::image_barrier(cmd, shadow_map_.handle(),
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // ---------------------------------------------------------------------------
    // Pass two: the scene from the camera
    // ---------------------------------------------------------------------------

    void render_scene(const vkc::FrameInfo& frame) {
        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.30F, 0.42F, 0.62F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        const VkDescriptorSet shadow_set =
            filter_ == ShadowFilter::OneTexel ? shadow_nearest_set_ : shadow_linear_set_;
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 1, 1, &shadow_set, 0, nullptr);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque_pipeline_);
        draw_geometry(frame.cmd);

        if (show_map_) {
            draw_inset(frame.cmd, frame.extent);
        }

        vkCmdEndRendering(frame.cmd);
    }

    // The floor and every crate, with whatever pipeline is bound. Both passes draw
    // exactly this; only the matrix and the fragment work differ.
    void draw_geometry(VkCommandBuffer cmd) {
        bind_indices(cmd, floor_);
        bind_texture(cmd, floor_set_);
        push(cmd, floor_.vertex_address, kFloorDraw);
        vkCmdDrawIndexed(cmd, floor_.index_count, 1, 0, 0, 0);

        bind_indices(cmd, cube_);
        bind_texture(cmd, crate_set_);
        for (uint32_t i = 0; i < kCrates.size(); ++i) {
            push(cmd, cube_.vertex_address, kFirstCrateDraw + i);
            vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
        }
    }

    // The shadow map drawn into a square in the bottom-left corner: one triangle that
    // covers the viewport, with the viewport made small.
    void draw_inset(VkCommandBuffer cmd, VkExtent2D extent) {
        const float side = static_cast<float>(extent.height) * 0.4F;
        const VkViewport viewport{
            .x = 0.0F,
            .y = static_cast<float>(extent.height) - side,
            .width = side,
            .height = side,
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, inset_pipeline_);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // ---------------------------------------------------------------------------
    // Filling the arena
    // ---------------------------------------------------------------------------

    void write_globals(FrameArena& arena, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = projection(extent),
            .light_view_projection = sun_view_projection(),
            .view_position = glm::vec4(camera_.position, 1.0F),
            .ambient = glm::vec4(kAmbient, 1.0F),
            .sun_direction = glm::vec4(kSunDirection, 0.0F),
            .sun_colour = glm::vec4(kSunColour, 1.0F),
            .shadow_texel = 1.0F / static_cast<float>(kShadowMapSize),
            .flags = shadows_ ? kShadowsOn : 0U,
            .filter = static_cast<uint32_t>(filter_),
        };
        globals_address_ = arena.write(&globals, sizeof(globals));
    }

    void write_draws(FrameArena& arena) {
        draws_.clear();

        draws_.push_back(draw_data_of(glm::mat4(1.0F), kWhite));

        for (const Placement& crate : kCrates) {
            draws_.push_back(draw_data_of(crate_model(crate), kWhite));
        }

        draws_address_ = arena.write(draws_.data(), draws_.size() * sizeof(DrawData));
    }

    void log_mode() const {
        spdlog::info("shadows {}, slope bias {}, {}{}", shadows_ ? "on" : "off",
                     kSlopeBiasSteps[slope_bias_],
                     kFilterNames[static_cast<uint32_t>(filter_)],
                     show_map_ ? ", map shown" : "");
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
    // ---------------------------------------------------------------------------

    void push(VkCommandBuffer cmd, VkDeviceAddress vertices, uint32_t draw_index) {
        const PushConstants constants{
            .globals = globals_address_,
            .draws = draws_address_,
            .vertices = vertices,
            .draw_index = draw_index,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
    }

    // Only the indices. There is no vkCmdBindVertexBuffers call anywhere in this file.
    static void bind_indices(VkCommandBuffer cmd, const Mesh& mesh) {
        vkCmdBindIndexBuffer(cmd, mesh.indices.handle(), 0, VK_INDEX_TYPE_UINT16);
    }

    void bind_texture(VkCommandBuffer cmd, VkDescriptorSet set) const {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0,
                                1, &set, 0, nullptr);
    }

    static void set_viewport(VkCommandBuffer cmd, VkExtent2D extent) {
        const VkViewport viewport{
            .x = 0.0F,
            .y = 0.0F,
            .width = static_cast<float>(extent.width),
            .height = static_cast<float>(extent.height),
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        const VkRect2D scissor{.offset = {0, 0}, .extent = extent};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    [[nodiscard]] static glm::mat4 projection(VkExtent2D extent) {
        const float aspect =
            static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 projection =
            glm::perspective(glm::radians(45.0F), aspect, 0.1F, 100.0F);
        projection[1][1] *= -1.0F;  // Vulkan's clip-space y points down.
        return projection;
    }

    void create_depth_buffer(VkExtent2D extent) {
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
    }

    void create_arenas() {
        for (FrameArena& arena : arenas_) {
            arena.create(context(), kArenaBytes);
        }
    }

    // ---------------------------------------------------------------------------
    // Resources
    // ---------------------------------------------------------------------------

    void load_textures() {
        crate_map_ = vkc::load_texture(
            context(), vkc::asset_path("textures/lvk_crate_diffuse.png"),
            VK_FORMAT_R8G8B8A8_SRGB);
        floor_map_ = vkc::load_texture(context(),
                                       vkc::asset_path("textures/lvk_ground.png"),
                                       VK_FORMAT_R8G8B8A8_SRGB);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});
    }

    // A depth image that is also a texture, and the three samplers that read it.
    void create_shadow_map() {
        // D32_SFLOAT, asked for by name rather than through choose_depth_format(): the
        // map has to be sampled as well as rendered into, which is a second question
        // about the format, and filtered, which is a third. Not every GPU promises the
        // third, so ask.
        constexpr VkFormat kShadowFormat = VK_FORMAT_D32_SFLOAT;
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(context().physical_device(), kShadowFormat,
                                            &properties);
        const VkFormatFeatureFlags features = properties.optimalTilingFeatures;
        constexpr VkFormatFeatureFlags kRequired =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((features & kRequired) != kRequired) {
            throw std::runtime_error(
                "D32_SFLOAT cannot be both rendered into and sampled on this GPU");
        }
        const bool filterable =
            (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
        if (!filterable) {
            spdlog::warn("D32_SFLOAT cannot be filtered here; bilinear reads fall back "
                         "to one texel");
        }

        shadow_map_ = vkc::Image(
            context(),
            vkc::ImageDesc{
                .format = kShadowFormat,
                .extent = {kShadowMapSize, kShadowMapSize},
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
                .mip_levels = 1,
                .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
            });
        context().name(shadow_map_.handle(), VK_OBJECT_TYPE_IMAGE, "shadow map");

        shadow_nearest_sampler_ = create_shadow_sampler(VK_FILTER_NEAREST);
        shadow_linear_sampler_ =
            create_shadow_sampler(filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);

        // A plain sampler as well, for key 4's picture of the map: this one returns the
        // stored depth itself rather than the result of comparing against it.
        depth_sampler_ = vkc::Sampler(
            context(), vkc::SamplerDesc{
                           .mag_filter = VK_FILTER_NEAREST,
                           .min_filter = VK_FILTER_NEAREST,
                           .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                           .address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                           .anisotropy = false,
                       });
    }

    // A comparison sampler, written out because SamplerDesc has no field for any of the
    // three things that make it one.
    [[nodiscard]] VkSampler create_shadow_sampler(VkFilter filter) {
        const VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            // Outside the map, the border: depth 1.0, the far plane, which nothing is
            // behind. A fragment the sun's camera never saw is lit.
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .mipLodBias = 0.0F,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0F,
            // The sampler compares the depth the shader passes in against each texel it
            // reads, and returns 1 where the comparison holds and 0 where it fails,
            // filtered: the fraction of texels that passed. LESS_OR_EQUAL holds when the
            // fragment is no further from the sun than what the sun saw.
            .compareEnable = VK_TRUE,
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .minLod = 0.0F,
            .maxLod = 0.0F,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
            .unnormalizedCoordinates = VK_FALSE,
        };
        VkSampler sampler = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSampler(context().device(), &info, nullptr, &sampler));
        return sampler;
    }

    void create_descriptors() {
        const VkDevice device = context().device();

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
                                             &texture_set_layout_));

        // Set 1 is the shadow map, twice: through a comparison sampler at binding 0 for
        // the lighting, and through a plain one at binding 1 for the inset.
        const std::array<VkDescriptorSetLayoutBinding, 2> shadow_bindings{{
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
        }};
        const VkDescriptorSetLayoutCreateInfo shadow_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(shadow_bindings.size()),
            .pBindings = shadow_bindings.data(),
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr,
                                             &shadow_set_layout_));

        // Two texture sets, and two shadow sets of two descriptors each: one with the
        // nearest comparison sampler, one with the linear.
        constexpr uint32_t kSets = 4;
        constexpr uint32_t kDescriptors = 2 + 2 * 2;
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                             kDescriptors};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = kSets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        crate_set_ = allocate_set(texture_set_layout_);
        write_image(crate_set_, 0, crate_map_.view(), sampler_.handle());
        floor_set_ = allocate_set(texture_set_layout_);
        write_image(floor_set_, 0, floor_map_.view(), sampler_.handle());

        shadow_nearest_set_ = allocate_set(shadow_set_layout_);
        write_image(shadow_nearest_set_, 0, shadow_map_.view(), shadow_nearest_sampler_);
        write_image(shadow_nearest_set_, 1, shadow_map_.view(), depth_sampler_.handle());
        shadow_linear_set_ = allocate_set(shadow_set_layout_);
        write_image(shadow_linear_set_, 0, shadow_map_.view(), shadow_linear_sampler_);
        write_image(shadow_linear_set_, 1, shadow_map_.view(), depth_sampler_.handle());
    }

    [[nodiscard]] VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) {
        const VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(context().device(), &alloc, &set));
        return set;
    }

    // The layout is the one the image will be in whenever a shader reads it. For the
    // shadow map that is true only after the barrier at the end of pass one, which is
    // the only time the camera pass reads it.
    void write_image(VkDescriptorSet set, uint32_t binding, VkImageView view,
                     VkSampler sampler) {
        const VkDescriptorImageInfo image_info{
            .sampler = sampler,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(context().device(), 1, &write, 0, nullptr);
    }

    // ---------------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------------

    void create_pipelines() {
        const VkDevice device = context().device();

        // No vertex_input() call on any builder below, so every pipeline is created with
        // zero bindings and zero attributes. That state used to configure the hardware
        // that fetched vertices; there is nothing left for it to configure.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        const std::array<VkDescriptorSetLayout, 2> set_layouts{texture_set_layout_,
                                                               shadow_set_layout_};
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

        const VkShaderModule scene_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "scene.slang");

        opaque_pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(scene_shader, "vertexMain", "fragmentMain")
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .layout(pipeline_layout_)
                .build();

        // The shadow pass: a vertex shader and nothing else, into a depth attachment
        // and nothing else. One pipeline per bias step, because depth bias is baked
        // into the pipeline unless it is declared dynamic.
        for (size_t i = 0; i < kSlopeBiasSteps.size(); ++i) {
            vkc::PipelineBuilder builder(device);
            builder.shaders(scene_shader, "shadowVertexMain", "")
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(shadow_map_.format(), /*test=*/true, /*write=*/true)
                .layout(pipeline_layout_);
            if (kSlopeBiasSteps[i] > 0.0F) {
                builder.depth_bias(/*constant_factor=*/0.0F, kSlopeBiasSteps[i]);
            }
            shadow_pipelines_[i] = builder.build();
            const std::string label =
                std::format("shadow, slope bias {}", kSlopeBiasSteps[i]);
            context().name(shadow_pipelines_[i], VK_OBJECT_TYPE_PIPELINE, label.c_str());
        }

        // Drawn last, over the scene, so it neither tests nor writes depth -- but the
        // rendering instance has a depth attachment, so the pipeline still has to say
        // which format it is.
        inset_pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(scene_shader, "insetVertexMain", "insetFragmentMain")
                .colour_attachment(swapchain().format())
                .depth(depth_format_, /*test=*/false, /*write=*/false)
                .layout(pipeline_layout_)
                .build();

        vkDestroyShaderModule(device, scene_shader, nullptr);

        context().name(opaque_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene opaque");
        context().name(inset_pipeline_, VK_OBJECT_TYPE_PIPELINE, "shadow map inset");
    }

    vkc::Camera camera_;

    Mesh cube_;
    Mesh floor_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    // One arena per frame in flight, and the addresses this frame wrote into it.
    std::array<FrameArena, vkc::kFramesInFlight> arenas_;
    VkDeviceAddress globals_address_ = 0;
    VkDeviceAddress draws_address_ = 0;
    std::vector<DrawData> draws_;

    vkc::Image crate_map_;
    vkc::Image floor_map_;
    vkc::Sampler sampler_;

    vkc::Image shadow_map_;
    VkSampler shadow_nearest_sampler_ = VK_NULL_HANDLE;
    VkSampler shadow_linear_sampler_ = VK_NULL_HANDLE;
    vkc::Sampler depth_sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shadow_nearest_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shadow_linear_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadow_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline opaque_pipeline_ = VK_NULL_HANDLE;
    std::array<VkPipeline, kSlopeBiasSteps.size()> shadow_pipelines_{};
    VkPipeline inset_pipeline_ = VK_NULL_HANDLE;

    bool shadows_ = true;
    size_t slope_bias_ = kDefaultSlopeBias;
    ShadowFilter filter_ = ShadowFilter::Pcf3x3;
    bool show_map_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 5.3 - Shadow mapping";
        ShadowMappingApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
