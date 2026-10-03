// 5.4 Point shadows
//
// 5.2's three lamps, back in the room 5.3 lit with the sun, and now casting shadows. A
// lamp shines in every direction, so its shadow map is a cube: six square depth images,
// one for each way out of the lamp, filled from the lamp's position every frame. The
// camera pass then asks each lamp's cube whether anything stands between the lamp and
// the point being lit.
//
//   1  shadows on or off
//   2  fill each cube with six rendering passes, or with one multiview pass
//   3  how the cubes are read: one texel, bilinear, or 3x3 bilinear taps
//   4  set the lamps moving

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>
#include <vkc/timer.hpp>

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
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

// 5.3's six placements: 5.2's four crates, the small one stacked on the first, and the
// post.
constexpr std::array<Placement, 6> kCrates{{
    {{-2.20F, kFloorY, 0.75F}, 0.36F, {1.0F, 1.0F}},
    {{2.05F, kFloorY, -1.05F}, -0.72F, {1.0F, 1.0F}},
    {{-1.35F, kFloorY, -3.40F}, 0.18F, {1.0F, 1.0F}},
    {{3.10F, kFloorY, 1.60F}, 0.95F, {1.0F, 1.0F}},
    {{-2.15F, kFloorY + 1.0F, 0.70F}, 1.10F, {0.55F, 0.55F}},
    {{0.35F, kFloorY, -1.90F}, 0.30F, {0.30F, 2.60F}},
}};

constexpr glm::vec4 kWhite{1.0F};

// 5.2's three lamps, in the same places and the same colours.
struct LampPlacement {
    glm::vec3 position;
    glm::vec3 colour;
};

constexpr uint32_t kLampCount = 3;
constexpr std::array<LampPlacement, kLampCount> kLamps{{
    {{0.0F, kFloorY + 0.55F, -0.60F}, {1.0F, 0.87F, 0.68F}},
    {{-3.30F, kFloorY + 0.90F, -2.50F}, {0.50F, 0.68F, 1.0F}},
    {{2.90F, kFloorY + 0.80F, 0.10F}, {1.0F, 0.52F, 0.46F}},
}};
constexpr float kLampRadius = 0.06F;
constexpr float kLampIntensity = 2.0F;

// With key 4, each lamp circles its own place, this far out and this fast.
constexpr float kOrbitRadius = 0.70F;
constexpr std::array<float, kLampCount> kOrbitRates{0.60F, -0.45F, 0.75F};

// 5.2's night.
constexpr glm::vec3 kAmbient{0.020F, 0.022F, 0.030F};

// ---------------------------------------------------------------------------
// The shadow cubes
// ---------------------------------------------------------------------------

// Texels along each side of each face.
constexpr uint32_t kShadowMapSize = 1024;
constexpr uint32_t kFaceCount = 6;

// Depth from each lamp is kept between these distances, measured along whichever axis a
// face looks down. The floor's farthest corner is 26 units along an axis from the lamp
// furthest from it, wherever key 4 has moved the lamps.
constexpr float kShadowNear = 0.1F;
constexpr float kShadowFar = 40.0F;

// 5.3's slope-scaled bias for 3x3 bilinear taps. The article explains why 5.3's argument
// carries over to a perspective projection unchanged.
constexpr float kSlopeBias = 4.0F;
// And, for the first time, a constant one: four steps of the depth format. A surface
// that faces a lamp head-on has no slope for the slope term to scale, and the article
// measures what that leaves to chance.
constexpr float kConstantBias = 4.0F;

// The six directions a cube map faces, in the order the specification gives its array
// layers, and for each the "up" vector that lays the face out the way a cube lookup will
// read it. The article derives every one of these from the specification's table.
struct CubeFace {
    glm::vec3 forward;
    glm::vec3 up;
};
constexpr std::array<CubeFace, kFaceCount> kCubeFaces{{
    {{1, 0, 0}, {0, -1, 0}},   // +X
    {{-1, 0, 0}, {0, -1, 0}},  // -X
    {{0, 1, 0}, {0, 0, 1}},    // +Y
    {{0, -1, 0}, {0, 0, -1}},  // -Y
    {{0, 0, 1}, {0, -1, 0}},   // +Z
    {{0, 0, -1}, {0, -1, 0}},  // -Z
}};

// One bit per face: all six views of a multiview pass.
constexpr uint32_t kAllFaces = (1U << kFaceCount) - 1;

// How the shadow pass fills the cubes: what key 2 switches between.
enum class FillMode : uint32_t { SixPasses = 0, Multiview = 1 };
constexpr std::array<const char*, 2> kFillNames{"six passes per lamp",
                                                "one multiview pass per lamp"};

// 5.3's three ways of reading a shadow map. The values are Globals::filter.
enum class ShadowFilter : uint32_t { OneTexel = 0, Bilinear = 1, Pcf3x3 = 2 };

constexpr std::array<const char*, 3> kFilterNames{"one texel", "bilinear",
                                                  "3x3 bilinear taps"};

// ---------------------------------------------------------------------------
// What the shaders read through a pointer
// ---------------------------------------------------------------------------

// The bits of Globals::flags.
constexpr uint32_t kShadowsOn = 1U << 0;

// 5.2's lamp, unchanged.
struct Lamp {
    glm::vec4 position;  // w unused
    glm::vec4 colour;    // rgb already scaled by kLampIntensity, w unused
};
static_assert(sizeof(Lamp) == 32);

// 5.2's Globals, with eighteen matrices -- six faces for each of three lamps -- and what
// the fragment shader needs to turn a distance into the depth a face stored.
struct Globals {
    glm::mat4 view;                                          // 0
    glm::mat4 projection;                                    // 64
    glm::vec4 view_position;                                 // 128
    glm::vec4 ambient;                                       // 144
    std::array<Lamp, kLampCount> lamps;                      // 160
    std::array<glm::mat4, kLampCount * kFaceCount> faces;    // 256, lamp * 6 + face
    float shadow_near;                                       // 1408
    float shadow_far;                                        // 1412
    float shadow_texel;                                      // 1416, in uv units
    uint32_t flags;                                          // 1420
    uint32_t filter;                                         // 1424
};
static_assert(offsetof(Globals, faces) == 256);
static_assert(offsetof(Globals, shadow_near) == 1408);
static_assert(sizeof(Globals) == 1428);

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

// 4.5's push constants with one more field, which fills the four bytes of padding that
// were already at the end: which of Globals::faces the shadow pass draws with.
struct PushConstants {
    VkDeviceAddress globals;   // 0
    VkDeviceAddress draws;     // 8
    VkDeviceAddress vertices;  // 16
    uint32_t draw_index;       // 24
    uint32_t face_base;        // 28
};
static_assert(offsetof(PushConstants, face_base) == 28);
static_assert(sizeof(PushConstants) == 32);

// Where each object's DrawData sits in the array the frame writes. The order is fixed by
// write_draws() below and nothing else may reorder it.
constexpr uint32_t kFloorDraw = 0;
constexpr uint32_t kFirstCrateDraw = kFloorDraw + 1;
constexpr uint32_t kFirstLampDraw =
    kFirstCrateDraw + static_cast<uint32_t>(kCrates.size());
constexpr uint32_t kDrawCount = kFirstLampDraw + kLampCount;

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

// 64 KiB, against the under 3 KiB a frame actually uses. Arena sizing is a guess that
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

[[nodiscard]] std::pair<std::vector<Vertex>, std::vector<uint16_t>> make_sphere(
    uint32_t rings, uint32_t segments, float radius) {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    vertices.reserve(static_cast<size_t>(rings + 1) * (segments + 1));
    indices.reserve(static_cast<size_t>(rings) * segments * 6);

    for (uint32_t ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(rings);
        const float phi = v * std::numbers::pi_v<float>;
        for (uint32_t segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(segments);
            const float theta = u * 2.0F * std::numbers::pi_v<float>;
            const glm::vec3 normal{std::sin(phi) * std::cos(theta), std::cos(phi),
                                   std::sin(phi) * std::sin(theta)};
            vertices.push_back({normal * radius, normal, {u, v}});
        }
    }

    for (uint32_t ring = 0; ring < rings; ++ring) {
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const auto a = static_cast<uint16_t>(ring * (segments + 1) + segment);
            const auto b = static_cast<uint16_t>(a + segments + 1);
            indices.insert(indices.end(), {a, static_cast<uint16_t>(a + 1), b,
                                           static_cast<uint16_t>(a + 1),
                                           static_cast<uint16_t>(b + 1), b});
        }
    }
    return {std::move(vertices), std::move(indices)};
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

// Where lamp i is at time t: its place, or a point on a small circle around it.
[[nodiscard]] glm::vec3 lamp_position(uint32_t i, float t) {
    const float angle = kOrbitRates[i] * t;
    return kLamps[i].position +
           kOrbitRadius * glm::vec3(std::cos(angle), 0.0F, std::sin(angle));
}

// The six cameras a lamp needs: one at the lamp, looking down each axis, each seeing
// exactly a quarter turn across so that together they leave no gap and no overlap.
[[nodiscard]] std::array<glm::mat4, kFaceCount> cube_view_projections(glm::vec3 lamp) {
    // A square, 90-degree frustum. There is no y flip here, unlike every other camera
    // in the series: the specification fixes which way up a cube face is read, and the
    // up vectors in kCubeFaces already lay each face out that way. The article works
    // through what that does to triangle winding.
    const glm::mat4 projection =
        glm::perspective(glm::radians(90.0F), 1.0F, kShadowNear, kShadowFar);

    std::array<glm::mat4, kFaceCount> faces{};
    for (uint32_t face = 0; face < kFaceCount; ++face) {
        const glm::mat4 view =
            glm::lookAt(lamp, lamp + kCubeFaces[face].forward, kCubeFaces[face].up);
        faces[face] = projection * view;
    }
    return faces;
}

// One lamp's shadow cube, and the views it is rendered through.
struct ShadowCube {
    // A cube-compatible depth image of six layers. Its own view is a CUBE view of all
    // six: the camera pass samples through it, and the multiview fill renders into it.
    vkc::Image image;
    // Each layer alone, as a 2D view, for the six-pass fill.
    std::array<VkImageView, kFaceCount> face_views{};

    void destroy(VkDevice device) noexcept {
        for (VkImageView view : face_views) {
            vkDestroyImageView(device, view, nullptr);
        }
        image.destroy();
    }
};

class PointShadowsApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        cube_ = upload_mesh(context(), kCubeVertices, kCubeIndices);
        floor_ = upload_mesh(context(), kFloorVertices, kFloorIndices);

        const auto [sphere_vertices, sphere_indices] = make_sphere(16, 32, kLampRadius);
        lamp_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        depth_format_ = vkc::choose_depth_format(context().physical_device());

        load_textures();
        create_shadow_cubes();
        create_arenas();
        create_descriptors();
        create_depth_buffer(swapchain().extent());
        create_pipelines();
        timer_.create(context());

        draws_.reserve(kDrawCount);

        camera_.position = {0.4F, 2.6F, 7.2F};
        camera_.pitch = -24.0F;

        spdlog::info("1 shadows, 2 six passes or multiview, 3 filter, 4 move the lamps.");
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
                fill_ = fill_ == FillMode::SixPasses ? FillMode::Multiview
                                                     : FillMode::SixPasses;
                log_mode();
                break;
            case SDL_SCANCODE_3:
                filter_ = static_cast<ShadowFilter>(
                    (static_cast<uint32_t>(filter_) + 1) % kFilterNames.size());
                log_mode();
                break;
            case SDL_SCANCODE_4:
                moving_ = !moving_;
                log_mode();
                break;
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        timer_.destroy();
        vkDestroyPipeline(device, lamp_pipeline_, nullptr);
        vkDestroyPipeline(device, multiview_pipeline_, nullptr);
        vkDestroyPipeline(device, face_pipeline_, nullptr);
        vkDestroyPipeline(device, opaque_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, shadow_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, texture_set_layout_, nullptr);

        shadow_linear_sampler_.destroy();
        shadow_nearest_sampler_.destroy();
        for (ShadowCube& cube : shadow_cubes_) {
            cube.destroy(device);
        }

        sampler_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();

        for (FrameArena& arena : arenas_) {
            arena.destroy();
        }
        depth_.destroy();
        lamp_.destroy();
        floor_.destroy();
        cube_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());
        if (moving_) {
            lamp_time_ += delta_time();
        }

        const uint32_t slot = frame.frame_number % vkc::kFramesInFlight;
        record_gpu_time(timer_.read(slot));

        FrameArena& arena = arenas_[slot];
        arena.reset();

        write_globals(arena, frame.extent);
        write_draws(arena);

        // Only the shadow cubes are timed: they are what key 2 changes.
        timer_.reset(frame.cmd, slot);
        timer_.begin(frame.cmd, slot);
        for (uint32_t lamp = 0; lamp < kLampCount; ++lamp) {
            render_shadow_cube(frame.cmd, lamp);
        }
        timer_.end(frame.cmd, slot);

        render_scene(frame);
    }

private:
    // ---------------------------------------------------------------------------
    // Pass one: depth from each lamp, in six directions
    // ---------------------------------------------------------------------------

    void render_shadow_cube(VkCommandBuffer cmd, uint32_t lamp) {
        const ShadowCube& cube = shadow_cubes_[lamp];

        // 5.3's two barriers, unchanged. image_barrier() covers every array layer, so
        // one call moves all six faces.
        vkc::image_barrier(cmd, cube.image.handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);

        const uint32_t first_face = lamp * kFaceCount;
        if (fill_ == FillMode::SixPasses) {
            // Six rendering passes, each into one layer through its own 2D view, each
            // drawing the whole scene with its own face's matrix.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, face_pipeline_);
            for (uint32_t face = 0; face < kFaceCount; ++face) {
                begin_shadow_pass(cmd, cube.face_views[face], /*view_mask=*/0);
                draw_geometry(cmd, first_face + face);
                vkCmdEndRendering(cmd);
            }
        } else {
            // One rendering pass into all six layers. The view mask says which layers;
            // every draw is then run once per set bit, and the vertex shader is told
            // which run it is in.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, multiview_pipeline_);
            begin_shadow_pass(cmd, cube.image.view(), kAllFaces);
            draw_geometry(cmd, first_face);
            vkCmdEndRendering(cmd);
        }

        vkc::image_barrier(cmd, cube.image.handle(),
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // 5.3's depth-only rendering instance, into whichever view it is given. With a view
    // mask, layerCount is ignored: the mask is what says how many layers are drawn.
    static void begin_shadow_pass(VkCommandBuffer cmd, VkImageView view,
                                  uint32_t view_mask) {
        const VkRenderingAttachmentInfo depth_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = view,
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
            .viewMask = view_mask,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthAttachment = &depth_attachment,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(cmd, &rendering);
        set_viewport(cmd, extent);
    }

    // ---------------------------------------------------------------------------
    // Pass two: the scene from the camera
    // ---------------------------------------------------------------------------

    void render_scene(const vkc::FrameInfo& frame) {
        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.0F, 0.0F, 0.0F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        const VkDescriptorSet shadow_set =
            filter_ == ShadowFilter::OneTexel ? shadow_nearest_set_ : shadow_linear_set_;
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 1, 1, &shadow_set, 0, nullptr);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque_pipeline_);
        draw_geometry(frame.cmd, 0);

        // The lamps themselves, drawn only here. In the shadow pass each would sit
        // around its own lamp's camera, and only two accidents would keep it from hiding
        // the room: seen from inside, every triangle faces away and is culled, and the
        // near plane is further out than the sphere's radius.
        bind_indices(frame.cmd, lamp_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        for (uint32_t i = 0; i < kLampCount; ++i) {
            push(frame.cmd, lamp_.vertex_address, kFirstLampDraw + i, 0);
            vkCmdDrawIndexed(frame.cmd, lamp_.index_count, 1, 0, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
    }

    // The floor and every crate, with whatever pipeline is bound. face_base is read only
    // by the shadow pass's vertex shader.
    void draw_geometry(VkCommandBuffer cmd, uint32_t face_base) {
        bind_indices(cmd, floor_);
        bind_texture(cmd, floor_set_);
        push(cmd, floor_.vertex_address, kFloorDraw, face_base);
        vkCmdDrawIndexed(cmd, floor_.index_count, 1, 0, 0, 0);

        bind_indices(cmd, cube_);
        bind_texture(cmd, crate_set_);
        for (uint32_t i = 0; i < kCrates.size(); ++i) {
            push(cmd, cube_.vertex_address, kFirstCrateDraw + i, face_base);
            vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
        }
    }

    // ---------------------------------------------------------------------------
    // Filling the arena
    // ---------------------------------------------------------------------------

    void write_globals(FrameArena& arena, VkExtent2D extent) {
        Globals globals{
            .view = camera_.view(),
            .projection = projection(extent),
            .view_position = glm::vec4(camera_.position, 1.0F),
            .ambient = glm::vec4(kAmbient, 1.0F),
            .lamps = {},
            .faces = {},
            .shadow_near = kShadowNear,
            .shadow_far = kShadowFar,
            .shadow_texel = 1.0F / static_cast<float>(kShadowMapSize),
            .flags = shadows_ ? kShadowsOn : 0U,
            .filter = static_cast<uint32_t>(filter_),
        };
        for (uint32_t i = 0; i < kLampCount; ++i) {
            const glm::vec3 position = lamp_position(i, lamp_time_);
            globals.lamps[i] = Lamp{
                .position = glm::vec4(position, 1.0F),
                .colour = glm::vec4(kLamps[i].colour * kLampIntensity, 1.0F),
            };
            const std::array<glm::mat4, kFaceCount> faces =
                cube_view_projections(position);
            std::copy(faces.begin(), faces.end(), globals.faces.begin() + i * kFaceCount);
        }
        globals_address_ = arena.write(&globals, sizeof(globals));
    }

    void write_draws(FrameArena& arena) {
        draws_.clear();

        draws_.push_back(draw_data_of(glm::mat4(1.0F), kWhite));

        for (const Placement& crate : kCrates) {
            draws_.push_back(draw_data_of(crate_model(crate), kWhite));
        }

        for (uint32_t i = 0; i < kLampCount; ++i) {
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), lamp_position(i, lamp_time_));
            draws_.push_back(draw_data_of(model, glm::vec4(kLamps[i].colour, 1.0F)));
        }

        draws_address_ = arena.write(draws_.data(), draws_.size() * sizeof(DrawData));
    }

    // 4.7's once-a-second average of what the timer measured.
    void record_gpu_time(double ms) {
        if (ms >= 0.0) {
            gpu_ms_ += ms;
            ++gpu_samples_;
        }
        if (elapsed() - stats_started_ < 1.0F || gpu_samples_ == 0) {
            return;
        }
        spdlog::info("shadow cubes, {}: GPU {:6.3f} ms",
                     kFillNames[static_cast<uint32_t>(fill_)], gpu_ms_ / gpu_samples_);
        gpu_ms_ = 0.0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
    }

    void log_mode() const {
        spdlog::info("shadows {}, {}, {}{}", shadows_ ? "on" : "off",
                     kFillNames[static_cast<uint32_t>(fill_)],
                     kFilterNames[static_cast<uint32_t>(filter_)],
                     moving_ ? ", lamps moving" : "");
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
    // ---------------------------------------------------------------------------

    void push(VkCommandBuffer cmd, VkDeviceAddress vertices, uint32_t draw_index,
              uint32_t face_base) {
        const PushConstants constants{
            .globals = globals_address_,
            .draws = draws_address_,
            .vertices = vertices,
            .draw_index = draw_index,
            .face_base = face_base,
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

    // Three cube-compatible depth images, and the extra views each is rendered through.
    void create_shadow_cubes() {
        // 5.3's format question, asked again: the faces are rendered into and then
        // sampled, and filtered by key 3's bilinear modes.
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

        for (uint32_t lamp = 0; lamp < kLampCount; ++lamp) {
            ShadowCube& cube = shadow_cubes_[lamp];
            // 4.4's cube, made from depth instead of loaded from six files.
            cube.image = vkc::Image(
                context(),
                vkc::ImageDesc{
                    .format = kShadowFormat,
                    .extent = {kShadowMapSize, kShadowMapSize},
                    .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT,
                    .mip_levels = 1,
                    .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .array_layers = kFaceCount,
                    .cube = true,
                });
            const std::string name = std::format("shadow cube {}", lamp);
            context().name(cube.image.handle(), VK_OBJECT_TYPE_IMAGE, name.c_str());

            for (uint32_t face = 0; face < kFaceCount; ++face) {
                cube.face_views[face] = create_face_view(cube.image.handle(), face);
            }
        }

        // 5.3's two comparison samplers, now made by vkcommon, with one difference: no
        // border. A cube has no outside -- every direction lands on some face -- and the
        // specification says a cube image ignores the sampler's address modes. A
        // bilinear read that runs off one face's edge takes its missing texels from the
        // neighbouring face instead.
        const auto shadow_sampler = [&](VkFilter filter) {
            return vkc::Sampler(context(),
                                vkc::SamplerDesc{
                                    .mag_filter = filter,
                                    .min_filter = filter,
                                    .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                    .address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    .anisotropy = false,
                                    .compare = true,
                                    .compare_op = VK_COMPARE_OP_LESS_OR_EQUAL,
                                });
        };
        shadow_nearest_sampler_ = shadow_sampler(VK_FILTER_NEAREST);
        shadow_linear_sampler_ =
            shadow_sampler(filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }

    // A second view of one layer of a cube, as an ordinary 2D image. A rendering pass
    // without a view mask draws into the first layer of its attachment's view, so to
    // draw into face 3 alone, a pass needs a view that starts at layer 3.
    [[nodiscard]] VkImageView create_face_view(VkImage image, uint32_t face) {
        const VkImageViewCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .components = {},
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = face,
                    .layerCount = 1,
                },
        };
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(context().device(), &info, nullptr, &view));
        return view;
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

        // Set 1 is one binding holding an array of three descriptors, one cube per lamp.
        const VkDescriptorSetLayoutBinding shadow_binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = kLampCount,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo shadow_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &shadow_binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr,
                                             &shadow_set_layout_));

        // Two texture sets, and two shadow sets of three descriptors each: one with the
        // nearest comparison sampler, one with the linear.
        constexpr uint32_t kSets = 4;
        constexpr uint32_t kDescriptors = 2 + 2 * kLampCount;
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

        // Each cube through its own CUBE view, the one vkc::Image made with it.
        shadow_nearest_set_ = allocate_set(shadow_set_layout_);
        shadow_linear_set_ = allocate_set(shadow_set_layout_);
        for (uint32_t lamp = 0; lamp < kLampCount; ++lamp) {
            const VkImageView view = shadow_cubes_[lamp].image.view();
            write_image(shadow_nearest_set_, lamp, view,
                        shadow_nearest_sampler_.handle());
            write_image(shadow_linear_set_, lamp, view, shadow_linear_sampler_.handle());
        }
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

    // 5.3's write_image(), with the binding replaced by an element of binding 0's array:
    // dstArrayElement is which of the three descriptors this write fills.
    void write_image(VkDescriptorSet set, uint32_t element, VkImageView view,
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
            .dstBinding = 0,
            .dstArrayElement = element,
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

        const auto camera_pipeline = [&](const char* fragment_entry) {
            return vkc::PipelineBuilder(device)
                .shaders(scene_shader, "vertexMain", fragment_entry)
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .layout(pipeline_layout_)
                .build();
        };
        opaque_pipeline_ = camera_pipeline("fragmentMain");
        lamp_pipeline_ = camera_pipeline("lampFragmentMain");

        // The two shadow pipelines differ only in their view mask, which has to match
        // the rendering pass each is used in. The front face is CLOCKWISE: a cube face
        // is a mirror image of what an ordinary camera would show, and the article
        // works out why.
        const auto shadow_pipeline = [&](uint32_t view_mask) {
            return vkc::PipelineBuilder(device)
                .shaders(scene_shader, "shadowVertexMain", "")
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE)
                .depth(VK_FORMAT_D32_SFLOAT, /*test=*/true, /*write=*/true)
                .depth_bias(kConstantBias, kSlopeBias)
                .view_mask(view_mask)
                .layout(pipeline_layout_)
                .build();
        };
        face_pipeline_ = shadow_pipeline(0);
        multiview_pipeline_ = shadow_pipeline(kAllFaces);

        vkDestroyShaderModule(device, scene_shader, nullptr);

        context().name(opaque_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene opaque");
        context().name(lamp_pipeline_, VK_OBJECT_TYPE_PIPELINE, "lamp");
        context().name(face_pipeline_, VK_OBJECT_TYPE_PIPELINE, "shadow, one face");
        context().name(multiview_pipeline_, VK_OBJECT_TYPE_PIPELINE, "shadow, multiview");
    }

    vkc::Camera camera_;

    Mesh cube_;
    Mesh floor_;
    Mesh lamp_;

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

    std::array<ShadowCube, kLampCount> shadow_cubes_;
    vkc::Sampler shadow_nearest_sampler_;
    vkc::Sampler shadow_linear_sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shadow_nearest_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shadow_linear_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadow_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline opaque_pipeline_ = VK_NULL_HANDLE;
    VkPipeline lamp_pipeline_ = VK_NULL_HANDLE;
    VkPipeline face_pipeline_ = VK_NULL_HANDLE;
    VkPipeline multiview_pipeline_ = VK_NULL_HANDLE;

    vkc::GpuTimer timer_;
    double gpu_ms_ = 0.0;
    uint32_t gpu_samples_ = 0;
    float stats_started_ = 0.0F;

    bool shadows_ = true;
    FillMode fill_ = FillMode::Multiview;
    ShadowFilter filter_ = ShadowFilter::Pcf3x3;
    bool moving_ = false;
    float lamp_time_ = 0.0F;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 5.4 - Point shadows";
        PointShadowsApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
