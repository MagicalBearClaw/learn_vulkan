// 5.2 Gamma correction
//
// 5.1's room with three lamps instead of one, and the two halves of sRGB handling made
// switchable. Every chapter since 1.4 has decoded colour textures to linear light when
// sampling them, and encoded linear light back to sRGB when writing the swapchain, without
// a line of code doing either. Here each half can be turned off, to see what it was
// doing. The third key shows the one thing the encoding cannot help with: light that adds
// up to more than 1.0.
//
//   1  decode the colour textures, or read their bytes as if they were linear
//   2  encode the output, or show linear values as a UNORM swapchain would
//   3  mark every pixel whose light was clipped at 1.0

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
    glm::vec2 size;
};

constexpr std::array<Placement, 4> kCrates{{
    {{-2.20F, kFloorY, 0.75F}, 0.36F, {1.0F, 1.0F}},
    {{2.05F, kFloorY, -1.05F}, -0.72F, {1.0F, 1.0F}},
    {{-1.35F, kFloorY, -3.40F}, 0.18F, {1.0F, 1.0F}},
    {{3.10F, kFloorY, 1.60F}, 0.95F, {1.0F, 1.0F}},
}};

constexpr glm::vec4 kWhite{1.0F};

// Three lamps: 5.1's, low over the middle of the floor, and two more at the sides, so that
// light from more than one lamp adds up on the crates between them.
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

// Each lamp's colour is scaled by this before it reaches the shader. Above 1, on purpose:
// a lamp is brighter than a white surface, and the article measures what happens to the
// pixels where that shows.
constexpr float kLampIntensity = 2.0F;

// Night: almost nothing but the lamps.
constexpr glm::vec3 kAmbient{0.020F, 0.022F, 0.030F};

// ---------------------------------------------------------------------------
// What the shaders read through a pointer
// ---------------------------------------------------------------------------

// One lamp as the shader sees it. Two vec4 rather than two vec3, so that an array of
// these has the same 32-byte stride in C++ and Slang without anyone having to think
// about it.
struct Lamp {
    glm::vec4 position;  // w unused
    glm::vec4 colour;    // rgb already scaled by kLampIntensity, w unused
};
static_assert(sizeof(Lamp) == 32);

// The bits of Globals::flags. Texture decoding is not among them: that is decided by
// which descriptor set is bound, and the shader never knows.
constexpr uint32_t kEncodeOutput = 1U << 0;
constexpr uint32_t kShowClipping = 1U << 1;

// 5.1's Globals with the one light replaced by an array of them, and the specular
// switches replaced by flags.
struct Globals {
    glm::mat4 view;                       // 0
    glm::mat4 projection;                 // 64
    glm::vec4 view_position;              // 128
    glm::vec4 ambient;                    // 144
    std::array<Lamp, kLampCount> lamps;   // 160
    uint32_t flags;                       // 256
};
static_assert(offsetof(Globals, lamps) == 160);
static_assert(sizeof(Globals) == 260);

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
        glm::vec3(placement.size.x, placement.size.y, 1.0F));
}

[[nodiscard]] DrawData draw_data_of(const glm::mat4& model, const glm::vec4& colour) {
    return DrawData{
        .model = model,
        .normal_matrix = glm::inverseTranspose(glm::mat3(model)),
        .colour = colour,
    };
}

class GammaCorrectionApp : public vkc::App {
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
        create_arenas();
        create_descriptors();
        create_depth_buffer(swapchain().extent());
        create_pipelines();

        draws_.reserve(kDrawCount);

        camera_.position = {0.0F, 0.35F, 6.40F};
        camera_.pitch = -9.0F;

        spdlog::info("1 decode textures, 2 encode output, 3 show clipping.");
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
                decode_textures_ = !decode_textures_;
                log_mode();
                break;
            case SDL_SCANCODE_2:
                encode_output_ = !encode_output_;
                log_mode();
                break;
            case SDL_SCANCODE_3:
                show_clipping_ = !show_clipping_;
                log_mode();
                break;
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, lamp_pipeline_, nullptr);
        vkDestroyPipeline(device, opaque_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, texture_set_layout_, nullptr);

        sampler_.destroy();
        floor_raw_map_.destroy();
        crate_raw_map_.destroy();
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

        // Everything the shaders will read this frame is written here, into this frame
        // slot's arena, in the order the frame happens to want it. The arena is emptied
        // first: the GPU finished with the last frame that used this slot before the
        // command buffer was handed over, so there is nothing left in it to protect.
        FrameArena& arena = arenas_[frame.frame_number % vkc::kFramesInFlight];
        arena.reset();

        write_globals(arena, frame.extent);
        write_draws(arena);

        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.0F, 0.0F, 0.0F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        bind_indices(frame.cmd, floor_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque_pipeline_);
        bind_texture(frame.cmd, decode_textures_ ? floor_set_ : floor_raw_set_);
        push(frame.cmd, floor_.vertex_address, kFloorDraw);
        vkCmdDrawIndexed(frame.cmd, floor_.index_count, 1, 0, 0, 0);

        bind_indices(frame.cmd, cube_);
        bind_texture(frame.cmd, decode_textures_ ? crate_set_ : crate_raw_set_);
        for (uint32_t i = 0; i < kCrates.size(); ++i) {
            push(frame.cmd, cube_.vertex_address, kFirstCrateDraw + i);
            vkCmdDrawIndexed(frame.cmd, cube_.index_count, 1, 0, 0, 0);
        }

        bind_indices(frame.cmd, lamp_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lamp_pipeline_);
        for (uint32_t i = 0; i < kLampCount; ++i) {
            push(frame.cmd, lamp_.vertex_address, kFirstLampDraw + i);
            vkCmdDrawIndexed(frame.cmd, lamp_.index_count, 1, 0, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
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
            .flags = (encode_output_ ? kEncodeOutput : 0U) |
                     (show_clipping_ ? kShowClipping : 0U),
        };
        for (uint32_t i = 0; i < kLampCount; ++i) {
            globals.lamps[i] = Lamp{
                .position = glm::vec4(kLamps[i].position, 1.0F),
                .colour = glm::vec4(kLamps[i].colour * kLampIntensity, 1.0F),
            };
        }
        globals_address_ = arena.write(&globals, sizeof(globals));
    }

    void write_draws(FrameArena& arena) {
        draws_.clear();

        draws_.push_back(draw_data_of(glm::mat4(1.0F), kWhite));

        for (const Placement& crate : kCrates) {
            draws_.push_back(draw_data_of(
                glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F)), kWhite));
        }

        for (const LampPlacement& lamp : kLamps) {
            draws_.push_back(draw_data_of(glm::translate(glm::mat4(1.0F), lamp.position),
                                          glm::vec4(lamp.colour, 1.0F)));
        }

        draws_address_ = arena.write(draws_.data(), draws_.size() * sizeof(DrawData));
    }

    void log_mode() const {
        spdlog::info("textures {}, output {}{}", decode_textures_ ? "decoded" : "raw",
                     encode_output_ ? "encoded" : "raw",
                     show_clipping_ ? ", clipping shown" : "");
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

    // Each colour texture twice: once as _SRGB, which 2.4 explained, and once as _UNORM,
    // which hands the shader the file's bytes divided by 255 with no curve applied. Two
    // images of the same bytes is the simple way to get both. The same pixels can also be
    // given two views of one image, by creating it with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    // vkc::load_texture does not, and twice the memory for two textures is no hardship.
    void load_textures() {
        crate_map_ = vkc::load_texture(
            context(), vkc::asset_path("textures/lvk_crate_diffuse.png"),
            VK_FORMAT_R8G8B8A8_SRGB);
        floor_map_ = vkc::load_texture(context(),
                                       vkc::asset_path("textures/lvk_ground.png"),
                                       VK_FORMAT_R8G8B8A8_SRGB);
        crate_raw_map_ = vkc::load_texture(
            context(), vkc::asset_path("textures/lvk_crate_diffuse.png"),
            VK_FORMAT_R8G8B8A8_UNORM);
        floor_raw_map_ = vkc::load_texture(context(),
                                           vkc::asset_path("textures/lvk_ground.png"),
                                           VK_FORMAT_R8G8B8A8_UNORM);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});
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

        constexpr uint32_t kTextureSets = 4;  // crate and floor, decoded and raw
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                             kTextureSets};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = kTextureSets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        crate_set_ = allocate_texture_set(crate_map_.view(), sampler_.handle());
        floor_set_ = allocate_texture_set(floor_map_.view(), sampler_.handle());
        crate_raw_set_ = allocate_texture_set(crate_raw_map_.view(), sampler_.handle());
        floor_raw_set_ = allocate_texture_set(floor_raw_map_.view(), sampler_.handle());
    }

    [[nodiscard]] VkDescriptorSet allocate_texture_set(VkImageView view,
                                                       VkSampler sampler) {
        const VkDevice device = context().device();

        const VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &texture_set_layout_,
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc, &set));

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
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &texture_set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule scene_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "scene.slang");

        const auto base = [&](const char* fragment_entry) -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(scene_shader, "vertexMain", fragment_entry)
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .layout(pipeline_layout_);
        };
        opaque_pipeline_ = base("fragmentMain").build();
        lamp_pipeline_ = base("lampFragmentMain").build();

        vkDestroyShaderModule(device, scene_shader, nullptr);

        context().name(opaque_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene opaque");
        context().name(lamp_pipeline_, VK_OBJECT_TYPE_PIPELINE, "lamp");
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
    vkc::Image crate_raw_map_;
    vkc::Image floor_raw_map_;
    vkc::Sampler sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet crate_raw_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_raw_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline opaque_pipeline_ = VK_NULL_HANDLE;
    VkPipeline lamp_pipeline_ = VK_NULL_HANDLE;

    bool decode_textures_ = true;
    bool encode_output_ = true;
    bool show_clipping_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 5.2 - Gamma correction";
        GammaCorrectionApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
