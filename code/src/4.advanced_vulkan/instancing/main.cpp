// 4.6 Instancing
//
// A planet and a belt of rocks around it -- up to a hundred thousand of them, all the
// same mesh, each with its own transform and colour.
//
// Drawing that the way 4.5 drew its crates means one vkCmdDrawIndexed per rock.
// Instancing asks for all the copies in one call: instanceCount says how many, and each
// copy's vertex shader is told which copy it belongs to.
//
// The sample draws the rocks three ways, so the difference can be measured rather than
// asserted:
//
//   1  one draw call per rock
//   2  one draw call, with per-instance vertex attributes (the fixed-function route)
//   3  one draw call, with the shader reading its instance through a pointer (default)
//   -  fewer rocks          =  more rocks
//   4  spin the belt
//
// Each second it logs what the rocks cost the CPU to record and the GPU to draw. The GPU
// figure comes from timestamps, which this chapter also introduces.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
#include <vkc/image.hpp>
#include <vkc/pipeline.hpp>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

// 4.5's vertex, unchanged.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32);

// ---------------------------------------------------------------------------
// What the shaders read through a pointer
// ---------------------------------------------------------------------------

// 4.5's Globals. The cube-face field is gone and the belt's rotation has taken its
// place, so the size is still 256.
struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::mat4 belt;           // 128
    glm::vec4 view_position;  // 192
    glm::vec4 sun_direction;  // 208
    glm::vec4 sun_colour;     // 224
    glm::vec4 ambient;        // 240, w is the specular exponent
};
static_assert(sizeof(Globals) == 256);

// 4.5's DrawData, renamed. One of these per copy of a mesh.
struct Instance {
    glm::mat4 model;          // 0
    glm::mat3 normal_matrix;  // 64
    glm::vec4 colour;         // 100
};
static_assert(offsetof(Instance, normal_matrix) == 64);
static_assert(offsetof(Instance, colour) == 100);
static_assert(sizeof(Instance) == 116);

// 4.5's push constants without draw_index. Which instance a vertex belongs to now comes
// from the draw call itself.
struct PushConstants {
    VkDeviceAddress globals;    // 0
    VkDeviceAddress instances;  // 8
    VkDeviceAddress vertices;   // 16
};
static_assert(sizeof(PushConstants) == 24);

// ---------------------------------------------------------------------------
// The scene
// ---------------------------------------------------------------------------

constexpr float kPlanetRadius = 4.0F;
constexpr glm::vec4 kPlanetColour{0.62F, 0.36F, 0.22F, 1.0F};

// The belt is an annulus in the xz plane, with a little thickness in y.
constexpr float kBeltInner = 9.0F;
constexpr float kBeltOuter = 17.0F;
constexpr float kBeltThickness = 0.6F;
constexpr float kRockMinSize = 0.04F;
constexpr float kRockMaxSize = 0.22F;

// Instance 0 is the planet; the rocks follow it. The buffer is filled once for the
// largest count and the keys choose how much of it to draw.
constexpr uint32_t kPlanetInstance = 0;
constexpr uint32_t kFirstRock = 1;
constexpr std::array<uint32_t, 7> kRockCounts{1'000,  2'000,  5'000,  10'000,
                                              20'000, 50'000, 100'000};
constexpr uint32_t kMaxRocks = kRockCounts.back();
constexpr size_t kDefaultRockCount = 2;  // 5,000

constexpr glm::vec3 kSunDirection{-0.8F, -0.35F, -0.45F};
constexpr glm::vec3 kSunColour{1.0F, 0.95F, 0.86F};
constexpr glm::vec3 kAmbient{0.05F, 0.055F, 0.07F};
constexpr float kShininess = 24.0F;

// Radians per second. The belt turns as one rigid disc, so the outer edge moves fastest:
// 0.05 x 17 = 0.85 units a second.
constexpr float kBeltSpin = 0.05F;

enum class DrawPath { PerRock, Attributes, Pulled };

// ---------------------------------------------------------------------------
// Random numbers that are the same everywhere
// ---------------------------------------------------------------------------

// The PCG hash from Jarzynski and Olano, "Hash Functions for GPU Rendering" (JCGT, 2020):
// one step of a PCG generator followed by its output permutation. Given the same input it
// returns the same 32 bits on every compiler, which <random> does not promise -- see the
// article for why.
[[nodiscard]] constexpr uint32_t pcg_hash(uint32_t input) {
    const uint32_t state = input * 747796405U + 2891336453U;
    const uint32_t word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
    return (word >> 22U) ^ word;
}
// Checked where it cannot drift: if these ever change, the belt has moved.
static_assert(pcg_hash(0) == 0x07bb2fe2U);
static_assert(pcg_hash(1) == 0xa8beea3cU);

// A stream of floats in [0, 1). Each call hashes the previous result, so one seed gives
// as many numbers as are asked for.
class Random {
public:
    explicit Random(uint32_t seed) noexcept : state_(pcg_hash(seed)) {}

    [[nodiscard]] float next() noexcept {
        state_ = pcg_hash(state_);
        // The top 24 bits, because a float has 24 bits of precision: every value this
        // returns is exactly representable, and 1.0 never is.
        return static_cast<float>(state_ >> 8U) * 0x1.0p-24F;
    }

    [[nodiscard]] float range(float low, float high) noexcept {
        return low + (high - low) * next();
    }

private:
    uint32_t state_;
};

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

// 4.5's buffer_address, unchanged.
[[nodiscard]] VkDeviceAddress buffer_address(VkDevice device, VkBuffer buffer) {
    const VkBufferDeviceAddressInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &info);
}

// 4.5's Mesh and upload_mesh, unchanged.
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

// 4.5's sphere, unchanged. It is the planet now.
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

// A rock: an icosahedron, subdivided, with every corner pushed in or out by a random
// amount, then drawn flat-shaded.
//
// The corners are displaced before the faces are emitted, and the midpoint cache is what
// makes that work: two faces that share an edge must agree on where its midpoint went,
// or the rock cracks open along the seam.
[[nodiscard]] std::pair<std::vector<Vertex>, std::vector<uint16_t>> make_rock(
    uint32_t subdivisions, float roughness, uint32_t seed) {
    // The twelve corners of a regular icosahedron are the cyclic permutations of
    // (0, +-1, +-t), where t is the golden ratio.
    const float t = std::numbers::phi_v<float>;
    std::vector<glm::vec3> corners{
        {-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
        {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1},
    };
    std::vector<glm::uvec3> faces{
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
    };
    for (glm::vec3& corner : corners) {
        corner = glm::normalize(corner);
    }

    // Each subdivision splits every triangle into four, putting a new corner at the
    // middle of each edge and pushing it out onto the unit sphere.
    for (uint32_t level = 0; level < subdivisions; ++level) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        const auto midpoint = [&](uint32_t a, uint32_t b) {
            const std::pair<uint32_t, uint32_t> key{std::min(a, b), std::max(a, b)};
            const auto found = midpoints.find(key);
            if (found != midpoints.end()) {
                return found->second;
            }
            corners.push_back(glm::normalize(corners[a] + corners[b]));
            const auto index = static_cast<uint32_t>(corners.size() - 1);
            midpoints.emplace(key, index);
            return index;
        };

        std::vector<glm::uvec3> split;
        split.reserve(faces.size() * 4);
        for (const glm::uvec3& face : faces) {
            const uint32_t ab = midpoint(face.x, face.y);
            const uint32_t bc = midpoint(face.y, face.z);
            const uint32_t ca = midpoint(face.z, face.x);
            split.insert(split.end(), {{face.x, ab, ca},
                                       {face.y, bc, ab},
                                       {face.z, ca, bc},
                                       {ab, bc, ca}});
        }
        faces = std::move(split);
    }

    Random random(seed);
    for (glm::vec3& corner : corners) {
        corner *= 1.0F + roughness * (random.next() * 2.0F - 1.0F);
    }

    // Flat shading: every triangle gets its own three vertices, all carrying the face's
    // normal. Sharing vertices would average the normals and round the rock off.
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
    vertices.reserve(faces.size() * 3);
    indices.reserve(faces.size() * 3);
    for (const glm::uvec3& face : faces) {
        const glm::vec3 a = corners[face.x];
        const glm::vec3 b = corners[face.y];
        const glm::vec3 c = corners[face.z];
        const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
        for (const glm::vec3& position : {a, b, c}) {
            indices.push_back(static_cast<uint16_t>(vertices.size()));
            vertices.push_back({position, normal, {0.0F, 0.0F}});
        }
    }
    return {std::move(vertices), std::move(indices)};
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

[[nodiscard]] Instance instance_of(const glm::mat4& model, const glm::vec4& colour) {
    return Instance{
        .model = model,
        .normal_matrix = glm::inverseTranspose(glm::mat3(model)),
        .colour = colour,
    };
}

// Where every rock is, decided once. Rock i is the same rock whatever the count, because
// its numbers come from its own seed and not from a shared stream: lowering the count
// removes rocks from the belt without moving the ones that remain.
[[nodiscard]] std::vector<Instance> make_instances() {
    std::vector<Instance> instances;
    instances.reserve(kFirstRock + kMaxRocks);

    instances.push_back(instance_of(glm::mat4(1.0F), kPlanetColour));

    for (uint32_t i = 0; i < kMaxRocks; ++i) {
        Random random(i);

        // Uniform over the annulus's area, not its radius. See the article: picking the
        // radius uniformly would crowd the inner edge.
        const float radius = std::sqrt(random.range(kBeltInner * kBeltInner,
                                                    kBeltOuter * kBeltOuter));
        const float angle = random.range(0.0F, 2.0F * std::numbers::pi_v<float>);
        const glm::vec3 position{radius * std::cos(angle),
                                 random.range(-0.5F, 0.5F) * kBeltThickness,
                                 radius * std::sin(angle)};

        const glm::vec3 axis = glm::normalize(glm::vec3(random.range(-1.0F, 1.0F),
                                                        random.range(-1.0F, 1.0F),
                                                        random.range(0.1F, 1.0F)));
        const float spin = random.range(0.0F, 2.0F * std::numbers::pi_v<float>);

        // Non-uniform: each axis is stretched on its own, which is what makes the normal
        // matrix necessary.
        const float size = random.range(kRockMinSize, kRockMaxSize);
        const glm::vec3 scale = size * glm::vec3(random.range(0.6F, 1.4F),
                                                 random.range(0.6F, 1.4F),
                                                 random.range(0.6F, 1.4F));

        const glm::mat4 model = glm::scale(
            glm::rotate(glm::translate(glm::mat4(1.0F), position), spin, axis), scale);

        const float grey = random.range(0.35F, 0.65F);
        const float warmth = random.range(0.0F, 0.12F);
        instances.push_back(
            instance_of(model, {grey + warmth, grey + warmth * 0.5F, grey, 1.0F}));
    }
    return instances;
}

// ---------------------------------------------------------------------------
// The frame arena: 4.5's code, unchanged
// ---------------------------------------------------------------------------

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
        return base_ + offset;
    }

private:
    vkc::Buffer buffer_;
    VkDeviceAddress base_ = 0;
    VkDeviceSize used_ = 0;
    VkDeviceSize capacity_ = 0;
};

// One Globals a frame now, so 4.5's 64 KiB is far more than enough; 4 KiB still is.
constexpr VkDeviceSize kArenaBytes = 4 * 1024;

// ---------------------------------------------------------------------------
// Timing the GPU
// ---------------------------------------------------------------------------

// How long a stretch of a command buffer took on the GPU.
//
// A timestamp query asks the GPU to write its own clock into a query pool at a point in
// the command stream. Two of them bracket the work, and the difference -- in ticks, which
// timestampPeriod converts to nanoseconds -- is how long the GPU spent between them.
//
// The answer is not available until the GPU has executed the command buffer, which is a
// frame or two after it was recorded. So each frame in flight gets its own pair of
// queries, and a frame reads back the pair its slot wrote last time round: that frame's
// fence has already been waited on, so the results are there.
class GpuTimer {
public:
    void create(vkc::Context& context) {
        device_ = context.device();

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
        period_ns_ = properties.limits.timestampPeriod;

        // A queue that cannot write timestamps reports zero valid bits. Every desktop
        // GPU can; the check is what makes the numbers below trustworthy.
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(),
                                                 &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(),
                                                 &family_count, families.data());
        const uint32_t valid_bits =
            families[context.queue_family()].timestampValidBits;
        if (valid_bits == 0) {
            throw std::runtime_error("This queue cannot write timestamps.");
        }
        mask_ = valid_bits == 64 ? ~uint64_t{0} : (uint64_t{1} << valid_bits) - 1;

        const VkQueryPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2 * vkc::kFramesInFlight,
            .pipelineStatistics = 0,
        };
        VK_CHECK(vkCreateQueryPool(device_, &pool_info, nullptr, &pool_));
    }

    void destroy() noexcept { vkDestroyQueryPool(device_, pool_, nullptr); }

    // What this slot measured the last time it was used, in milliseconds, or a negative
    // number if it has never been used. Must be called before reset() reuses the slot.
    [[nodiscard]] double read(uint32_t slot) const {
        if (!written_[slot]) {
            return -1.0;
        }
        std::array<uint64_t, 2> ticks{};
        VK_CHECK(vkGetQueryPoolResults(device_, pool_, 2 * slot, 2, sizeof(ticks),
                                       ticks.data(), sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT));
        const uint64_t elapsed = (ticks[1] - ticks[0]) & mask_;
        return static_cast<double>(elapsed) * period_ns_ * 1e-6;
    }

    // A query has to be reset before it is written, and the reset is a command like any
    // other: it happens in order, on the GPU, before the timestamps that follow it. It
    // is also forbidden inside a rendering pass, which is why it is separate from
    // begin(): the timestamps may go anywhere, the reset has to go first.
    void reset(VkCommandBuffer cmd, uint32_t slot) {
        vkCmdResetQueryPool(cmd, pool_, 2 * slot, 2);
    }

    void begin(VkCommandBuffer cmd, uint32_t slot) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * slot);
    }

    void end(VkCommandBuffer cmd, uint32_t slot) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_,
                             2 * slot + 1);
        written_[slot] = true;
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    float period_ns_ = 0.0F;
    uint64_t mask_ = 0;
    std::array<bool, vkc::kFramesInFlight> written_{};
};

// ---------------------------------------------------------------------------
// The sample
// ---------------------------------------------------------------------------

class InstancingApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        const auto [sphere_vertices, sphere_indices] = make_sphere(48, 96, kPlanetRadius);
        planet_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        const auto [rock_vertices, rock_indices] = make_rock(2, 0.18F, 7);
        rock_ = upload_mesh(context(), rock_vertices, rock_indices);

        // Written once and never again, so it is staged into device-local memory. 4.5's
        // arena was host-visible because its contents changed every frame; these do
        // not. VERTEX_BUFFER as well as SHADER_DEVICE_ADDRESS, because draw path 2 binds
        // this same buffer as a vertex buffer.
        const std::vector<Instance> instances = make_instances();
        instances_ = vkc::upload_to_device_local(
            context(), instances.data(), instances.size() * sizeof(Instance),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        instances_address_ = buffer_address(context().device(), instances_.handle());

        depth_format_ = vkc::choose_depth_format(context().physical_device());

        for (FrameArena& arena : arenas_) {
            arena.create(context(), kArenaBytes);
        }
        timer_.create(context());
        create_depth_buffer(swapchain().extent());
        create_pipelines();

        camera_.position = {0.0F, 7.5F, 30.0F};
        camera_.pitch = -14.0F;
        camera_.speed = 8.0F;

        spdlog::info("rock: {} triangles; instance buffer: {} x {} B = {:.1f} MiB",
                     rock_.index_count / 3, instances.size(), sizeof(Instance),
                     static_cast<double>(instances.size() * sizeof(Instance)) /
                         (1024.0 * 1024.0));
        spdlog::info("1 a draw per rock, 2 instance attributes, 3 instance pointer, "
                     "-/= rock count, 4 spin the belt.");
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                path_ = DrawPath::PerRock;
                break;
            case SDL_SCANCODE_2:
                path_ = DrawPath::Attributes;
                break;
            case SDL_SCANCODE_3:
                path_ = DrawPath::Pulled;
                break;
            case SDL_SCANCODE_MINUS:
                count_index_ = count_index_ > 0 ? count_index_ - 1 : 0;
                break;
            case SDL_SCANCODE_EQUALS:
                count_index_ = std::min(count_index_ + 1, kRockCounts.size() - 1);
                break;
            case SDL_SCANCODE_4:
                spinning_ = !spinning_;
                break;
            default:
                return;
        }
        reset_stats();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, attribute_pipeline_, nullptr);
        vkDestroyPipeline(device, planet_pipeline_, nullptr);
        vkDestroyPipeline(device, rock_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        timer_.destroy();
        for (FrameArena& arena : arenas_) {
            arena.destroy();
        }
        depth_.destroy();
        instances_.destroy();
        rock_.destroy();
        planet_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());
        if (spinning_) {
            belt_angle_ += kBeltSpin * delta_time();
        }

        const uint32_t slot = frame.frame_number % vkc::kFramesInFlight;
        record_gpu_time(timer_.read(slot));

        FrameArena& arena = arenas_[slot];
        arena.reset();
        const VkDeviceAddress globals = write_globals(arena, frame.extent);

        timer_.reset(frame.cmd, slot);
        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.01F, 0.012F, 0.02F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        push(frame.cmd, globals, planet_.vertex_address);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, planet_pipeline_);
        vkCmdBindIndexBuffer(frame.cmd, planet_.indices.handle(), 0,
                             VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(frame.cmd, planet_.index_count, 1, 0, 0, kPlanetInstance);

        // Only the rocks are timed. The planet is one draw whichever path is chosen.
        timer_.begin(frame.cmd, slot);
        const auto cpu_start = std::chrono::steady_clock::now();
        draw_rocks(frame.cmd, globals);
        const auto cpu_end = std::chrono::steady_clock::now();
        timer_.end(frame.cmd, slot);

        vkCmdEndRendering(frame.cmd);

        record_cpu_time(
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());
    }

private:
    [[nodiscard]] uint32_t rock_count() const { return kRockCounts[count_index_]; }

    // The whole chapter is these three branches.
    void draw_rocks(VkCommandBuffer cmd, VkDeviceAddress globals) {
        const uint32_t count = rock_count();
        vkCmdBindIndexBuffer(cmd, rock_.indices.handle(), 0, VK_INDEX_TYPE_UINT16);

        switch (path_) {
            case DrawPath::PerRock:
                // One draw per rock, each asking for a single instance. firstInstance is
                // what tells the shader which one, so nothing else changes between
                // draws -- not the pipeline, not the push constants. What is left is the
                // cost of a draw call and nothing else.
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rock_pipeline_);
                push(cmd, globals, rock_.vertex_address);
                for (uint32_t i = 0; i < count; ++i) {
                    vkCmdDrawIndexed(cmd, rock_.index_count, 1, 0, 0, kFirstRock + i);
                }
                break;

            case DrawPath::Attributes: {
                // The instance buffer bound as a vertex buffer. The pipeline's binding
                // says it advances once per instance rather than once per vertex, so
                // every vertex of copy i sees element i.
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  attribute_pipeline_);
                push(cmd, globals, rock_.vertex_address);
                const VkBuffer buffer = instances_.handle();
                const VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
                vkCmdDrawIndexed(cmd, rock_.index_count, count, 0, 0, kFirstRock);
                break;
            }

            case DrawPath::Pulled:
                // The same single draw, with the shader indexing the buffer itself.
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rock_pipeline_);
                push(cmd, globals, rock_.vertex_address);
                vkCmdDrawIndexed(cmd, rock_.index_count, count, 0, 0, kFirstRock);
                break;
        }
    }

    [[nodiscard]] VkDeviceAddress write_globals(FrameArena& arena, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = projection(extent),
            .belt = glm::rotate(glm::mat4(1.0F), belt_angle_,
                                glm::vec3(0.0F, 1.0F, 0.0F)),
            .view_position = glm::vec4(camera_.position, 1.0F),
            .sun_direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
            .sun_colour = glm::vec4(kSunColour, 1.0F),
            .ambient = glm::vec4(kAmbient, kShininess),
        };
        return arena.write(&globals, sizeof(globals));
    }

    void push(VkCommandBuffer cmd, VkDeviceAddress globals, VkDeviceAddress vertices) {
        const PushConstants constants{
            .globals = globals,
            .instances = instances_address_,
            .vertices = vertices,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
    }

    // ---------------------------------------------------------------------------
    // Measuring
    // ---------------------------------------------------------------------------

    void reset_stats() {
        cpu_ms_ = 0.0;
        gpu_ms_ = 0.0;
        cpu_samples_ = 0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
        spdlog::info("{} rocks, {}", rock_count(), path_name());
    }

    void record_cpu_time(double ms) {
        cpu_ms_ += ms;
        ++cpu_samples_;
    }

    // Averages over a second and logs, so one slow frame does not decide the number.
    void record_gpu_time(double ms) {
        if (ms >= 0.0) {
            gpu_ms_ += ms;
            ++gpu_samples_;
        }
        if (elapsed() - stats_started_ < 1.0F || gpu_samples_ == 0 || cpu_samples_ == 0) {
            return;
        }
        spdlog::info("{:>7} rocks, {:<20} CPU {:7.3f} ms  GPU {:7.3f} ms", rock_count(),
                     path_name(), cpu_ms_ / cpu_samples_, gpu_ms_ / gpu_samples_);
        cpu_ms_ = 0.0;
        gpu_ms_ = 0.0;
        cpu_samples_ = 0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
    }

    [[nodiscard]] const char* path_name() const {
        switch (path_) {
            case DrawPath::PerRock:
                return "a draw per rock";
            case DrawPath::Attributes:
                return "instance attributes";
            case DrawPath::Pulled:
                return "instance pointer";
        }
        return "";
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
    // ---------------------------------------------------------------------------

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
            glm::perspective(glm::radians(45.0F), aspect, 0.1F, 200.0F);
        projection[1][1] *= -1.0F;  // Vulkan's clip-space y points down.
        return projection;
    }

    void create_depth_buffer(VkExtent2D extent) {
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
    }

    // ---------------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------------

    void create_pipelines() {
        const VkDevice device = context().device();

        // No descriptor set layouts at all. 4.5's only descriptors were its textures,
        // and this scene has none.
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "instancing.slang");

        const auto base = [&]() -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .layout(pipeline_layout_);
        };

        rock_pipeline_ = base().shaders(shader).build();
        planet_pipeline_ =
            base().shaders(shader, "vertexMain", "planetFragmentMain").build();

        // One binding, stepping once per instance, and eight attributes that carve an
        // Instance into the pieces the shader declares. A mat4 is four vec4 columns and
        // a mat3 is three vec3 columns: vertex input has no matrix formats.
        const std::array<VkVertexInputBindingDescription, 1> bindings{{
            {.binding = 0,
             .stride = sizeof(Instance),
             .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE},
        }};
        std::array<VkVertexInputAttributeDescription, 8> attributes{};
        for (uint32_t column = 0; column < 4; ++column) {
            attributes[column] = {
                .location = column,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = static_cast<uint32_t>(offsetof(Instance, model) +
                                                column * sizeof(glm::vec4)),
            };
        }
        for (uint32_t column = 0; column < 3; ++column) {
            attributes[4 + column] = {
                .location = 4 + column,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = static_cast<uint32_t>(offsetof(Instance, normal_matrix) +
                                                column * sizeof(glm::vec3)),
            };
        }
        attributes[7] = {
            .location = 7,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = static_cast<uint32_t>(offsetof(Instance, colour)),
        };

        attribute_pipeline_ = base()
                                  .shaders(shader, "attributeVertexMain", "fragmentMain")
                                  .vertex_input(bindings, attributes)
                                  .build();

        vkDestroyShaderModule(device, shader, nullptr);

        context().name(rock_pipeline_, VK_OBJECT_TYPE_PIPELINE, "rocks (pointer)");
        context().name(attribute_pipeline_, VK_OBJECT_TYPE_PIPELINE,
                       "rocks (instance attributes)");
        context().name(planet_pipeline_, VK_OBJECT_TYPE_PIPELINE, "planet");
    }

    vkc::Camera camera_;

    Mesh planet_;
    Mesh rock_;
    vkc::Buffer instances_;
    VkDeviceAddress instances_address_ = 0;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<FrameArena, vkc::kFramesInFlight> arenas_;
    GpuTimer timer_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline rock_pipeline_ = VK_NULL_HANDLE;
    VkPipeline planet_pipeline_ = VK_NULL_HANDLE;
    VkPipeline attribute_pipeline_ = VK_NULL_HANDLE;

    DrawPath path_ = DrawPath::Pulled;
    size_t count_index_ = kDefaultRockCount;
    bool spinning_ = false;
    float belt_angle_ = 0.0F;

    double cpu_ms_ = 0.0;
    double gpu_ms_ = 0.0;
    uint32_t cpu_samples_ = 0;
    uint32_t gpu_samples_ = 0;
    float stats_started_ = 0.0F;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.6 - Instancing";
        InstancingApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
