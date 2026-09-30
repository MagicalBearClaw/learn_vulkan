// 4.9 Compute
//
// 4.8's belt, moving the way a belt of rocks actually moves.
//
// 4.6 to 4.8 turned the whole belt as one rigid disc: one matrix in the globals, applied
// to every rock, so the outer edge went fastest. Real orbits are the other way round --
// the inner rocks overtake the outer ones -- and every rock also tumbles at its own rate.
// That means a new model matrix and a new normal matrix for every rock, every frame.
//
// There are two ways to do that here, on key 8: a loop on the CPU that writes the
// matrices into the frame arena, as 4.5 would have, or a compute shader that writes them
// into a device-local buffer, one GPU thread per rock. Both run the same arithmetic.
//
//   1 2 3 4   1x, 2x, 4x, 8x multisampling
//   5         sample shading
//   6         orbits on or off
//   7         stars on or off
//   8         update on the GPU (default) or on the CPU
//   - =       fewer / more rocks

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
#include <vkc/image.hpp>
#include <vkc/pipeline.hpp>
#include <vkc/timer.hpp>

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

// 4.8's Globals without the belt's rotation: each rock's own matrix now says where it is
// in its orbit, so there is nothing to apply on top of it.
struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::vec4 view_position;  // 128
    glm::vec4 sun_direction;  // 144
    glm::vec4 sun_colour;     // 160
    glm::vec4 ambient;        // 176, w is the specular exponent
};
static_assert(sizeof(Globals) == 192);

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

// The compute shader's push constants: where to write, which slots, and when. The shader
// puts the four fields at offsets 0, 8, 12 and 16, and so does C++. C++ then pads the
// struct to 24 bytes, a multiple of the address's 8-byte alignment; the shader reads the
// first 20, and the push-constant range covers all 24.
struct OrbitPush {
    VkDeviceAddress instances;  // 0
    uint32_t first;             // 8
    uint32_t count;             // 12
    float time;                 // 16
};
static_assert(sizeof(OrbitPush) == 24);

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

// What keys 1 to 4 ask for. VkSampleCountFlagBits values are the counts themselves, one
// bit each, which is what lets a device report every count it supports as one mask.
constexpr std::array<VkSampleCountFlagBits, 4> kSampleCounts{
    VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT};
constexpr VkSampleCountFlagBits kDefaultSamples = VK_SAMPLE_COUNT_4_BIT;

constexpr glm::vec3 kSunDirection{-0.8F, -0.35F, -0.45F};
constexpr glm::vec3 kSunColour{1.0F, 0.95F, 0.86F};
constexpr glm::vec3 kAmbient{0.05F, 0.055F, 0.07F};
constexpr float kShininess = 24.0F;

// Each rock tumbles about its own axis at up to this many radians a second.
constexpr float kMaxTumble = 0.5F;

// GM, the planet's mass times the gravitational constant, in this scene's units. Chosen
// for the periods it gives, not for any real planet: 2 pi sqrt(r^3 / GM) is 20 seconds at
// the inner edge, radius 9, and 52 at the outer, radius 17.
constexpr float kGravity = 72.0F;

// Threads per workgroup. Must match [numthreads] in orbits.slang.
constexpr uint32_t kWorkgroupSize = 64;

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

// Where rock `index` is at `time`, and how it is turned.
//
// 4.6's make_instances(), made a function of time. The random numbers are drawn in 4.6's
// order, with one more at the end for the tumble, so at time zero every rock is where 4.6
// put it. orbits.slang does exactly this on the GPU.
[[nodiscard]] Instance rock_at(uint32_t index, float time) {
    Random random(index);

    const float radius =
        std::sqrt(random.range(kBeltInner * kBeltInner, kBeltOuter * kBeltOuter));
    const float angle = random.range(0.0F, 2.0F * std::numbers::pi_v<float>);
    const float height = random.range(-0.5F, 0.5F) * kBeltThickness;
    const glm::vec3 axis = glm::normalize(glm::vec3{random.range(-1.0F, 1.0F),
                                                    random.range(-1.0F, 1.0F),
                                                    random.range(0.1F, 1.0F)});
    const float spin = random.range(0.0F, 2.0F * std::numbers::pi_v<float>);
    const float size = random.range(kRockMinSize, kRockMaxSize);
    const glm::vec3 scale = size * glm::vec3{random.range(0.6F, 1.4F),
                                             random.range(0.6F, 1.4F),
                                             random.range(0.6F, 1.4F)};
    const float grey = random.range(0.35F, 0.65F);
    const float warmth = random.range(0.0F, 0.12F);
    const float tumble = random.range(-kMaxTumble, kMaxTumble);

    // Kepler: a circular orbit's angular speed is sqrt(GM / r^3). See the article.
    const float speed = std::sqrt(kGravity / (radius * radius * radius));
    const float orbit_angle = angle + speed * time;
    const glm::vec3 position{radius * std::cos(orbit_angle), height,
                             radius * std::sin(orbit_angle)};

    const glm::mat3 rotation{
        glm::rotate(glm::mat4(1.0F), spin + tumble * time, axis)};

    // translate * rotate * scale, built by columns; and its normal matrix, which for a
    // rotation times a scale is the rotation's columns divided by the scale.
    return Instance{
        .model = glm::mat4{glm::vec4(rotation[0] * scale.x, 0.0F),
                           glm::vec4(rotation[1] * scale.y, 0.0F),
                           glm::vec4(rotation[2] * scale.z, 0.0F),
                           glm::vec4(position, 1.0F)},
        .normal_matrix = glm::mat3{rotation[0] / scale.x, rotation[1] / scale.y,
                                   rotation[2] / scale.z},
        .colour = {grey + warmth, grey + warmth * 0.5F, grey, 1.0F},
    };
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

// Room for the CPU path's whole belt, 100,001 x 116 bytes, and the globals. 4 KiB was
// enough when the arena held only the globals; the CPU path puts every rock in it.
constexpr VkDeviceSize kArenaBytes = 12 * 1024 * 1024;

// ---------------------------------------------------------------------------
// The sample
// ---------------------------------------------------------------------------

class ComputeApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        const auto [sphere_vertices, sphere_indices] = make_sphere(48, 96, kPlanetRadius);
        planet_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        const auto [rock_vertices, rock_indices] = make_rock(2, 0.18F, 7);
        rock_ = upload_mesh(context(), rock_vertices, rock_indices);

        // The GPU path's instances: the planet in slot 0, written once here, and a slot
        // for every rock, which the compute shader fills every frame before anything
        // reads it. Zeroes until then.
        std::vector<Instance> instances(kFirstRock + kMaxRocks);
        instances[kPlanetInstance] = instance_of(glm::mat4(1.0F), kPlanetColour);
        instances_ = vkc::upload_to_device_local(
            context(), instances.data(), instances.size() * sizeof(Instance),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        instances_address_ = buffer_address(context().device(), instances_.handle());
        cpu_instances_.reserve(kFirstRock + kMaxRocks);

        depth_format_ = vkc::choose_depth_format(context().physical_device());

        // Which sample counts this device can render with. A count has to be supported
        // by both kinds of attachment the frame uses, so the two masks are intersected.
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context().physical_device(), &properties);
        supported_samples_ = properties.limits.framebufferColorSampleCounts &
                             properties.limits.framebufferDepthSampleCounts;
        spdlog::info("sample counts supported for colour and depth: mask 0x{:x}",
                     supported_samples_);

        for (FrameArena& arena : arenas_) {
            arena.create(context(), kArenaBytes);
        }
        update_timer_.create(context());
        render_timer_.create(context());
        create_layout();
        create_targets(swapchain().extent());
        create_pipelines();
        create_compute_pipeline();

        camera_.position = {0.0F, 7.5F, 30.0F};
        camera_.pitch = -14.0F;
        camera_.speed = 8.0F;

        spdlog::info("1-4 1x/2x/4x/8x, 5 sample shading, 6 orbits, 7 stars, "
                     "8 CPU/GPU update, -/= rock count.");
    }

    void on_resize(VkExtent2D extent) override { create_targets(extent); }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
            case SDL_SCANCODE_2:
            case SDL_SCANCODE_3:
            case SDL_SCANCODE_4:
                set_samples(kSampleCounts[event.key.scancode - SDL_SCANCODE_1]);
                break;
            case SDL_SCANCODE_5:
                sample_shading_ = !sample_shading_;
                rebuild_pipelines();
                break;
            case SDL_SCANCODE_6:
                orbiting_ = !orbiting_;
                break;
            case SDL_SCANCODE_7:
                stars_ = !stars_;
                break;
            case SDL_SCANCODE_8:
                gpu_update_ = !gpu_update_;
                break;
            case SDL_SCANCODE_MINUS:
                count_index_ = count_index_ > 0 ? count_index_ - 1 : 0;
                break;
            case SDL_SCANCODE_EQUALS:
                count_index_ = std::min(count_index_ + 1, kRockCounts.size() - 1);
                break;
            default:
                return;
        }
        reset_stats();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, orbit_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, compute_layout_, nullptr);
        destroy_pipelines();
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        render_timer_.destroy();
        update_timer_.destroy();
        for (FrameArena& arena : arenas_) {
            arena.destroy();
        }
        colour_.destroy();
        depth_.destroy();
        instances_.destroy();
        rock_.destroy();
        planet_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());
        if (orbiting_) {
            orbit_time_ += delta_time();
        }

        const uint32_t slot = frame.frame_number % vkc::kFramesInFlight;
        record_gpu_times(update_timer_.read(slot), render_timer_.read(slot));

        FrameArena& arena = arenas_[slot];
        arena.reset();
        const VkDeviceAddress globals = write_globals(arena, frame.extent);

        // Every rock's matrices for this moment, one way or the other, before any
        // rendering: a dispatch is not allowed inside a rendering pass.
        update_timer_.reset(frame.cmd, slot);
        render_timer_.reset(frame.cmd, slot);
        const auto cpu_start = std::chrono::steady_clock::now();
        const VkDeviceAddress instances =
            gpu_update_ ? update_on_gpu(frame.cmd, slot) : update_on_cpu(arena);
        const auto cpu_end = std::chrono::steady_clock::now();
        record_cpu_time(
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());

        render_timer_.begin(frame.cmd, slot);
        begin_rendering(frame);
        set_viewport(frame.cmd, frame.extent);

        push(frame.cmd, globals, instances, planet_.vertex_address);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, planet_pipeline_);
        vkCmdBindIndexBuffer(frame.cmd, planet_.indices.handle(), 0,
                             VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(frame.cmd, planet_.index_count, 1, 0, 0, kPlanetInstance);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rock_pipeline_);
        vkCmdBindIndexBuffer(frame.cmd, rock_.indices.handle(), 0, VK_INDEX_TYPE_UINT16);
        push(frame.cmd, globals, instances, rock_.vertex_address);
        vkCmdDrawIndexed(frame.cmd, rock_.index_count, rock_count(), 0, 0, kFirstRock);

        if (stars_) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              stars_pipeline_);
            vkCmdDraw(frame.cmd, 3, 1, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
        render_timer_.end(frame.cmd, slot);
    }

private:
    [[nodiscard]] uint32_t rock_count() const { return kRockCounts[count_index_]; }

    [[nodiscard]] bool multisampled() const { return samples_ != VK_SAMPLE_COUNT_1_BIT; }

    // ---------------------------------------------------------------------------
    // Updating the rocks
    // ---------------------------------------------------------------------------

    // One thread per rock, in workgroups of 64, then a barrier so that the vertex shader
    // reads what the compute shader wrote. Returns where the instances are.
    [[nodiscard]] VkDeviceAddress update_on_gpu(VkCommandBuffer cmd, uint32_t slot) {
        update_timer_.begin(cmd, slot);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, orbit_pipeline_);
        const OrbitPush push{
            .instances = instances_address_,
            .first = kFirstRock,
            .count = rock_count(),
            .time = orbit_time_,
        };
        vkCmdPushConstants(cmd, compute_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        // Enough workgroups to cover every rock: the count divided by 64, rounded up.
        vkCmdDispatch(cmd, (rock_count() + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);

        // The compute shader's writes, made available and then visible to the vertex
        // shader's reads -- and to the next frame's dispatch, which writes the same
        // buffer again: a barrier's second scope is everything after it in submission
        // order, the next command buffer included. A buffer barrier, because what is
        // being protected is a buffer.
        const VkBufferMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = instances_.handle(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
        };
        vkCmdPipelineBarrier2(cmd, &dependency);

        update_timer_.end(cmd, slot);
        return instances_address_;
    }

    // The same arithmetic in a loop, into this frame's arena. No barrier: the CPU's
    // writes are finished before the command buffer is even submitted, and submission
    // makes writes to host-coherent memory visible to the device. Every host-visible
    // memory type this GPU offers is coherent; 1.10 said what to do on one that is not.
    [[nodiscard]] VkDeviceAddress update_on_cpu(FrameArena& arena) {
        cpu_instances_.clear();
        cpu_instances_.push_back(instance_of(glm::mat4(1.0F), kPlanetColour));
        for (uint32_t i = 0; i < rock_count(); ++i) {
            cpu_instances_.push_back(rock_at(i, orbit_time_));
        }
        return arena.write(cpu_instances_.data(),
                           cpu_instances_.size() * sizeof(Instance));
    }

    // ---------------------------------------------------------------------------
    // Rendering into many samples, and resolving them into one
    // ---------------------------------------------------------------------------

    // vkc::begin_rendering from 1.13 cannot say "resolve", so the rendering info is
    // written out here, as it was in 1.13 before it moved into vkcommon.
    //
    // At 1x the swapchain image is the colour attachment, as it has been all series.
    // Multisampled, the frame renders into colour_ -- several samples per pixel -- and
    // the resolve fields say what to do with them when rendering ends: average each
    // pixel's samples and write the result into the swapchain image. After that the
    // samples themselves are never read again, so they are not stored.
    void begin_rendering(const vkc::FrameInfo& frame) {
        const VkRenderingAttachmentInfo colour{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = multisampled() ? colour_.view() : frame.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode =
                multisampled() ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
            .resolveImageView = multisampled() ? frame.view : VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = multisampled() ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                      : VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.color = {{0.01F, 0.012F, 0.02F, 1.0F}}},
        };

        // Depth is multisampled too -- each sample is depth-tested on its own -- and is
        // never resolved. Nothing reads it after the pass.
        const VkRenderingAttachmentInfo depth{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = depth_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue = {.depthStencil = {1.0F, 0}},
        };

        const VkRenderingInfo rendering{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {{0, 0}, frame.extent},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour,
            .pDepthAttachment = &depth,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.cmd, &rendering);
    }

    // The attachments for the current sample count and window size.
    //
    // TRANSIENT_ATTACHMENT says the contents never outlive a rendering pass: they are
    // cleared at the start, and not stored at the end. A GPU that renders in tiles can
    // keep such an image entirely in on-chip memory and never give it real memory at all,
    // if it is bound to a memory type with VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT. The
    // article looks at whether this GPU has one.
    void create_targets(VkExtent2D extent) {
        context().wait_idle();

        depth_ = vkc::Image(context(),
                            vkc::ImageDesc{
                                .format = depth_format_,
                                .extent = extent,
                                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                         VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                                .mip_levels = 1,
                                .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .samples = samples_,
                            });
        colour_.destroy();
        if (multisampled()) {
            colour_ = vkc::Image(context(),
                                 vkc::ImageDesc{
                                     .format = swapchain().format(),
                                     .extent = extent,
                                     .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                              VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                                     .mip_levels = 1,
                                     .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                     .samples = samples_,
                                 });
        }

        // Into their attachment layouts once, as 1.13 did for the depth buffer. Every
        // frame clears them, so nothing they held before matters.
        vkc::immediate_submit(context(), [&](VkCommandBuffer cmd) {
            depth_.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            if (multisampled()) {
                colour_.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
        });
    }

    // A pipeline's sample count is baked in and has to match its attachments', so
    // changing it means new attachments and new pipelines.
    void set_samples(VkSampleCountFlagBits samples) {
        if ((supported_samples_ & samples) == 0) {
            spdlog::info("{}x is not supported here", static_cast<uint32_t>(samples));
            return;
        }
        samples_ = samples;
        create_targets(swapchain().extent());
        rebuild_pipelines();
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
    // ---------------------------------------------------------------------------

    [[nodiscard]] VkDeviceAddress write_globals(FrameArena& arena, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = projection(extent),
            .view_position = glm::vec4(camera_.position, 1.0F),
            .sun_direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
            .sun_colour = glm::vec4(kSunColour, 1.0F),
            .ambient = glm::vec4(kAmbient, kShininess),
        };
        return arena.write(&globals, sizeof(globals));
    }

    void push(VkCommandBuffer cmd, VkDeviceAddress globals, VkDeviceAddress instances,
              VkDeviceAddress vertices) {
        const PushConstants constants{
            .globals = globals,
            .instances = instances,
            .vertices = vertices,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
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
            glm::perspective(glm::radians(45.0F), aspect, 0.1F, 200.0F);
        projection[1][1] *= -1.0F;  // Vulkan's clip-space y points down.
        return projection;
    }

    // ---------------------------------------------------------------------------
    // Measuring
    // ---------------------------------------------------------------------------

    void reset_stats() {
        cpu_ms_ = 0.0;
        update_ms_ = 0.0;
        render_ms_ = 0.0;
        cpu_samples_ = 0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
        spdlog::info("{} rocks, updated on the {}", rock_count(),
                     gpu_update_ ? "GPU" : "CPU");
    }

    void record_cpu_time(double ms) {
        cpu_ms_ += ms;
        ++cpu_samples_;
    }

    // A negative update time means the slot recorded no dispatch -- the CPU path -- and
    // counts as zero; a negative render time means the slot has never been used.
    void record_gpu_times(double update_ms, double render_ms) {
        if (render_ms >= 0.0) {
            update_ms_ += std::max(update_ms, 0.0);
            render_ms_ += render_ms;
            ++gpu_samples_;
        }
        if (elapsed() - stats_started_ < 1.0F || gpu_samples_ == 0 ||
            cpu_samples_ == 0) {
            return;
        }
        spdlog::info("{:>7} rocks, {} update: CPU {:7.3f} ms  GPU update {:6.3f} ms  "
                     "GPU render {:6.3f} ms",
                     rock_count(), gpu_update_ ? "GPU" : "CPU", cpu_ms_ / cpu_samples_,
                     update_ms_ / gpu_samples_, render_ms_ / gpu_samples_);
        reset_stats_quietly();
    }

    void reset_stats_quietly() {
        cpu_ms_ = 0.0;
        update_ms_ = 0.0;
        render_ms_ = 0.0;
        cpu_samples_ = 0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
    }

    // ---------------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------------

    void create_layout() {
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
        VK_CHECK(vkCreatePipelineLayout(context().device(), &layout_info, nullptr,
                                        &pipeline_layout_));
    }

    // 4.7's pipelines, from the shader files that import the modules. load_shader is
    // given the file with the entry points; Slang finds `scene` and `random` itself, in
    // the same directory, because that directory is the session's search path.
    void create_pipelines() {
        const VkDevice device = context().device();
        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "surfaces.slang");

        const auto base = [&]() -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true)
                .multisample(samples_, sample_shading_ ? 1.0F : 0.0F)
                .layout(pipeline_layout_);
        };

        rock_pipeline_ = base().shaders(shader).build();
        planet_pipeline_ =
            base().shaders(shader, "vertexMain", "planetFragmentMain").build();

        vkDestroyShaderModule(device, shader, nullptr);

        // The stars add light to whatever is behind them rather than replacing it:
        // colour = 1 x star + 1 x background, so a pixel with no star in it, whose
        // shader returns zero, is left exactly as the clear colour made it. No depth
        // write, and LESS_OR_EQUAL, because the stars sit at depth 1.0, which is also
        // what the depth buffer was cleared to.
        const VkPipelineColorBlendAttachmentState additive{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        const VkShaderModule stars_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "stars.slang");
        stars_pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(stars_shader, "starsVertexMain", "starsFragmentMain")
                .colour_attachment(swapchain().format())
                .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/false,
                       VK_COMPARE_OP_LESS_OR_EQUAL)
                .colour_blend(additive)
                .multisample(samples_, sample_shading_ ? 1.0F : 0.0F)
                .layout(pipeline_layout_)
                .build();
        vkDestroyShaderModule(device, stars_shader, nullptr);

        context().name(rock_pipeline_, VK_OBJECT_TYPE_PIPELINE, "rocks");
        context().name(planet_pipeline_, VK_OBJECT_TYPE_PIPELINE, "planet");
        context().name(stars_pipeline_, VK_OBJECT_TYPE_PIPELINE, "stars");
    }

    void destroy_pipelines() {
        vkDestroyPipeline(context().device(), stars_pipeline_, nullptr);
        vkDestroyPipeline(context().device(), planet_pipeline_, nullptr);
        vkDestroyPipeline(context().device(), rock_pipeline_, nullptr);
    }

    void rebuild_pipelines() {
        context().wait_idle();
        destroy_pipelines();
        create_pipelines();
    }

    // A compute pipeline is one shader stage and a layout, and nothing else: no vertex
    // input, no rasteriser, no attachments, no blending. PipelineBuilder builds graphics
    // pipelines, so this one is written out.
    void create_compute_pipeline() {
        const VkDevice device = context().device();

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(OrbitPush),
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
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &compute_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "orbits.slang");
        const VkComputePipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage =
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = shader,
                    .pName = "computeMain",
                    .pSpecializationInfo = nullptr,
                },
            .layout = compute_layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                          nullptr, &orbit_pipeline_));
        vkDestroyShaderModule(device, shader, nullptr);

        context().name(orbit_pipeline_, VK_OBJECT_TYPE_PIPELINE, "orbits");
    }

    vkc::Camera camera_;

    Mesh planet_;
    Mesh rock_;
    vkc::Buffer instances_;
    VkDeviceAddress instances_address_ = 0;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlags supported_samples_ = 0;
    VkSampleCountFlagBits samples_ = kDefaultSamples;
    bool sample_shading_ = false;
    vkc::Image colour_;
    vkc::Image depth_;

    std::array<FrameArena, vkc::kFramesInFlight> arenas_;
    std::vector<Instance> cpu_instances_;
    vkc::GpuTimer update_timer_;
    vkc::GpuTimer render_timer_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline rock_pipeline_ = VK_NULL_HANDLE;
    VkPipeline planet_pipeline_ = VK_NULL_HANDLE;
    VkPipeline stars_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout compute_layout_ = VK_NULL_HANDLE;
    VkPipeline orbit_pipeline_ = VK_NULL_HANDLE;

    size_t count_index_ = kDefaultRockCount;
    bool orbiting_ = false;
    bool stars_ = true;
    bool gpu_update_ = true;
    float orbit_time_ = 0.0F;

    double cpu_ms_ = 0.0;
    double update_ms_ = 0.0;
    double render_ms_ = 0.0;
    uint32_t cpu_samples_ = 0;
    uint32_t gpu_samples_ = 0;
    float stats_started_ = 0.0F;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.9 - Compute";
        ComputeApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
