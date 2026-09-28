// 4.8 Slang modules
//
// 4.7's belt, with a sky full of stars behind it -- and underneath, shaders that stop
// repeating themselves.
//
// Until now each chapter had one shader file, or two that declared the same structs word
// for word. This chapter's shaders are four files: two modules that hold what is shared,
// and two that import them. The C++ changes very little: which files it loads, and one
// more pipeline for the stars.
//
//   1 2 3 4   1x, 2x, 4x, 8x multisampling
//   5         sample shading
//   6         spin the belt
//   7         stars on or off
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

// Radians per second. The belt turns as one rigid disc, so the outer edge moves fastest:
// 0.05 x 17 = 0.85 units a second.
constexpr float kBeltSpin = 0.05F;

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

constexpr VkDeviceSize kArenaBytes = 4 * 1024;

// ---------------------------------------------------------------------------
// The sample
// ---------------------------------------------------------------------------

class SlangModulesApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        const auto [sphere_vertices, sphere_indices] = make_sphere(48, 96, kPlanetRadius);
        planet_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        const auto [rock_vertices, rock_indices] = make_rock(2, 0.18F, 7);
        rock_ = upload_mesh(context(), rock_vertices, rock_indices);

        const std::vector<Instance> instances = make_instances();
        // Only SHADER_DEVICE_ADDRESS now: the instance-attribute path is gone, and
        // nothing binds this buffer as a vertex buffer any more.
        instances_ = vkc::upload_to_device_local(
            context(), instances.data(), instances.size() * sizeof(Instance),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        instances_address_ = buffer_address(context().device(), instances_.handle());

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
        timer_.create(context());
        create_layout();
        create_targets(swapchain().extent());
        create_pipelines();

        camera_.position = {0.0F, 7.5F, 30.0F};
        camera_.pitch = -14.0F;
        camera_.speed = 8.0F;

        spdlog::info("1-4 1x/2x/4x/8x, 5 sample shading, 6 spin the belt, 7 stars, "
                     "-/= rock count.");
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
                spinning_ = !spinning_;
                break;
            case SDL_SCANCODE_7:
                stars_ = !stars_;
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

        destroy_pipelines();
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        timer_.destroy();
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
        if (spinning_) {
            belt_angle_ += kBeltSpin * delta_time();
        }

        const uint32_t slot = frame.frame_number % vkc::kFramesInFlight;
        record_gpu_time(timer_.read(slot));

        FrameArena& arena = arenas_[slot];
        arena.reset();
        const VkDeviceAddress globals = write_globals(arena, frame.extent);

        // The whole rendering pass is timed this time, because the resolve happens at
        // its end, inside vkCmdEndRendering, and is part of what multisampling costs.
        timer_.reset(frame.cmd, slot);
        timer_.begin(frame.cmd, slot);
        begin_rendering(frame);
        set_viewport(frame.cmd, frame.extent);

        push(frame.cmd, globals, planet_.vertex_address);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, planet_pipeline_);
        vkCmdBindIndexBuffer(frame.cmd, planet_.indices.handle(), 0,
                             VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(frame.cmd, planet_.index_count, 1, 0, 0, kPlanetInstance);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rock_pipeline_);
        vkCmdBindIndexBuffer(frame.cmd, rock_.indices.handle(), 0, VK_INDEX_TYPE_UINT16);
        push(frame.cmd, globals, rock_.vertex_address);
        vkCmdDrawIndexed(frame.cmd, rock_.index_count, rock_count(), 0, 0, kFirstRock);

        // Last, so the depth test can reject every pixel something nearer already covers.
        // Three vertices and no buffers: the shader makes the triangle from its index.
        if (stars_) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              stars_pipeline_);
            vkCmdDraw(frame.cmd, 3, 1, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
        timer_.end(frame.cmd, slot);
    }

private:
    [[nodiscard]] uint32_t rock_count() const { return kRockCounts[count_index_]; }

    [[nodiscard]] bool multisampled() const { return samples_ != VK_SAMPLE_COUNT_1_BIT; }

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
        gpu_ms_ = 0.0;
        gpu_samples_ = 0;
        stats_started_ = elapsed();
        spdlog::info("{} rocks, {}x{}", rock_count(), static_cast<uint32_t>(samples_),
                     sample_shading_ ? ", sample shading" : "");
    }

    void record_gpu_time(double ms) {
        if (ms >= 0.0) {
            gpu_ms_ += ms;
            ++gpu_samples_;
        }
        if (elapsed() - stats_started_ < 1.0F || gpu_samples_ == 0) {
            return;
        }
        spdlog::info("{:>7} rocks, {}x{:<16} GPU {:7.3f} ms", rock_count(),
                     static_cast<uint32_t>(samples_),
                     sample_shading_ ? ", sample shading" : "", gpu_ms_ / gpu_samples_);
        gpu_ms_ = 0.0;
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
    vkc::GpuTimer timer_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline rock_pipeline_ = VK_NULL_HANDLE;
    VkPipeline planet_pipeline_ = VK_NULL_HANDLE;
    VkPipeline stars_pipeline_ = VK_NULL_HANDLE;

    size_t count_index_ = kDefaultRockCount;
    bool spinning_ = false;
    bool stars_ = true;
    float belt_angle_ = 0.0F;

    double gpu_ms_ = 0.0;
    uint32_t gpu_samples_ = 0;
    float stats_started_ = 0.0F;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.8 - Slang modules";
        SlangModulesApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
