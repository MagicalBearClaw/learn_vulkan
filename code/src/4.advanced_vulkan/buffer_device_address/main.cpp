// 4.5 Buffer device address & dynamic data
//
// The scene is 4.4's, pixel for pixel. What has changed is how the shaders are told
// where anything is.
//
// Until now every buffer reached a shader the same way: it was written into a
// descriptor, the descriptor went into a set, the set was bound to a slot, and the
// shader named the slot. Chapter 4.4 needed fourteen uniform buffers and fourteen
// descriptor sets to say seven things per frame, and every one of those sets had to be
// allocated, written and kept alive for as long as the GPU might still be reading it.
//
// A buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT has an address --
// an ordinary 64-bit number in the GPU's address space, which vkGetBufferDeviceAddress
// hands back. Put that number in a push constant and the shader can dereference it.
// There is no descriptor, no set, no layout binding and nothing to keep in step.
//
// Three consequences run through this file:
//
//   * the per-frame uniform buffers and their descriptor sets are gone, replaced by one
//     bump allocator per frame in flight (FrameArena below) that the frame writes into
//     from the top and hands out addresses within;
//
//   * the vertex buffers are no longer vertex buffers. Nothing is bound with
//     vkCmdBindVertexBuffers, no pipeline declares a vertex attribute, and every vertex
//     shader reads its own vertex out of memory through a pointer;
//
//   * the only descriptors left are the textures, because an image has no address. That
//     asymmetry is the subject of the article.
//
//   1  sway the ferns, to show the per-frame data is genuinely per-frame
//   2  report what this frame put in the arena, and at which addresses

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
#include <filesystem>
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

// 2.4's cube. It is two things here: the crates, and the skybox.
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
    glm::vec3 position;
    float angle;
    glm::vec2 size;
};

constexpr std::array<Placement, 2> kCrates{{
    {{-2.20F, kFloorY, 0.75F}, 0.36F, {1.0F, 1.0F}},
    {{2.05F, kFloorY, -1.05F}, -0.72F, {1.0F, 1.0F}},
}};

constexpr std::array<Placement, 6> kFerns{{
    {{-3.30F, kFloorY, 1.70F}, 0.30F, {2.0F, 2.0F}},
    {{-2.10F, kFloorY, 2.90F}, -0.55F, {1.7F, 1.7F}},
    {{-2.70F, kFloorY, 2.30F}, 0.95F, {1.5F, 1.5F}},
    {{2.55F, kFloorY, 2.10F}, 0.80F, {2.1F, 2.1F}},
    {{3.40F, kFloorY, 1.30F}, -0.20F, {1.6F, 1.6F}},
    {{0.35F, kFloorY, 3.30F}, 0.15F, {1.8F, 1.8F}},
}};

constexpr glm::vec4 kWhite{1.0F};

constexpr glm::vec3 kSunDirection{-0.45F, -1.0F, -0.38F};
constexpr glm::vec3 kSunColour{0.92F, 0.88F, 0.80F};
constexpr glm::vec3 kAmbient{0.24F, 0.26F, 0.31F};
constexpr float kShininess = 48.0F;

constexpr float kMirrorRadius = 0.9F;
constexpr glm::vec3 kMirrorCentre{0.0F, kFloorY + kMirrorRadius, 0.0F};
constexpr uint32_t kFaceSize = 512;

// How far a fern leans at the ends of its sway, in radians. Small on purpose: the point
// of the key is that per-object data is rebuilt from nothing every frame, not that the
// ferns are interesting.
constexpr float kSwayAngle = 0.07F;

constexpr size_t kFacePassCount = 6;
constexpr size_t kPassCount = kFacePassCount + 1;
constexpr size_t kScreenPass = kFacePassCount;

struct CubeFace {
    glm::vec3 forward;
    glm::vec3 up;
};
constexpr std::array<CubeFace, 6> kCubeFaces{{
    {{1.0F, 0.0F, 0.0F}, {0.0F, -1.0F, 0.0F}},
    {{-1.0F, 0.0F, 0.0F}, {0.0F, -1.0F, 0.0F}},
    {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}},
    {{0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, -1.0F}},
    {{0.0F, 0.0F, 1.0F}, {0.0F, -1.0F, 0.0F}},
    {{0.0F, 0.0F, -1.0F}, {0.0F, -1.0F, 0.0F}},
}};

constexpr std::array<const char*, 6> kSkyFiles{
    "textures/lvk_sky_px.png", "textures/lvk_sky_nx.png", "textures/lvk_sky_py.png",
    "textures/lvk_sky_ny.png", "textures/lvk_sky_pz.png", "textures/lvk_sky_nz.png",
};

// ---------------------------------------------------------------------------
// What the shaders read through a pointer
// ---------------------------------------------------------------------------

// Byte for byte 4.4's uniform block, and no longer a uniform block. Nothing about the
// contents changed, because nothing needed to: a struct that satisfied std140 also
// satisfies the plainer rule a pointer follows.
struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::mat4 view_rotation;  // 128
    glm::vec4 view_position;  // 192
    glm::vec4 sun_direction;  // 208
    glm::vec4 sun_colour;     // 224
    glm::vec4 ambient;        // 240, w is the specular exponent
};
static_assert(sizeof(Globals) == 256);

// One object's share of the frame. This is 4.4's push-constant block, moved out of the
// command buffer and into memory: the same three fields, written once per frame per
// object instead of re-pushed for every one of the seven passes that draws it.
//
// The normal matrix is a glm::mat3 here and was three glm::vec4 in 4.4. That is not a
// tidy-up, it is the layout rule changing. A uniform block obeys std140, where a
// three-float column is rounded up to sixteen bytes, so the C++ side had to pad by hand
// to match. A struct reached through a pointer is laid out the way Slang lays out a
// struct -- tightly, exactly as a C++ compiler would -- so a 36-byte glm::mat3 and a
// 36-byte float3x3 are the same 36 bytes and the padding simply stops existing.
//
// The static_asserts are the check. If glm ever aligned its vectors differently, or
// Slang changed its mind about packing, this stops compiling rather than rendering
// something subtly wrong.
struct DrawData {
    glm::mat4 model;           // 0
    glm::mat3 normal_matrix;   // 64
    glm::vec4 colour;          // 100
};
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, colour) == 100);
static_assert(sizeof(DrawData) == 116);

// The whole of what a draw call says now. Three addresses and an index, against 4.4's
// 128 bytes of matrices -- and unlike those matrices, none of this has to be rebuilt
// for each of the seven passes, because the addresses do not change between them.
//
// VkDeviceAddress is a uint64_t. Nothing in the type system distinguishes an address
// into the globals from an address into the vertices, which is the honest cost of this
// technique and the reason the three fields are named as carefully as they are.
struct PushConstants {
    VkDeviceAddress globals;   // 0
    VkDeviceAddress draws;     // 8
    VkDeviceAddress vertices;  // 16
    uint32_t draw_index;       // 24
};
static_assert(offsetof(PushConstants, draw_index) == 24);
static_assert(sizeof(PushConstants) == 32);

// Where each object's DrawData sits in the array the frame writes. The order is fixed by
// build_draw_list() below and nothing else may reorder it.
constexpr uint32_t kFloorDraw = 0;
constexpr uint32_t kFirstCrateDraw = kFloorDraw + 1;
constexpr uint32_t kFirstFernDraw =
    kFirstCrateDraw + static_cast<uint32_t>(kCrates.size());
constexpr uint32_t kSphereDraw = kFirstFernDraw + static_cast<uint32_t>(kFerns.size());
constexpr uint32_t kDrawCount = kSphereDraw + 1;

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

// The three lines this chapter is about.
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

    [[nodiscard]] VkDeviceAddress base() const noexcept { return base_; }
    [[nodiscard]] VkDeviceSize used() const noexcept { return used_; }
    [[nodiscard]] VkDeviceSize capacity() const noexcept { return capacity_; }

private:
    vkc::Buffer buffer_;
    VkDeviceAddress base_ = 0;
    VkDeviceSize used_ = 0;
    VkDeviceSize capacity_ = 0;
};

// 64 KiB, against the roughly 3 KiB a frame actually uses. Arena sizing is a guess that
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

// ---------------------------------------------------------------------------
// The live cubemap: 4.4's, with the loading half deleted
// ---------------------------------------------------------------------------

// 4.4 wrote this out in full and explained every field. The sky cubemap it also built by
// hand now comes from vkc::load_cubemap, which is where that half of the chapter ended
// up; the six single-layer views have no home in vkcommon, so this stays here.
struct Cubemap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView cube_view = VK_NULL_HANDLE;
    std::array<VkImageView, 6> face_views{};

    void destroy(VkDevice device, VmaAllocator allocator) noexcept {
        for (VkImageView& view : face_views) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, view, nullptr);
                view = VK_NULL_HANDLE;
            }
        }
        if (cube_view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, cube_view, nullptr);
            cube_view = VK_NULL_HANDLE;
        }
        if (image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator, image, allocation);
            image = VK_NULL_HANDLE;
            allocation = VK_NULL_HANDLE;
        }
    }
};

[[nodiscard]] Cubemap create_cubemap(vkc::Context& context, VkFormat format,
                                     uint32_t size, VkImageUsageFlags usage) {
    Cubemap cube;

    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {size, size, 1},
        .mipLevels = 1,
        .arrayLayers = 6,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo alloc_info{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };
    VK_CHECK(vmaCreateImage(context.allocator(), &image_info, &alloc_info, &cube.image,
                            &cube.allocation, nullptr));

    const VkImageViewCreateInfo cube_view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = cube.image,
        .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
        .format = format,
        .components = {},
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6,
            },
    };
    VK_CHECK(vkCreateImageView(context.device(), &cube_view_info, nullptr,
                               &cube.cube_view));

    for (uint32_t face = 0; face < 6; ++face) {
        const VkImageViewCreateInfo face_view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = cube.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {},
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = face,
                    .layerCount = 1,
                },
        };
        VK_CHECK(vkCreateImageView(context.device(), &face_view_info, nullptr,
                                   &cube.face_views[face]));
    }
    return cube;
}

// The two pipelines whose front face depends on what is being drawn into. 4.4 explains
// why a cube face needs the opposite winding from the screen.
struct ScenePipelines {
    VkPipeline opaque = VK_NULL_HANDLE;
    VkPipeline sky = VK_NULL_HANDLE;
};

class BufferDeviceAddressApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        cube_ = upload_mesh(context(), kCubeVertices, kCubeIndices);
        quad_ = upload_mesh(context(), kQuadVertices, kQuadIndices);

        const auto [sphere_vertices, sphere_indices] = make_sphere(48, 96, kMirrorRadius);
        sphere_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        depth_format_ = vkc::choose_depth_format(context().physical_device());
        cube_format_ = swapchain().format();

        load_textures();
        create_cubemaps();
        create_arenas();
        create_descriptors();
        create_depth_buffer(swapchain().extent());
        create_pipelines();

        draws_.reserve(kDrawCount);

        camera_.position = {0.0F, 1.80F, 5.60F};
        camera_.pitch = -6.0F;

        spdlog::info("1 sway the ferns, 2 report this frame's arena.");
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                sway_ = !sway_;
                spdlog::info("Ferns {}", sway_ ? "swaying" : "still");
                break;
            case SDL_SCANCODE_2:
                report_ = true;
                break;
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();
        const VmaAllocator allocator = context().allocator();

        vkDestroyPipeline(device, mirror_pipeline_, nullptr);
        vkDestroyPipeline(device, face_pipelines_.sky, nullptr);
        vkDestroyPipeline(device, screen_pipelines_.sky, nullptr);
        vkDestroyPipeline(device, fern_pipeline_, nullptr);
        vkDestroyPipeline(device, face_pipelines_.opaque, nullptr);
        vkDestroyPipeline(device, screen_pipelines_.opaque, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, texture_set_layout_, nullptr);

        live_cube_.destroy(device, allocator);
        sky_cube_.destroy();

        cube_sampler_.destroy();
        clamp_sampler_.destroy();
        sampler_.destroy();
        fern_map_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();

        for (FrameArena& arena : arenas_) {
            arena.destroy();
        }
        face_depth_.destroy();
        depth_.destroy();
        sphere_.destroy();
        quad_.destroy();
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
        pushes_ = 0;

        write_globals(arena, frame.extent);
        write_draws(arena);

        render_cube_faces(frame);
        render_screen(frame);

        if (report_) {
            report(arena);
            report_ = false;
        }
    }

private:
    // ---------------------------------------------------------------------------
    // Filling the arena
    // ---------------------------------------------------------------------------

    // Seven sets of per-pass constants, one per cube face and one for the screen. In 4.4
    // these lived in seven separate VkBuffers with seven descriptor sets pointing at
    // them, times two for the frames in flight. Here they are seven consecutive structs
    // in one allocation, and what the shader is given is where each one starts.
    void write_globals(FrameArena& arena, VkExtent2D screen_extent) {
        const auto write = [&](size_t pass, const glm::mat4& view,
                               const glm::mat4& projection, const glm::vec3& eye) {
            const Globals globals{
                .view = view,
                .projection = projection,
                .view_rotation = glm::mat4(glm::mat3(view)),
                .view_position = glm::vec4(eye, 1.0F),
                .sun_direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
                .sun_colour = glm::vec4(kSunColour, 1.0F),
                .ambient = glm::vec4(kAmbient, kShininess),
            };
            globals_[pass] = arena.write(&globals, sizeof(globals));
        };

        for (size_t face = 0; face < kFacePassCount; ++face) {
            const glm::mat4 view =
                glm::lookAt(kMirrorCentre, kMirrorCentre + kCubeFaces[face].forward,
                            kCubeFaces[face].up);
            write(face, view, face_projection(), kMirrorCentre);
        }

        write(kScreenPass, camera_.view(), screen_projection(screen_extent),
              camera_.position);
    }

    // The per-object data, rebuilt from nothing every frame and written as one array.
    //
    // Rebuilding a static scene every frame looks wasteful and is the point: the cost of
    // a frame's worth of this data is one memcpy into mapped memory, so there is no
    // reason to distinguish the objects that moved from the ones that did not. Press 1
    // and the ferns start swaying without a single line here changing shape.
    void write_draws(FrameArena& arena) {
        draws_.clear();

        draws_.push_back(draw_data_of(glm::mat4(1.0F), kWhite));

        for (const Placement& crate : kCrates) {
            draws_.push_back(draw_data_of(
                glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F)), kWhite));
        }

        for (size_t i = 0; i < kFerns.size(); ++i) {
            glm::mat4 model = model_of(kFerns[i]);
            if (sway_) {
                // Around the base of the quad, so the fern leans rather than slides.
                const float phase = elapsed() * 1.6F + static_cast<float>(i);
                model = glm::rotate(model, kSwayAngle * std::sin(phase),
                                    glm::vec3(0.0F, 0.0F, 1.0F));
            }
            draws_.push_back(draw_data_of(model, kWhite));
        }

        draws_.push_back(
            draw_data_of(glm::translate(glm::mat4(1.0F), kMirrorCentre), kWhite));

        // One write for every object in the scene. The shader is handed the address of
        // element zero and picks its own element out with the index in the push
        // constant, which is why the order above is fixed by the kFloorDraw constants
        // and not by convenience.
        draws_address_ = arena.write(draws_.data(), draws_.size() * sizeof(DrawData));
    }

    // ---------------------------------------------------------------------------
    // Six passes into six faces of one image. 4.4, unchanged.
    // ---------------------------------------------------------------------------

    void render_cube_faces(const vkc::FrameInfo& frame) {
        vkc::image_barrier(frame.cmd, live_cube_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        const VkExtent2D face_extent{kFaceSize, kFaceSize};

        for (size_t face = 0; face < kFacePassCount; ++face) {
            if (face > 0) {
                vkc::image_barrier(frame.cmd, face_depth_.handle(),
                                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                   VK_IMAGE_ASPECT_DEPTH_BIT);
            }

            vkc::begin_rendering(frame.cmd, live_cube_.face_views[face],
                                 face_depth_.view(), face_extent,
                                 {{0.05F, 0.06F, 0.09F, 1.0F}});
            set_viewport(frame.cmd, face_extent);

            // The only thing that distinguishes this pass from the other six: which of
            // the seven addresses written into the arena is pushed. There is no
            // vkCmdBindDescriptorSets for it and nothing to allocate per pass.
            draw_scene(frame.cmd, face_pipelines_, globals_[face]);
            draw_sky(frame.cmd, face_pipelines_, globals_[face]);

            vkCmdEndRendering(frame.cmd);
        }

        vkc::image_barrier(frame.cmd, live_cube_.image,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    void render_screen(const vkc::FrameInfo& frame) {
        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.05F, 0.06F, 0.09F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        draw_scene(frame.cmd, screen_pipelines_, globals_[kScreenPass]);
        draw_mirror(frame.cmd, globals_[kScreenPass]);
        draw_sky(frame.cmd, screen_pipelines_, globals_[kScreenPass]);

        vkCmdEndRendering(frame.cmd);
    }

    void draw_scene(VkCommandBuffer cmd, const ScenePipelines& pipelines,
                    VkDeviceAddress globals) {
        bind_indices(cmd, quad_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.opaque);
        bind_texture(cmd, floor_set_);
        push(cmd, globals, quad_.vertex_address, kFloorDraw);
        vkCmdDrawIndexed(cmd, kQuadIndexCount, 1, kFloorFirstIndex, 0, 0);

        bind_indices(cmd, cube_);
        bind_texture(cmd, crate_set_);
        for (uint32_t i = 0; i < kCrates.size(); ++i) {
            push(cmd, globals, cube_.vertex_address, kFirstCrateDraw + i);
            vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
        }

        bind_indices(cmd, quad_);
        bind_texture(cmd, fern_set_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fern_pipeline_);
        for (uint32_t i = 0; i < kFerns.size(); ++i) {
            push(cmd, globals, quad_.vertex_address, kFirstFernDraw + i);
            vkCmdDrawIndexed(cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }
    }

    void draw_mirror(VkCommandBuffer cmd, VkDeviceAddress globals) {
        bind_indices(cmd, sphere_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mirror_pipeline_);
        bind_texture(cmd, live_set_);
        push(cmd, globals, sphere_.vertex_address, kSphereDraw);
        vkCmdDrawIndexed(cmd, sphere_.index_count, 1, 0, 0, 0);
    }

    void draw_sky(VkCommandBuffer cmd, const ScenePipelines& pipelines,
                  VkDeviceAddress globals) {
        bind_indices(cmd, cube_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.sky);
        bind_texture(cmd, sky_set_);
        // The skybox reads no DrawData at all -- its model matrix is the identity and it
        // is never anywhere else -- so the index it is given is never dereferenced.
        push(cmd, globals, cube_.vertex_address, kFloorDraw);
        vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
    // ---------------------------------------------------------------------------

    void push(VkCommandBuffer cmd, VkDeviceAddress globals, VkDeviceAddress vertices,
              uint32_t draw_index) {
        const PushConstants constants{
            .globals = globals,
            .draws = draws_address_,
            .vertices = vertices,
            .draw_index = draw_index,
        };
        vkCmdPushConstants(cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);
        ++pushes_;
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

    [[nodiscard]] static glm::mat4 screen_projection(VkExtent2D extent) {
        const float aspect =
            static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 projection =
            glm::perspective(glm::radians(45.0F), aspect, 0.1F, 100.0F);
        projection[1][1] *= -1.0F;  // Vulkan's clip-space y points down.
        return projection;
    }

    // No Y-flip, for the reason 4.4 spends a section on: a cube face is sampled by
    // direction, and its v axis already runs the way unflipped Vulkan clip space does.
    [[nodiscard]] static glm::mat4 face_projection() {
        return glm::perspective(glm::radians(90.0F), 1.0F, 0.1F, 100.0F);
    }

    // What the frame actually cost, printed rather than reasoned about.
    void report(const FrameArena& arena) const {
        spdlog::info("arena: {} of {} bytes used -- {} globals x {} B, {} draws x {} B",
                     arena.used(), arena.capacity(), kPassCount, sizeof(Globals),
                     draws_.size(), sizeof(DrawData));
        spdlog::info("       {} push constants x {} B recorded into the command buffer",
                     pushes_, sizeof(PushConstants));
        spdlog::info("addresses: arena base 0x{:x}, globals[0] 0x{:x}, draws 0x{:x}",
                     arena.base(), globals_[0], draws_address_);
        spdlog::info("           cube vertices 0x{:x}, quad 0x{:x}, sphere 0x{:x}",
                     cube_.vertex_address, quad_.vertex_address, sphere_.vertex_address);
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
        fern_map_ = vkc::load_texture(context(),
                                      vkc::asset_path("textures/lvk_foliage.png"),
                                      VK_FORMAT_R8G8B8A8_SRGB);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});
        clamp_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE});
        cube_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .anisotropy = false});
    }

    void create_cubemaps() {
        // 4.4 wrote the six-file load out by hand; it lives in vkcommon now.
        std::array<std::filesystem::path, 6> sky_faces;
        for (size_t face = 0; face < sky_faces.size(); ++face) {
            sky_faces[face] = vkc::asset_path(kSkyFiles[face]);
        }
        sky_cube_ = vkc::load_cubemap(context(), sky_faces, VK_FORMAT_R8G8B8A8_SRGB);

        live_cube_ = create_cubemap(context(), cube_format_, kFaceSize,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT);

        face_depth_ = vkc::create_depth_buffer(context(), depth_format_,
                                               {kFaceSize, kFaceSize});

        context().name(sky_cube_.handle(), VK_OBJECT_TYPE_IMAGE, "sky cubemap");
        context().name(live_cube_.image, VK_OBJECT_TYPE_IMAGE, "live cubemap");
    }

    // Five descriptor sets, all of them textures, and one descriptor set layout.
    //
    // 4.4 had two layouts and nineteen sets: fourteen of them -- two frames in flight
    // times seven passes -- existed only to point at a uniform buffer. Those are what
    // addresses replaced; these five are what they cannot replace. An image is
    // read through hardware that needs a descriptor -- format, tiling, swizzle and
    // filtering live in it, and none of those things are expressible as a number a
    // shader can add to.
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

        constexpr uint32_t kTextureSets = 5;  // crate, floor, fern, sky, live
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
        fern_set_ = allocate_texture_set(fern_map_.view(), clamp_sampler_.handle());
        sky_set_ = allocate_texture_set(sky_cube_.view(), cube_sampler_.handle());
        live_set_ = allocate_texture_set(live_cube_.cube_view, cube_sampler_.handle());
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

        const auto base = [&](VkShaderModule module) -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(module)
                .colour_attachment(cube_format_)
                .layout(pipeline_layout_);
        };

        const VkShaderModule scene_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "scene.slang");

        // Two windings, for the screen and for a cube face. 4.4's caution applies
        // unchanged: a cube face is stored mirrored, so the triangles that are
        // counter-clockwise on screen are clockwise in a face.
        for (size_t pass = 0; pass < 2; ++pass) {
            const VkFrontFace winding =
                pass == 0 ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            const VkPipeline opaque = base(scene_shader)
                                          .cull(VK_CULL_MODE_BACK_BIT, winding)
                                          .depth(depth_format_, /*test=*/true,
                                                 /*write=*/true)
                                          .build();
            (pass == 0 ? screen_pipelines_ : face_pipelines_).opaque = opaque;
        }

        fern_pipeline_ = base(scene_shader)
                             .shaders(scene_shader, "vertexMain", "cutoutFragmentMain")
                             .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                             .depth(depth_format_, /*test=*/true, /*write=*/true)
                             .build();

        vkDestroyShaderModule(device, scene_shader, nullptr);

        const VkShaderModule environment_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "environment.slang");

        for (size_t pass = 0; pass < 2; ++pass) {
            const VkFrontFace winding =
                pass == 0 ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            const VkPipeline sky = base(environment_shader)
                                       .cull(VK_CULL_MODE_FRONT_BIT, winding)
                                       .depth(depth_format_, /*test=*/true,
                                              /*write=*/false,
                                              VK_COMPARE_OP_LESS_OR_EQUAL)
                                       .build();
            (pass == 0 ? screen_pipelines_ : face_pipelines_).sky = sky;
        }

        mirror_pipeline_ = base(environment_shader)
                               .shaders(environment_shader, "mirrorVertexMain",
                                        "mirrorFragmentMain")
                               .cull(VK_CULL_MODE_BACK_BIT,
                                     VK_FRONT_FACE_COUNTER_CLOCKWISE)
                               .depth(depth_format_, /*test=*/true, /*write=*/true)
                               .build();

        vkDestroyShaderModule(device, environment_shader, nullptr);

        context().name(screen_pipelines_.opaque, VK_OBJECT_TYPE_PIPELINE, "scene opaque");
        context().name(face_pipelines_.opaque, VK_OBJECT_TYPE_PIPELINE,
                       "scene opaque (cube face)");
        context().name(fern_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene ferns");
        context().name(screen_pipelines_.sky, VK_OBJECT_TYPE_PIPELINE, "skybox");
        context().name(face_pipelines_.sky, VK_OBJECT_TYPE_PIPELINE,
                       "skybox (cube face)");
        context().name(mirror_pipeline_, VK_OBJECT_TYPE_PIPELINE, "mirrored sphere");
    }

    vkc::Camera camera_;

    Mesh cube_;
    Mesh quad_;
    Mesh sphere_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkFormat cube_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;
    vkc::Image face_depth_;

    vkc::Image sky_cube_;
    Cubemap live_cube_;

    // One arena per frame in flight, and the addresses this frame wrote into it.
    std::array<FrameArena, vkc::kFramesInFlight> arenas_;
    std::array<VkDeviceAddress, kPassCount> globals_{};
    VkDeviceAddress draws_address_ = 0;
    std::vector<DrawData> draws_;

    vkc::Image crate_map_;
    vkc::Image floor_map_;
    vkc::Image fern_map_;
    vkc::Sampler sampler_;
    vkc::Sampler clamp_sampler_;
    vkc::Sampler cube_sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fern_set_ = VK_NULL_HANDLE;
    VkDescriptorSet sky_set_ = VK_NULL_HANDLE;
    VkDescriptorSet live_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    ScenePipelines screen_pipelines_;
    ScenePipelines face_pipelines_;
    VkPipeline fern_pipeline_ = VK_NULL_HANDLE;
    VkPipeline mirror_pipeline_ = VK_NULL_HANDLE;

    bool sway_ = false;
    bool report_ = false;
    uint32_t pushes_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.5 - Buffer device address";
        BufferDeviceAddressApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
