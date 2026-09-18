// 4.4 Cubemaps
//
// A cubemap is one image with six layers, addressed by a direction instead of by a pair
// of texture coordinates. This chapter uses two of them, and they are built the same way
// and filled completely differently.
//
// The first is loaded from six PNG files and drawn as the sky. The second is never
// loaded at all: it is a render target with six faces, and every frame the scene is
// drawn into it six times -- once along each axis, from the centre of the sphere sitting
// in the middle of the room. The sphere then samples it, and reflects the room it is
// standing in.
//
// The second one is 4.3 done six times. What is new is the image itself:
//
//   * arrayLayers = 6 with VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, which is what allows a
//     view to call it a cube rather than a stack of six unrelated pictures
//   * seven views onto that single image -- one CUBE view covering all six layers, which
//     is what the shader samples, and six single-layer 2D views, which are what the six
//     rendering passes write into
//   * a 90 degree field of view, and -- uniquely in this series -- no Y-flip in the
//     projection. face_projection() below explains why that is not an oversight
//
//   1  reflect / refract / a Fresnel mix of the two
//   2  where the sphere's environment comes from
//   3  what the sphere is made of, when it is refracting

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

// The six faces arrive as PNG files, so this chapter decodes them itself. As in chapter
// 1.11, STB_IMAGE_STATIC keeps stb's symbols private to this translation unit: Assimp's
// static library exports the same names, and one global definition anywhere would break
// every Part 3 link. A private copy cannot clash with anything.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

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

// Moved further out than 4.3's, to leave the middle of the room to the sphere.
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

// The sun the sky was drawn with, pointing the way the light travels. tools/
// make_textures.py puts the sun at the direction this one comes *from*, so the sky and
// the shading agree -- which matters the moment a mirrored surface shows both at once.
constexpr glm::vec3 kSunDirection{-0.45F, -1.0F, -0.38F};
constexpr glm::vec3 kSunColour{0.92F, 0.88F, 0.80F};
constexpr glm::vec3 kAmbient{0.24F, 0.26F, 0.31F};
constexpr float kShininess = 48.0F;

constexpr float kMirrorRadius = 0.9F;
// Resting on the floor, so the lower half of the reflection has something in it.
constexpr glm::vec3 kMirrorCentre{0.0F, kFloorY + kMirrorRadius, 0.0F};

// The resolution of one face of the live cubemap. Six of these are rendered per frame,
// so it buys six times what it costs -- which is the honest reason live environment maps
// are usually smaller than they look like they ought to be.
constexpr uint32_t kFaceSize = 512;

// How the sphere turns a direction into a colour.
enum MirrorMode : int32_t {
    kModeReflect = 0,
    kModeRefract = 1,
    kModeFresnel = 2,
};
constexpr std::array<const char*, 3> kModeNames{"reflect", "refract", "Fresnel mix"};

// The ratio refract() wants: the index of refraction of the medium the light is leaving
// over the one it is entering. Air is 1.00.
constexpr std::array<float, 3> kRefractionRatios{
    1.0F / 1.33F,  // water
    1.0F / 1.52F,  // glass
    1.0F / 2.42F,  // diamond
};
constexpr std::array<const char*, 3> kRefractionNames{"water 1.33", "glass 1.52",
                                                      "diamond 2.42"};

// Where the sphere's environment comes from.
enum EnvironmentSource : size_t {
    // The six PNGs. The sphere mirrors the sky and nothing else -- the crates and the
    // ferns are simply not in the image being sampled.
    kSourceStaticSky = 0,
    // The live cubemap, with only the sky drawn into it. This exists to be checked
    // against the one above: the two should be indistinguishable, and anything that
    // makes them differ is a mistake in the face setup. See the article.
    kSourceLiveSky = 1,
    // The live cubemap with the whole scene drawn into it. This is the one worth having.
    kSourceLiveScene = 2,
};
constexpr std::array<const char*, 3> kSourceNames{
    "the six PNGs", "live, sky only (the check)", "live, the whole scene"};

// One camera per cube face, as a forward direction and an up vector, in the order the
// layers are stored: +X, -X, +Y, -Y, +Z, -Z.
//
// The up vectors look wrong and are not. A cube face stores its rows top to bottom, and
// "down the face" is the direction the face's own v axis points -- which for the four
// side faces is world -Y. The camera's up is therefore -Y as well, and the two faces
// that look along Y have to break the tie with Z instead. Get one of these wrong and
// that face alone comes out upside down or mirrored, with the other five perfect.
struct CubeFace {
    glm::vec3 forward;
    glm::vec3 up;
    const char* name;
};
constexpr std::array<CubeFace, 6> kCubeFaces{{
    {{1.0F, 0.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, "+X"},
    {{-1.0F, 0.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, "-X"},
    {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, "+Y"},
    {{0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, -1.0F}, "-Y"},
    {{0.0F, 0.0F, 1.0F}, {0.0F, -1.0F, 0.0F}, "+Z"},
    {{0.0F, 0.0F, -1.0F}, {0.0F, -1.0F, 0.0F}, "-Z"},
}};

// The six files, in the same order.
constexpr std::array<const char*, 6> kSkyFiles{
    "textures/lvk_sky_px.png", "textures/lvk_sky_nx.png", "textures/lvk_sky_py.png",
    "textures/lvk_sky_ny.png", "textures/lvk_sky_pz.png", "textures/lvk_sky_nz.png",
};

// Seven sets of per-frame constants per frame in flight: one per cube face, and one for
// the view the screen actually shows.
constexpr size_t kFacePassCount = 6;
constexpr size_t kPassCount = kFacePassCount + 1;
constexpr size_t kScreenPass = kFacePassCount;

struct Globals {
    glm::mat4 view;           // 0
    glm::mat4 projection;     // 64
    glm::mat4 view_rotation;  // 128
    glm::vec4 view_position;  // 192
    glm::vec4 sun_direction;  // 208
    glm::vec4 sun_colour;     // 224
    glm::vec4 ambient;        // 240, w is the specular exponent
};
static_assert(offsetof(Globals, view_rotation) == 128);
static_assert(sizeof(Globals) == 256);

struct PushConstants {
    glm::mat4 model;                         // 0
    std::array<glm::vec4, 3> normal_matrix;  // 64
    // A tint for the scene objects. The sphere reads the same bytes as
    // (mode, refraction ratio, unused, unused) -- one push range, two meanings.
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

// A vertex buffer, an index buffer, and how many indices are in it.
struct Mesh {
    vkc::Buffer vertices;
    vkc::Buffer indices;
    uint32_t index_count = 0;

    void destroy() noexcept {
        indices.destroy();
        vertices.destroy();
    }
};

[[nodiscard]] Mesh upload_mesh(vkc::Context& context, std::span<const Vertex> vertices,
                               std::span<const uint16_t> indices) {
    Mesh mesh;
    mesh.vertices = vkc::upload_to_device_local(context, vertices.data(),
                                                vertices.size_bytes(),
                                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.indices = vkc::upload_to_device_local(context, indices.data(),
                                               indices.size_bytes(),
                                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.index_count = static_cast<uint32_t>(indices.size());
    return mesh;
}

// A UV sphere: `rings` bands from pole to pole, `segments` around. The normal of a point
// on a unit sphere is the point itself, which is the only reason this is the shape the
// chapter uses -- every fragment gets an exactly correct normal with no interpolation
// error, so what the reflection shows is the environment map and never the tessellation.
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
            // Counter-clockwise seen from outside, which is what the cull mode expects.
            indices.insert(indices.end(), {a, static_cast<uint16_t>(a + 1), b,
                                           static_cast<uint16_t>(a + 1),
                                           static_cast<uint16_t>(b + 1), b});
        }
    }
    return {std::move(vertices), std::move(indices)};
}

// ---------------------------------------------------------------------------
// The cubemap itself
// ---------------------------------------------------------------------------

// One image, six layers, and up to seven views onto it.
//
// vkc::Image has made a single-layer 2D image since 1.11 and cannot express this, so it
// is written out here in full -- which is just as well, because the three lines that
// make it a cube are the chapter.
struct Cubemap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    // All six layers, VK_IMAGE_VIEW_TYPE_CUBE. This is what a descriptor points at and
    // what the shader's SamplerCube samples.
    VkImageView cube_view = VK_NULL_HANDLE;
    // One layer each, VK_IMAGE_VIEW_TYPE_2D. These are what rendering passes attach.
    // Empty for a cubemap that is only ever read.
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
                                     uint32_t size, VkImageUsageFlags usage,
                                     bool with_face_views) {
    Cubemap cube;

    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        // Without this flag the image is six ordinary array layers and no view may call
        // it a cube. It is the whole difference, and it must be asked for at creation:
        // a view cannot add it later.
        .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        // Still 2D. A cubemap is not a 3D image -- it is six 2D images that a sampler
        // has been told how to choose between.
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {size, size, 1},
        .mipLevels = 1,
        // Six, and exactly six. CUBE_COMPATIBLE requires it, and the order is fixed by
        // the specification: +X, -X, +Y, -Y, +Z, -Z.
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

    // The sampling view: type CUBE, all six layers at once.
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

    if (!with_face_views) {
        return cube;
    }

    // The rendering views: one layer each, and type 2D rather than CUBE. A rendering
    // pass attaches a single 2D image; it has no concept of a cube and does not need
    // one. Six views onto six layers of one image is all "render to a cubemap" is.
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

// Decodes the six files and uploads them into the six layers.
void upload_cubemap(vkc::Context& context, const Cubemap& cube, uint32_t size,
                    std::span<const char* const> files) {
    const VkDeviceSize face_bytes =
        static_cast<VkDeviceSize>(size) * static_cast<VkDeviceSize>(size) * 4;

    // One staging buffer holding all six faces back to back, so the whole cubemap is one
    // allocation and one submit rather than six of each.
    vkc::Buffer staging(context.allocator(), face_bytes * 6,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    for (size_t face = 0; face < files.size(); ++face) {
        const std::filesystem::path path = vkc::asset_path(files[face]);

        int width = 0;
        int height = 0;
        int channels = 0;
        const std::unique_ptr<stbi_uc, void (*)(void*)> pixels(
            stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha),
            stbi_image_free);
        if (pixels == nullptr) {
            throw std::runtime_error(std::format(
                "Could not load '{}': {}. Did you run tools/make_textures.py?",
                path.string(), stbi_failure_reason()));
        }
        // Every face of a cubemap is square and they are all the same size. The
        // validation layers would not catch a mismatch here -- the copy would simply
        // read the wrong number of bytes -- so it is worth checking.
        if (width != static_cast<int>(size) || height != static_cast<int>(size)) {
            throw std::runtime_error(
                std::format("'{}' is {}x{}, but every cubemap face must be {}x{}.",
                            path.string(), width, height, size, size));
        }

        std::memcpy(static_cast<std::byte*>(staging.mapped()) +
                        static_cast<size_t>(face_bytes) * face,
                    pixels.get(), static_cast<size_t>(face_bytes));
    }

    vkc::immediate_submit(context, [&](VkCommandBuffer cmd) {
        // 4.3's barrier, and it covers all six layers because its subresource range says
        // VK_REMAINING_ARRAY_LAYERS. A range that named layer 0 alone would leave the
        // other five in UNDEFINED, and the copy into them would be undefined behaviour
        // that happens to work on most drivers.
        vkc::image_barrier(cmd, cube.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        // One region per face. baseArrayLayer is the only field that differs, and it is
        // how the six blocks of the staging buffer find their six layers.
        std::array<VkBufferImageCopy, 6> regions{};
        for (uint32_t face = 0; face < 6; ++face) {
            regions[face] = VkBufferImageCopy{
                .bufferOffset = face_bytes * face,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {size, size, 1},
            };
        }
        vkCmdCopyBufferToImage(cmd, staging.handle(), cube.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(regions.size()), regions.data());

        vkc::image_barrier(cmd, cube.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });
}

// The two pipelines whose front-face setting depends on what is being drawn into.
// Everything else about them is identical; see create_pipelines().
struct ScenePipelines {
    VkPipeline opaque = VK_NULL_HANDLE;
    VkPipeline sky = VK_NULL_HANDLE;
};

class CubemapsApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        cube_ = upload_mesh(context(), kCubeVertices, kCubeIndices);
        quad_ = upload_mesh(context(), kQuadVertices, kQuadIndices);

        const auto [sphere_vertices, sphere_indices] = make_sphere(48, 96, kMirrorRadius);
        sphere_ = upload_mesh(context(), sphere_vertices, sphere_indices);

        depth_format_ = vkc::choose_depth_format(context().physical_device());
        // The live cubemap takes the swapchain's format so that one set of pipelines can
        // render into either target. Pipelines are told their attachment formats, and
        // formats that differed would mean two of every pipeline.
        cube_format_ = swapchain().format();

        load_textures();
        create_cubemaps();
        create_uniform_buffers();
        create_descriptors();
        create_depth_buffer(swapchain().extent());
        create_pipelines();

        camera_.position = {0.0F, 1.80F, 5.60F};
        camera_.pitch = -6.0F;

        spdlog::info("1 material, 2 environment source, 3 refractive index.");
    }

    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                mode_ = (mode_ + 1) % static_cast<int32_t>(kModeNames.size());
                spdlog::info("Sphere: {}", kModeNames[static_cast<size_t>(mode_)]);
                break;
            case SDL_SCANCODE_2:
                source_ = (source_ + 1) % kSourceNames.size();
                spdlog::info("Environment: {}", kSourceNames[source_]);
                break;
            case SDL_SCANCODE_3:
                refraction_ = (refraction_ + 1) % kRefractionRatios.size();
                spdlog::info("Refracting like {}", kRefractionNames[refraction_]);
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
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        live_cube_.destroy(device, allocator);
        sky_cube_.destroy(device, allocator);

        cube_sampler_.destroy();
        clamp_sampler_.destroy();
        sampler_.destroy();
        fern_map_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();

        for (auto& slot : uniform_buffers_) {
            for (vkc::Buffer& buffer : slot) {
                buffer.destroy();
            }
        }
        face_depth_.destroy();
        depth_.destroy();
        sphere_.destroy();
        quad_.destroy();
        cube_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        if (source_ != kSourceStaticSky) {
            render_cube_faces(frame, slot);
        }
        render_screen(frame, slot);
    }

private:
    // ---------------------------------------------------------------------------
    // Six passes into six faces of one image
    // ---------------------------------------------------------------------------

    void render_cube_faces(const vkc::FrameInfo& frame, size_t slot) {
        // Every layer at once, from UNDEFINED: the loadOp clears each face, so there is
        // nothing in there worth preserving. The source scope is the previous frame's
        // *reading* of this cubemap by the sphere, two frames back at worst -- writing
        // after a read is a hazard exactly as it was in 4.3.
        vkc::image_barrier(frame.cmd, live_cube_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        const VkExtent2D face_extent{kFaceSize, kFaceSize};

        for (size_t face = 0; face < kFacePassCount; ++face) {
            // The six faces are six *different* layers, so there is no hazard between
            // them on the colour attachment. They share one depth buffer, though, and
            // that one does need saying: this face's depth clear must not begin before
            // the previous face's depth writes have finished. Nothing in dynamic
            // rendering implies it -- two rendering passes in a row are not ordered
            // against each other just for being in the same command buffer.
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

            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout_, 0, 1, &frame_sets_[slot][face], 0,
                                    nullptr);

            if (source_ == kSourceLiveScene) {
                draw_scene(frame.cmd, face_pipelines_);
            }
            // The sky goes into the live cubemap too, and it is always the *static* sky
            // that goes in. Feeding the live cubemap back into itself would be a hall of
            // mirrors with a frame of lag in it.
            draw_sky(frame.cmd, face_pipelines_);

            vkCmdEndRendering(frame.cmd);
        }

        // And now it is a texture. One barrier covers all six layers.
        vkc::image_barrier(frame.cmd, live_cube_.image,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    // ---------------------------------------------------------------------------
    // The pass the window shows
    // ---------------------------------------------------------------------------

    void render_screen(const vkc::FrameInfo& frame, size_t slot) {
        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.05F, 0.06F, 0.09F, 1.0F}});
        set_viewport(frame.cmd, frame.extent);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot][kScreenPass],
                                0, nullptr);

        draw_scene(frame.cmd, screen_pipelines_);
        draw_mirror(frame.cmd);
        draw_sky(frame.cmd, screen_pipelines_);

        vkCmdEndRendering(frame.cmd);
    }

    // The floor, the crates and the ferns: everything that is drawn identically whether
    // it is going to the screen or into a cube face.
    void draw_scene(VkCommandBuffer cmd, const ScenePipelines& pipelines) {
        bind_mesh(cmd, quad_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.opaque);
        bind_texture(cmd, floor_set_);
        push(cmd, glm::mat4(1.0F), kWhite);
        vkCmdDrawIndexed(cmd, kQuadIndexCount, 1, kFloorFirstIndex, 0, 0);

        bind_mesh(cmd, cube_);
        bind_texture(cmd, crate_set_);
        for (const Placement& crate : kCrates) {
            push(cmd, glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F)),
                 kWhite);
            vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
        }

        bind_mesh(cmd, quad_);
        bind_texture(cmd, fern_set_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fern_pipeline_);
        for (const Placement& fern : kFerns) {
            push(cmd, model_of(fern), kWhite);
            vkCmdDrawIndexed(cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }
    }

    void draw_mirror(VkCommandBuffer cmd) {
        bind_mesh(cmd, sphere_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mirror_pipeline_);
        bind_texture(cmd, source_ == kSourceStaticSky ? sky_set_ : live_set_);
        push(cmd, glm::translate(glm::mat4(1.0F), kMirrorCentre),
             glm::vec4(static_cast<float>(mode_), kRefractionRatios[refraction_], 0.0F,
                       0.0F));
        vkCmdDrawIndexed(cmd, sphere_.index_count, 1, 0, 0, 0);
    }

    // Drawn last, always. Its fragments are at depth 1.0 and every pixel the scene
    // already covered rejects them before the fragment shader runs, so the sky costs
    // only the pixels on which it is actually visible.
    void draw_sky(VkCommandBuffer cmd, const ScenePipelines& pipelines) {
        bind_mesh(cmd, cube_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.sky);
        bind_texture(cmd, sky_set_);
        push(cmd, glm::mat4(1.0F), kWhite);
        vkCmdDrawIndexed(cmd, cube_.index_count, 1, 0, 0, 0);
    }

    // ---------------------------------------------------------------------------
    // Frame plumbing
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

    static void bind_mesh(VkCommandBuffer cmd, const Mesh& mesh) {
        const VkDeviceSize offset = 0;
        const VkBuffer handle = mesh.vertices.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &handle, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indices.handle(), 0, VK_INDEX_TYPE_UINT16);
    }

    void bind_texture(VkCommandBuffer cmd, VkDescriptorSet set) const {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 1,
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

    // The projection for one cube face, and the one place in this series where the
    // Y-flip above is deliberately absent.
    //
    // That flip exists because Vulkan's clip space has +y pointing *down* the screen
    // while everyone writing a scene wants +y to be up. A window is looked at, so it
    // gets the flip. A cube face is not looked at -- it is sampled by direction, and the
    // convention it is sampled with stores each face with its v axis running downward,
    // which is precisely what unflipped Vulkan clip space already does. Adding the flip
    // here would store all six faces upside down, and the sphere would then reflect an
    // upside-down world convincingly enough to be missed.
    //
    // 90 degrees, square, is forced: six frusta have to tile the sphere of directions
    // exactly, with no gap and no overlap.
    [[nodiscard]] static glm::mat4 face_projection() {
        return glm::perspective(glm::radians(90.0F), 1.0F, 0.1F, 100.0F);
    }

    void create_uniform_buffers() {
        for (auto& slot : uniform_buffers_) {
            for (vkc::Buffer& buffer : slot) {
                buffer =
                    vkc::Buffer(context().allocator(), sizeof(Globals),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
        }
    }

    // Seven sets of constants: one per cube face and one for the screen. They differ only
    // in the two matrices, which is the entire difference between the seven passes.
    void update_globals(size_t slot, VkExtent2D screen_extent) {
        const auto write = [&](size_t pass, const glm::mat4& view,
                               const glm::mat4& projection, const glm::vec3& eye) {
            const Globals globals{
                .view = view,
                .projection = projection,
                // The view matrix with its translation removed, for the skybox. Taking
                // the upper-left 3x3 and putting it back into a 4x4 is exactly that.
                .view_rotation = glm::mat4(glm::mat3(view)),
                .view_position = glm::vec4(eye, 1.0F),
                .sun_direction = glm::vec4(glm::normalize(kSunDirection), 0.0F),
                .sun_colour = glm::vec4(kSunColour, 1.0F),
                .ambient = glm::vec4(kAmbient, kShininess),
            };
            std::memcpy(uniform_buffers_[slot][pass].mapped(), &globals, sizeof(globals));
        };

        for (size_t face = 0; face < kFacePassCount; ++face) {
            // The camera for a face sits at the centre of the sphere and looks along one
            // axis. This is what makes the reflection correct only for that one point --
            // the sphere reflects the world as seen from its middle, which is why a
            // cubemap rendered for one object cannot be reused for another standing
            // somewhere else.
            const glm::mat4 view =
                glm::lookAt(kMirrorCentre, kMirrorCentre + kCubeFaces[face].forward,
                            kCubeFaces[face].up);
            write(face, view, face_projection(), kMirrorCentre);
        }

        write(kScreenPass, camera_.view(), screen_projection(screen_extent),
              camera_.position);
    }

    void create_depth_buffer(VkExtent2D extent) {
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
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

        // The sampler both cubemaps are read with. CLAMP_TO_EDGE is the only sensible
        // address mode for a cube: there is no "past the edge" of a face to repeat into,
        // because the neighbour is another face. Vulkan filters across the seams between
        // faces for us -- seamless cube filtering is core behaviour, not an extension
        // and not a flag -- so the twelve edges never show.
        cube_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .anisotropy = false});
    }

    void create_cubemaps() {
        // Read only: the six PNGs go in and nothing ever writes it again.
        sky_cube_ = create_cubemap(context(), VK_FORMAT_R8G8B8A8_SRGB, kFaceSize,
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT,
                                   /*with_face_views=*/false);
        upload_cubemap(context(), sky_cube_, kFaceSize, kSkyFiles);

        // Written every frame and read every frame: 4.3's two usage bits, on a
        // six-layer image.
        live_cube_ = create_cubemap(context(), cube_format_, kFaceSize,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT,
                                    /*with_face_views=*/true);

        // One depth buffer, reused by all six faces. They are rendered one after another,
        // never at once, so six would be six times the memory for nothing -- at the cost
        // of the barrier between faces in render_cube_faces().
        face_depth_ = vkc::create_depth_buffer(context(), depth_format_,
                                               {kFaceSize, kFaceSize});

        context().name(sky_cube_.image, VK_OBJECT_TYPE_IMAGE, "sky cubemap");
        context().name(live_cube_.image, VK_OBJECT_TYPE_IMAGE, "live cubemap");
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

        // One layout for every texture in the sample, cubemaps included. This is worth
        // stopping on: a descriptor set layout says "one combined image sampler, visible
        // to the fragment shader" and says nothing whatever about 2D or cube. The
        // dimensionality lives in the *view* the descriptor is written with, and in the
        // type the shader declares. The layout cannot tell them apart and does not need
        // to.
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

        constexpr uint32_t kGlobalsSets = vkc::kFramesInFlight * kPassCount;
        constexpr uint32_t kTextureSets = 5;  // crate, floor, fern, sky, live
        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kGlobalsSets},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kTextureSets},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = kGlobalsSets + kTextureSets,
            .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        for (size_t slot = 0; slot < vkc::kFramesInFlight; ++slot) {
            for (size_t pass = 0; pass < kPassCount; ++pass) {
                const VkDescriptorSetAllocateInfo alloc{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .descriptorPool = descriptor_pool_,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &frame_set_layout_,
                };
                VK_CHECK(
                    vkAllocateDescriptorSets(device, &alloc, &frame_sets_[slot][pass]));

                const VkDescriptorBufferInfo buffer_info{
                    .buffer = uniform_buffers_[slot][pass].handle(),
                    .offset = 0,
                    .range = sizeof(Globals),
                };
                const VkWriteDescriptorSet write{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = frame_sets_[slot][pass],
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
        }

        crate_set_ = allocate_texture_set(crate_map_.view(), sampler_.handle());
        floor_set_ = allocate_texture_set(floor_map_.view(), sampler_.handle());
        fern_set_ = allocate_texture_set(fern_map_.view(), clamp_sampler_.handle());

        // The two cubemaps, written into exactly the same kind of set as the three
        // textures above. The only difference is that these views are CUBE views.
        sky_set_ = allocate_texture_set(sky_cube_.cube_view, cube_sampler_.handle());
        live_set_ = allocate_texture_set(live_cube_.cube_view, cube_sampler_.handle());

        context().name(sky_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "sky cubemap");
        context().name(live_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "live cubemap");
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

        // Each pipeline is given only the attributes its vertex shader actually reads.
        // All four share one vertex buffer and one binding stride -- the mesh is the
        // same -- but the skybox wants nothing except the position, because a direction
        // is the only input it has, and the sphere wants the position and the normal.
        // Handing all three to all four is not harmless: the validation layers report
        // every declared attribute a shader does not consume, and they are right to.
        const std::span<const VkVertexInputAttributeDescription> scene_attributes{
            attributes};
        const std::span<const VkVertexInputAttributeDescription> mirror_attributes{
            attributes.data(), 2};
        const std::span<const VkVertexInputAttributeDescription> sky_attributes{
            attributes.data(), 1};

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };
        const std::array<VkDescriptorSetLayout, 2> set_layouts{frame_set_layout_,
                                                               texture_set_layout_};
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

        // Every pipeline here is told the same colour format, which is what lets all four
        // of them render into the swapchain and into a cube face without being rebuilt.
        const auto base =
            [&](VkShaderModule module,
                std::span<const VkVertexInputAttributeDescription> attrs)
            -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(module)
                .vertex_input(std::span(&binding, 1), attrs)
                .colour_attachment(cube_format_)
                .layout(pipeline_layout_);
        };

        const VkShaderModule scene_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "cubemaps.slang");

        // Every one of these is built twice: once to be drawn on the screen, and once to
        // be drawn into a cube face. The two differ in one field, the front face.
        //
        // A cube map is specified in a left-handed coordinate system. A face is
        // therefore stored *mirrored* compared with what an ordinary camera would
        // record -- which is precisely why face_projection() leaves out the Y-flip that
        // every other projection in this series applies. A mirror reverses the apparent
        // winding of every triangle it shows, so geometry that is counter-clockwise on
        // screen is clockwise in a face.
        //
        // Nothing about the geometry, the meshes or the shaders changes. The only thing
        // that has to change is the sentence the pipeline uses to say which side is the
        // front, and the symptom of forgetting is very specific: single-sided things --
        // the floor, the inside of the skybox -- vanish from the reflection, while
        // anything drawn with culling off stays put.
        for (size_t pass = 0; pass < 2; ++pass) {
            const VkFrontFace winding =
                pass == 0 ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            const VkPipeline opaque = base(scene_shader, scene_attributes)
                                          .cull(VK_CULL_MODE_BACK_BIT, winding)
                                          .depth(depth_format_, /*test=*/true,
                                                 /*write=*/true)
                                          .build();
            (pass == 0 ? screen_pipelines_ : face_pipelines_).opaque = opaque;
        }

        // Culling is off for the cut-out ferns, so winding cannot affect them and one
        // pipeline serves both passes.
        fern_pipeline_ = base(scene_shader, scene_attributes)
                             .shaders(scene_shader, "vertexMain", "cutoutFragmentMain")
                             .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                             .depth(depth_format_, /*test=*/true, /*write=*/true)
                             .build();

        vkDestroyShaderModule(device, scene_shader, nullptr);

        const VkShaderModule environment_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "environment.slang");

        // The sky. Four deliberate differences from every other pipeline in the series:
        //
        //   LESS_OR_EQUAL, because the vertex shader forces depth to exactly 1.0 and a
        //   plain LESS would lose to a buffer cleared to 1.0 at every single pixel;
        //
        //   no depth writes, because the sky is behind everything and must never stop
        //   anything else from drawing;
        //
        //   cull FRONT, because the camera is inside the box. The cube's triangles wind
        //   counter-clockwise seen from outside, so from inside they are back faces, and
        //   the usual BACK setting would cull away the entire sky;
        //
        //   and the same two-windings treatment as the scene above.
        for (size_t pass = 0; pass < 2; ++pass) {
            const VkFrontFace winding =
                pass == 0 ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
            const VkPipeline sky = base(environment_shader, sky_attributes)
                                       .cull(VK_CULL_MODE_FRONT_BIT, winding)
                                       .depth(depth_format_, /*test=*/true,
                                              /*write=*/false,
                                              VK_COMPARE_OP_LESS_OR_EQUAL)
                                       .build();
            (pass == 0 ? screen_pipelines_ : face_pipelines_).sky = sky;
        }

        // The sphere is only ever drawn on the screen -- rendering it into its own
        // environment map would fill all six faces with the inside of itself -- so it
        // needs one winding and one pipeline.
        mirror_pipeline_ = base(environment_shader, mirror_attributes)
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

    Cubemap sky_cube_;
    Cubemap live_cube_;

    std::array<std::array<vkc::Buffer, kPassCount>, vkc::kFramesInFlight>
        uniform_buffers_;
    std::array<std::array<VkDescriptorSet, kPassCount>, vkc::kFramesInFlight>
        frame_sets_{};

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

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    // The same scene, wound for the screen and wound for a cube face.
    ScenePipelines screen_pipelines_;
    ScenePipelines face_pipelines_;
    VkPipeline fern_pipeline_ = VK_NULL_HANDLE;
    VkPipeline mirror_pipeline_ = VK_NULL_HANDLE;

    int32_t mode_ = kModeReflect;
    size_t source_ = kSourceLiveScene;
    size_t refraction_ = 1;  // glass
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.4 - Cubemaps";
        CubemapsApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
