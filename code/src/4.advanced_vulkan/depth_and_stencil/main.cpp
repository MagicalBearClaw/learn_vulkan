// 4.1 Pipeline state: depth & stencil
//
// Part 4 goes back to the pipeline. Every chapter so far changed the picture by changing
// what the shaders computed; this one changes it by changing what the fixed-function
// hardware does with the fragments *after* the shader has finished with them.
//
// Three things are new here, and all three are pipeline state rather than code:
//
//   * the depth test's comparison and its write mask, which have been on and unexamined
//     since 1.13
//   * depth bias, which is the answer to two coplanar surfaces fighting over which one
//     is in front
//   * the stencil test, which needs a depth buffer format that carries a stencil aspect,
//     a second attachment in the rendering info, and two VkStencilOpState structs
//
// The scene is two crates on a grid floor, with a painted slab lying exactly in the
// floor's plane and an outline around each crate. Press 1 to turn the outlines off and
// 2 to turn the depth bias off; both are there so the failure modes can be looked at
// rather than described.

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

// The floor sits at y = -1, which is exactly where the crates' bottom faces land.
constexpr float kFloorY = -1.0F;
constexpr float kFloorHalf = 14.0F;
// The sampler repeats, so a uv that runs past 1 tiles the grid rather than stretching it.
constexpr float kFloorTiles = 8.0F;

// Two quads in one buffer, both already in world space: the floor, and a painted slab
// lying in *exactly* the same plane. Identical y, by construction -- which is the setup
// for the depth-bias half of the chapter. Nothing here decides which one wins; that is
// the depth test's job, and without help it cannot do it.
constexpr std::array<Vertex, 8> kQuadVertices{{
    // floor, indices 0..5
    {{-kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {0, 0}},
    {{-kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {0, kFloorTiles}},
    {{kFloorHalf, kFloorY, kFloorHalf}, {0, 1, 0}, {kFloorTiles, kFloorTiles}},
    {{kFloorHalf, kFloorY, -kFloorHalf}, {0, 1, 0}, {kFloorTiles, 0}},
    // decal, indices 6..11
    {{-3.0F, kFloorY, -2.0F}, {0, 1, 0}, {0, 0}},
    {{-3.0F, kFloorY, 2.6F}, {0, 1, 0}, {0, 1}},
    {{3.0F, kFloorY, 2.6F}, {0, 1, 0}, {1, 1}},
    {{3.0F, kFloorY, -2.0F}, {0, 1, 0}, {1, 0}},
}};

constexpr std::array<uint16_t, 12> kQuadIndices{
    0, 1, 2, 2, 3, 0,  // floor
    4, 5, 6, 6, 7, 4,  // decal
};

constexpr uint32_t kFloorFirstIndex = 0;
constexpr uint32_t kDecalFirstIndex = 6;
constexpr uint32_t kQuadIndexCount = 6;

struct CratePlacement {
    glm::vec3 position;
    float angle;  // radians about y
};

constexpr std::array<CratePlacement, 2> kCrates{{
    {{-1.35F, -0.5F, 0.5F}, 0.36F},
    {{1.30F, -0.5F, -1.20F}, -0.72F},
}};

// How much bigger the outline pass draws the crate. Small enough to read as a border
// rather than as a second crate.
constexpr float kOutlineScale = 1.06F;
constexpr glm::vec4 kOutlineColour{1.0F, 0.62F, 0.16F, 1.0F};
constexpr glm::vec4 kDecalColour{0.10F, 0.31F, 0.37F, 1.0F};
constexpr glm::vec4 kWhite{1.0F};

// Depth bias, in units of "the smallest difference the depth buffer can resolve" -- a
// number the implementation defines, which is exactly why the bias is expressed as a
// multiple of it rather than as a distance. Negative pulls toward the viewer, and with
// VK_COMPARE_OP_LESS that is the direction that wins.
//
// The slope factor is scaled by how steeply the polygon recedes from the viewer. A
// floor viewed at a glancing angle covers a lot of depth in very few pixels, and a
// constant offset that is ample head-on is nothing out at the horizon.
constexpr float kDecalBiasConstant = -2.0F;
constexpr float kDecalBiasSlope = -2.0F;

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

// 2.6's push block, unchanged and still exactly the 128 bytes Vulkan guarantees.
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

// ---------------------------------------------------------------------------------
// The depth-stencil buffer, written out here rather than taken from vkcommon
// ---------------------------------------------------------------------------------
//
// vkcommon has choose_depth_format() and create_depth_buffer(), which 1.13 wrote and
// every chapter since has called. Neither is usable in this chapter, for one reason:
// they produce a depth buffer with no stencil in it.
//
// 1.13's chooser asks for VK_FORMAT_D32_SFLOAT first, which is the most precise depth
// format and has no stencil aspect at all. That was the right default for eleven
// chapters that never mentioned stencil. It is the wrong one here, so this chapter
// writes its own chooser that only considers the combined formats.

[[nodiscard]] VkFormat choose_depth_stencil_format(VkPhysicalDevice physical_device) {
    // The two combined formats, most precise first. Vulkan guarantees that *at least
    // one* of these is supported as a depth-stencil attachment on every implementation,
    // but not which one -- D24_UNORM_S8_UINT is absent on some desktop GPUs and
    // D32_SFLOAT_S8_UINT on some mobile ones -- so both have to be asked for.
    constexpr std::array candidates{
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };

    for (const VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
        if ((properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    throw std::runtime_error("No supported depth-stencil attachment format was found.");
}

[[nodiscard]] vkc::Image create_depth_stencil_buffer(vkc::Context& context,
                                                     VkFormat format,
                                                     VkExtent2D extent) {
    context.wait_idle();

    vkc::Image image(context,
                     vkc::ImageDesc{
                         .format = format,
                         .extent = extent,
                         .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                         .mip_levels = 1,
                         // The difference from 1.13's depth buffer, and the only one.
                         // The aspect mask has to name both, because it is used for the
                         // view and for every barrier, and a view that covers only the
                         // depth half cannot be used as a stencil attachment.
                         .aspect = VK_IMAGE_ASPECT_DEPTH_BIT |
                                   VK_IMAGE_ASPECT_STENCIL_BIT,
                     });

    vkc::immediate_submit(context, [&](VkCommandBuffer cmd) {
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL, not DEPTH_ATTACHMENT_OPTIMAL: the layout
        // has to cover both aspects for the same reason the view does.
        image.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    });

    context.name(image.handle(), VK_OBJECT_TYPE_IMAGE, "depth-stencil buffer");
    return image;
}

// vkcommon's begin_rendering() attaches colour and depth. This one adds the third
// attachment, and it is worth seeing how little there is to it.
//
// pDepthAttachment and pStencilAttachment point at two different structs that name the
// same VkImageView. That is not a mistake and not a copy: a combined format is one
// image holding two aspects, and dynamic rendering describes them separately because
// they are cleared, loaded and stored separately. Here the stencil clears to 0 and is
// discarded at the end of the frame, exactly like the depth.
void begin_rendering(VkCommandBuffer cmd, VkImageView colour_view,
                     VkImageView depth_stencil_view, VkExtent2D extent,
                     const VkClearColorValue& clear_colour) {
    const VkRenderingAttachmentInfo colour_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = colour_view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = VkClearValue{.color = clear_colour},
    };

    const VkClearValue clear_depth_stencil{.depthStencil = {.depth = 1.0F, .stencil = 0}};

    const VkRenderingAttachmentInfo depth_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = depth_stencil_view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = clear_depth_stencil,
    };

    const VkRenderingAttachmentInfo stencil_attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = depth_stencil_view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        // Clearing to 0 is what makes "did anything write here?" a meaningful question
        // later in the frame. Without it the outline pass would test against whatever
        // the previous frame left behind.
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue = clear_depth_stencil,
    };

    const VkRenderingInfo rendering{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colour_attachment,
        .pDepthAttachment = &depth_attachment,
        .pStencilAttachment = &stencil_attachment,
    };
    vkCmdBeginRendering(cmd, &rendering);
}

class DepthAndStencilApp : public vkc::App {
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

        depth_stencil_format_ =
            choose_depth_stencil_format(context().physical_device());
        depth_stencil_ = create_depth_stencil_buffer(context(), depth_stencil_format_,
                                                     swapchain().extent());
        spdlog::info("Depth-stencil format: {}",
                     depth_stencil_format_ == VK_FORMAT_D32_SFLOAT_S8_UINT
                         ? "D32_SFLOAT_S8_UINT"
                         : "D24_UNORM_S8_UINT");

        load_textures();
        create_uniform_buffers();
        create_descriptors();
        create_pipelines();

        camera_.position = {0.0F, 4.20F, 8.50F};
        camera_.pitch = -26.0F;

        spdlog::info("1 toggles the outlines, 2 toggles the decal's depth bias.");
    }

    void on_resize(VkExtent2D extent) override {
        depth_stencil_ =
            create_depth_stencil_buffer(context(), depth_stencil_format_, extent);
    }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
            if (event.key.scancode == SDL_SCANCODE_1) {
                outlines_ = !outlines_;
                spdlog::info("Outlines {}", outlines_ ? "on" : "off");
            } else if (event.key.scancode == SDL_SCANCODE_2) {
                depth_bias_ = !depth_bias_;
                spdlog::info("Decal depth bias {}", depth_bias_ ? "on" : "off");
            }
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, outline_pipeline_, nullptr);
        vkDestroyPipeline(device, decal_unbiased_pipeline_, nullptr);
        vkDestroyPipeline(device, decal_pipeline_, nullptr);
        vkDestroyPipeline(device, crate_pipeline_, nullptr);
        vkDestroyPipeline(device, floor_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, material_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        sampler_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();
        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        depth_stencil_.destroy();
        quad_indices_.destroy();
        quad_vertices_.destroy();
        cube_indices_.destroy();
        cube_vertices_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        begin_rendering(frame.cmd, frame.view, depth_stencil_.view(), frame.extent,
                        {{0.05F, 0.06F, 0.09F, 1.0F}});

        set_viewport(frame);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        // --- the two coplanar quads ---------------------------------------------
        bind_geometry(frame.cmd, quad_vertices_, quad_indices_);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, floor_pipeline_);
        bind_material(frame.cmd, floor_set_);
        push(frame.cmd, glm::mat4(1.0F), kWhite);
        vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kFloorFirstIndex, 0, 0);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          depth_bias_ ? decal_pipeline_ : decal_unbiased_pipeline_);
        push(frame.cmd, glm::mat4(1.0F), kDecalColour);
        vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kDecalFirstIndex, 0, 0);

        // --- the crates, which also stamp the stencil buffer ---------------------
        bind_geometry(frame.cmd, cube_vertices_, cube_indices_);
        bind_material(frame.cmd, crate_set_);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, crate_pipeline_);
        for (const CratePlacement& crate : kCrates) {
            push(frame.cmd, model_of(crate, 1.0F), kWhite);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kCubeIndices.size()), 1, 0,
                             0, 0);
        }

        // --- the outlines --------------------------------------------------------
        //
        // Every crate has stamped its 1 into the stencil buffer before any outline is
        // drawn. That ordering is the whole trick: if the passes were interleaved, the
        // first crate's halo would spill over the second crate, which has not yet
        // written the stencil that would have masked it.
        if (outlines_) {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              outline_pipeline_);
            for (const CratePlacement& crate : kCrates) {
                push(frame.cmd, model_of(crate, kOutlineScale), kOutlineColour);
                vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kCubeIndices.size()), 1,
                                 0, 0, 0);
            }
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------

    [[nodiscard]] static glm::mat4 model_of(const CratePlacement& crate, float scale) {
        return glm::scale(
            glm::rotate(glm::translate(glm::mat4(1.0F), crate.position), crate.angle,
                        glm::vec3(0.0F, 1.0F, 0.0F)),
            glm::vec3(scale));
    }

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
        floor_map_ = vkc::load_texture(context(),
                                       vkc::asset_path("textures/lvk_grid.png"),
                                       VK_FORMAT_R8G8B8A8_SRGB);
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

        context().name(crate_map_.handle(), VK_OBJECT_TYPE_IMAGE, "crate diffuse map");
        context().name(floor_map_.handle(), VK_OBJECT_TYPE_IMAGE, "floor grid map");
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
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + 2,
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

        crate_set_ = allocate_material_set(crate_map_);
        floor_set_ = allocate_material_set(floor_map_);

        context().name(crate_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "crate material");
        context().name(floor_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "floor material");
    }

    [[nodiscard]] VkDescriptorSet allocate_material_set(const vkc::Image& texture) {
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
            .sampler = sampler_.handle(),
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
            vkc::load_shader(device, LVK_CHAPTER_ID, "depth_and_stencil.slang");

        // Everything below shares this opening. The one line worth pausing on is
        // stencil_attachment(): *every* pipeline in this chapter declares the stencil
        // format, including the three that never test the stencil.
        //
        // Declaring the attachment and enabling the test are separate decisions.
        // A pipeline's VkPipelineRenderingCreateInfo has to describe the same set of
        // attachments the render pass instance will have, or the two do not match and
        // the draw is invalid -- and the floor, the decal and the crates are all drawn
        // into a rendering instance that has a stencil attachment, whether they care
        // about it or not.
        // The return type is spelled out because the setters return a reference to the
        // builder, and `auto` would deduce a reference to a temporary that has already
        // died. Naming the type makes each call hand back a fresh copy to configure.
        const auto base = [&]() -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(shader)
                .vertex_input(std::span(&binding, 1), attributes)
                .colour_attachment(swapchain().format())
                .stencil_attachment(depth_stencil_format_)
                .layout(pipeline_layout_);
        };

        // The floor. Ordinary opaque geometry: test against what is already there, write
        // what survives, and leave the stencil alone.
        floor_pipeline_ = base()
                              .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                              .depth(depth_stencil_format_, /*test=*/true, /*write=*/true,
                                     VK_COMPARE_OP_LESS)
                              .build();

        // The crates, with the only difference that matters: they stamp the stencil.
        //
        // compareOp ALWAYS means the stencil test never rejects anything -- this pass is
        // writing, not reading. passOp REPLACE writes `reference` (1) wherever a
        // fragment also survives the depth test, so what ends up in the stencil buffer is
        // exactly the crates' visible silhouette, holes and occlusions included.
        const VkStencilOpState write_one{
            .failOp = VK_STENCIL_OP_KEEP,
            .passOp = VK_STENCIL_OP_REPLACE,
            // A fragment that passes the stencil test and fails the depth test is behind
            // something. KEEP is what stops a crate hidden by a wall from stamping a
            // silhouette that would then be outlined through the wall.
            .depthFailOp = VK_STENCIL_OP_KEEP,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .compareMask = 0xFF,
            .writeMask = 0xFF,
            .reference = 1,
        };
        crate_pipeline_ =
            base()
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_stencil_format_, /*test=*/true, /*write=*/true,
                       VK_COMPARE_OP_LESS)
                .stencil_test(write_one, write_one)
                .build();

        // The decal, which is the depth half of the chapter.
        //
        // It lies in exactly the floor's plane, so the depth values the two produce at
        // any given pixel are the same number computed two different ways, and which one
        // is smaller comes down to the last bit of a float. The result is z-fighting: a
        // speckle of floor and decal that changes as the camera moves.
        //
        // depthBias is the fixed-function answer. A negative constant factor subtracts a
        // multiple of the smallest resolvable depth difference from every fragment this
        // pipeline produces, before the depth test sees it -- which pulls the decal
        // reliably in front of the floor without moving a single vertex.
        const auto decal = [&]() -> vkc::PipelineBuilder {
            return base()
                .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_stencil_format_, /*test=*/true, /*write=*/true,
                       VK_COMPARE_OP_LESS);
        };
        decal_pipeline_ = decal()
                              .shaders(shader, "vertexMain", "flatFragmentMain")
                              .depth_bias(kDecalBiasConstant, kDecalBiasSlope)
                              .build();
        decal_unbiased_pipeline_ =
            decal().shaders(shader, "vertexMain", "flatFragmentMain").build();

        // The outline, and the stencil half of the chapter.
        //
        // NOT_EQUAL 1 keeps only the fragments that land where no crate was drawn, which
        // of a slightly-enlarged crate is precisely the rim sticking out past the real
        // one. writeMask 0 because this pass reads the stencil and must not disturb it
        // for the next crate.
        //
        // Depth testing is off. The enlarged crate intersects the floor and would be
        // partly buried otherwise, and an outline is a screen-space decoration rather
        // than a thing in the world.
        const VkStencilOpState keep_where_unstamped{
            .failOp = VK_STENCIL_OP_KEEP,
            .passOp = VK_STENCIL_OP_KEEP,
            .depthFailOp = VK_STENCIL_OP_KEEP,
            .compareOp = VK_COMPARE_OP_NOT_EQUAL,
            .compareMask = 0xFF,
            .writeMask = 0x00,
            .reference = 1,
        };
        outline_pipeline_ =
            base()
                .shaders(shader, "vertexMain", "flatFragmentMain")
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_stencil_format_, /*test=*/false, /*write=*/false)
                .stencil_test(keep_where_unstamped, keep_where_unstamped)
                .build();

        vkDestroyShaderModule(device, shader, nullptr);

        context().name(floor_pipeline_, VK_OBJECT_TYPE_PIPELINE, "floor");
        context().name(crate_pipeline_, VK_OBJECT_TYPE_PIPELINE, "crate (stencil write)");
        context().name(decal_pipeline_, VK_OBJECT_TYPE_PIPELINE, "decal (biased)");
        context().name(decal_unbiased_pipeline_, VK_OBJECT_TYPE_PIPELINE,
                       "decal (unbiased)");
        context().name(outline_pipeline_, VK_OBJECT_TYPE_PIPELINE,
                       "outline (stencil test)");
    }

    vkc::Camera camera_;

    vkc::Buffer cube_vertices_;
    vkc::Buffer cube_indices_;
    vkc::Buffer quad_vertices_;
    vkc::Buffer quad_indices_;

    VkFormat depth_stencil_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_stencil_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    vkc::Image crate_map_;
    vkc::Image floor_map_;
    vkc::Sampler sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout material_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline floor_pipeline_ = VK_NULL_HANDLE;
    VkPipeline crate_pipeline_ = VK_NULL_HANDLE;
    VkPipeline decal_pipeline_ = VK_NULL_HANDLE;
    VkPipeline decal_unbiased_pipeline_ = VK_NULL_HANDLE;
    VkPipeline outline_pipeline_ = VK_NULL_HANDLE;

    bool outlines_ = true;
    bool depth_bias_ = true;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.1 - Depth & Stencil";
        DepthAndStencilApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
