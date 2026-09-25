// 4.3 Offscreen rendering & render targets
//
// Every chapter so far has drawn into a swapchain image, because that is the image the
// window system hands over and there was no reason to want a different one. This chapter
// stops doing that. The scene is drawn into an image this sample allocates itself, and a
// second pass then reads that image as a texture and writes the result to the screen.
//
// Three things make it work, and only one of them is new:
//
//   * an image created with COLOR_ATTACHMENT_BIT *and* SAMPLED_BIT, which is the whole
//     of what a "render target" is
//   * the same vkc::begin_rendering() every chapter since 1.13 has called, handed a
//     different view -- there is no VkFramebuffer to build and nothing to rebind
//   * a barrier between the two passes, moving the image from the layout it is written
//     in to the layout it is read in. That one is new, and it is the chapter.
//
// The scene itself is 4.2's, drawn exactly as 4.2 left it, so that any difference on
// screen is the fault of the new pass rather than of the old one.
//
//   1  cycles the post-processing effect
//   2  cycles the render scale: the scene image need not be the size of the window

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
constexpr float kFloorHalf = 22.0F;
constexpr float kFloorTiles = 19.0F;

// 4.2's two quads: the floor in world space, and a unit upright quad that every fern and
// every pane of glass is an instance of.
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

constexpr std::array<Placement, 6> kFerns{{
    {{-3.30F, kFloorY, 1.70F}, 0.30F, {2.0F, 2.0F}},
    {{-2.10F, kFloorY, 2.90F}, -0.55F, {1.7F, 1.7F}},
    {{-2.70F, kFloorY, 2.30F}, 0.95F, {1.5F, 1.5F}},
    {{2.55F, kFloorY, 2.10F}, 0.80F, {2.1F, 2.1F}},
    {{3.40F, kFloorY, 1.30F}, -0.20F, {1.6F, 1.6F}},
    {{0.35F, kFloorY, 3.30F}, 0.15F, {1.8F, 1.8F}},
}};

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

constexpr glm::vec3 kSunDirection{-0.45F, -1.0F, -0.38F};
constexpr glm::vec3 kSunColour{0.92F, 0.88F, 0.80F};
constexpr glm::vec3 kAmbient{0.24F, 0.26F, 0.31F};
constexpr float kShininess = 48.0F;

// The effects the second pass can apply, in the order key 1 cycles them. These match
// the constants at the top of post_process.slang.
constexpr std::array<const char*, 7> kEffectNames{
    "none", "grayscale", "invert", "sharpen", "blur", "edges", "alpha channel",
};

// Fractions of the window the scene image can be rendered at. The first is the obvious
// one; the rest exist to make the point that nothing ties a render target to the size of
// the window it eventually reaches.
constexpr std::array<float, 4> kRenderScales{1.0F, 0.5F, 0.25F, 2.0F};

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

struct PushConstants {
    glm::mat4 model;                         // 0
    std::array<glm::vec4, 3> normal_matrix;  // 64
    glm::vec4 colour;                        // 112
};
static_assert(offsetof(PushConstants, colour) == 112);
static_assert(sizeof(PushConstants) == 128);

// The second pass's push block. Tiny, and a different shape from the scene's -- which is
// why the two passes have two pipeline layouts rather than sharing one.
struct PostPushConstants {
    glm::vec2 texel;  // 1 / scene image size, in uv units
    int32_t effect;
    int32_t padding = 0;
};
static_assert(sizeof(PostPushConstants) == 16);

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

[[nodiscard]] glm::vec3 centre_of(const Placement& placement) {
    return placement.position + glm::vec3(0.0F, 0.5F * placement.size.y, 0.0F);
}

class OffscreenRenderingApp : public vkc::App {
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

        depth_format_ = vkc::choose_depth_format(context().physical_device());

        // The scene image takes the swapchain's format, which keeps the two passes
        // agreeing about what a colour means and is not a requirement. A render target
        // is free to be any format the GPU supports for the usage asked of it; the HDR
        // chapter in Part 5 puts R16G16B16A16_SFLOAT here for exactly that freedom.
        offscreen_format_ = swapchain().format();

        load_textures();
        create_uniform_buffers();
        create_descriptors();
        create_offscreen_target(swapchain().extent());
        create_pipelines();
        write_post_descriptor();

        camera_.position = {0.0F, 2.60F, 12.60F};
        camera_.pitch = -14.0F;

        spdlog::info("1 post-processing effect, 2 render scale.");
    }

    void on_resize(VkExtent2D extent) override {
        context().wait_idle();
        create_offscreen_target(extent);
        write_post_descriptor();
    }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());

        if (event.type != SDL_EVENT_KEY_DOWN || event.key.repeat) {
            return;
        }
        switch (event.key.scancode) {
            case SDL_SCANCODE_1:
                effect_ = (effect_ + 1) % static_cast<int32_t>(kEffectNames.size());
                spdlog::info("Effect: {}", kEffectNames[static_cast<size_t>(effect_)]);
                break;
            case SDL_SCANCODE_2: {
                scale_index_ = (scale_index_ + 1) % kRenderScales.size();
                // The scene image is being replaced, and the GPU may still be reading
                // the old one. There is a cheaper answer than stalling -- keep the old
                // image alive for kFramesInFlight more frames -- and it is not worth it
                // for something a key press triggers.
                context().wait_idle();
                create_offscreen_target(swapchain().extent());
                write_post_descriptor();
                spdlog::info("Render scale {:.0f}%: scene image is {}x{}",
                             kRenderScales[scale_index_] * 100.0F,
                             offscreen_extent_.width, offscreen_extent_.height);
                break;
            }
            default:
                break;
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, post_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, post_layout_, nullptr);
        vkDestroyPipeline(device, pane_pipeline_, nullptr);
        vkDestroyPipeline(device, fern_pipeline_, nullptr);
        vkDestroyPipeline(device, opaque_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, post_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, material_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        post_sampler_.destroy();
        clamp_sampler_.destroy();
        sampler_.destroy();
        fern_map_.destroy();
        floor_map_.destroy();
        crate_map_.destroy();
        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        offscreen_depth_.destroy();
        offscreen_colour_.destroy();
        quad_indices_.destroy();
        quad_vertices_.destroy();
        cube_indices_.destroy();
        cube_vertices_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, offscreen_extent_);

        render_scene(frame, slot);
        render_post(frame);
    }

private:
    // ---------------------------------------------------------------------------
    // Pass one: the scene, into an image of our own
    // ---------------------------------------------------------------------------

    void render_scene(const vkc::FrameInfo& frame, size_t slot) {
        // Into COLOR_ATTACHMENT_OPTIMAL, from nothing.
        //
        // UNDEFINED as the old layout means "I do not want what is in there", and it is
        // the right answer every frame: the colour attachment's loadOp is CLEAR, so the
        // previous frame's pixels are about to be thrown away regardless. Naming the
        // real old layout instead would ask the driver to preserve contents nobody reads.
        //
        // The source scope is not empty, though, and this is the part that is easy to
        // get wrong. There is one scene image and two frames in flight, so the previous
        // frame's *second* pass may still be sampling this image when the current
        // frame's first pass wants to overwrite it. Writing after a read is still a
        // hazard. FRAGMENT_SHADER / SHADER_SAMPLED_READ is what that previous reader was.
        vkc::image_barrier(frame.cmd, offscreen_colour_.handle(), VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        // And this is the whole of "render to texture" in Vulkan 1.3: the function every
        // chapter since 1.13 has called, with a different view and a different extent.
        // There is no framebuffer object to create, nothing to bind, and nothing to
        // unbind afterwards.
        vkc::begin_rendering(frame.cmd, offscreen_colour_.view(), offscreen_depth_.view(),
                             offscreen_extent_, {{0.05F, 0.06F, 0.09F, 1.0F}});

        // The viewport follows the *target*, not the window. Getting this wrong is the
        // classic first bug of a render-to-texture pass: the image is the right size and
        // the scene is drawn into one corner of it.
        set_viewport(frame.cmd, offscreen_extent_);

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        bind_geometry(frame.cmd, quad_vertices_, quad_indices_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque_pipeline_);
        bind_material(frame.cmd, floor_set_);
        push(frame.cmd, glm::mat4(1.0F), kWhite);
        vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kFloorFirstIndex, 0, 0);

        bind_geometry(frame.cmd, cube_vertices_, cube_indices_);
        bind_material(frame.cmd, crate_set_);
        for (const Placement& crate : kCrates) {
            push(frame.cmd,
                 glm::translate(model_of(crate), glm::vec3(0.0F, 0.5F, 0.0F)), kWhite);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kCubeIndices.size()), 1, 0,
                             0, 0);
        }

        bind_geometry(frame.cmd, quad_vertices_, quad_indices_);
        bind_material(frame.cmd, fern_set_);
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fern_pipeline_);
        for (const Placement& fern : kFerns) {
            push(frame.cmd, model_of(fern), kWhite);
            vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }

        // 4.2's back-to-front sort, still required for the same reason.
        std::array<size_t, kPanes.size()> order{};
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        const glm::vec3 eye = camera_.position;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return glm::dot(centre_of(kPanes[a]) - eye, centre_of(kPanes[a]) - eye) >
                   glm::dot(centre_of(kPanes[b]) - eye, centre_of(kPanes[b]) - eye);
        });

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pane_pipeline_);
        for (const size_t index : order) {
            push(frame.cmd, model_of(kPanes[index]), kPaneColours[index]);
            vkCmdDrawIndexed(frame.cmd, kQuadIndexCount, 1, kUprightFirstIndex, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);
    }

    // ---------------------------------------------------------------------------
    // Pass two: one triangle, into the swapchain
    // ---------------------------------------------------------------------------

    void render_post(const vkc::FrameInfo& frame) {
        // The scene image stops being an attachment and becomes a texture. Two things
        // happen here and they are worth separating: the *layout* changes, so the
        // hardware can rearrange the memory into whatever a sampler likes; and the
        // *dependency* is recorded, so the fragment shaders of the second pass cannot
        // start reading before the colour writes of the first pass have landed.
        //
        // Without this barrier the second pass would sample an image that is still
        // being written. The picture would usually still look right, which is the worst
        // possible failure mode -- the synchronisation validation layer is what catches
        // it, and the article shows what it says.
        vkc::image_barrier(frame.cmd, offscreen_colour_.handle(),
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // The second pass's rendering info, written out rather than taken from vkcommon,
        // because it differs from every previous one in this series in two ways.
        //
        // There is no depth attachment at all: one triangle covering the screen has
        // nothing to be occluded by and nothing to occlude.
        //
        // And the colour loadOp is DONT_CARE rather than CLEAR. The triangle writes
        // every pixel, so clearing first would be work whose every result is
        // immediately overwritten. DONT_CARE is not "leave the old contents" -- it is a
        // promise that nothing will read them, which lets a tiled GPU skip fetching the
        // attachment into tile memory in the first place.
        const VkRenderingAttachmentInfo colour{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = frame.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {},
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
            .pDepthAttachment = nullptr,
            .pStencilAttachment = nullptr,
        };
        vkCmdBeginRendering(frame.cmd, &rendering);

        // Back to the window's size. The scene image may be a quarter of this, and the
        // sampler is what bridges the difference.
        set_viewport(frame.cmd, frame.extent);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, post_pipeline_);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout_,
                                0, 1, &post_set_, 0, nullptr);

        const PostPushConstants constants{
            .texel = {1.0F / static_cast<float>(offscreen_extent_.width),
                      1.0F / static_cast<float>(offscreen_extent_.height)},
            .effect = effect_,
        };
        vkCmdPushConstants(frame.cmd, post_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(constants), &constants);

        // Three vertices, no vertex buffer, no index buffer. The shader builds them.
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);
    }

    // ---------------------------------------------------------------------------
    // The render target
    // ---------------------------------------------------------------------------

    void create_offscreen_target(VkExtent2D window_extent) {
        const float scale = kRenderScales[scale_index_];
        offscreen_extent_ = {
            .width = std::max(1U, static_cast<uint32_t>(
                                      static_cast<float>(window_extent.width) * scale)),
            .height = std::max(1U, static_cast<uint32_t>(
                                       static_cast<float>(window_extent.height) * scale)),
        };

        // Two usage bits, and that is the entire difference between a texture and a
        // render target. COLOR_ATTACHMENT_BIT lets it be written by a rendering pass;
        // SAMPLED_BIT lets it be read by a shader. An image with only the first is a
        // target that nothing can look at; an image with only the second is 1.11's
        // texture. Asking for both is what makes a two-pass frame possible.
        offscreen_colour_ = vkc::Image(
            context(), vkc::ImageDesc{
                           .format = offscreen_format_,
                           .extent = offscreen_extent_,
                           .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT,
                           .mip_levels = 1,
                           .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                       });

        // The scene still needs a depth buffer, and it belongs to the scene image rather
        // than to the window: it has to match the target being rendered into, not the
        // one being presented.
        offscreen_depth_ =
            vkc::create_depth_buffer(context(), depth_format_, offscreen_extent_);

        context().name(offscreen_colour_.handle(), VK_OBJECT_TYPE_IMAGE, "scene colour");
        context().name(offscreen_depth_.handle(), VK_OBJECT_TYPE_IMAGE, "scene depth");
    }

    // Points the second pass's descriptor at whatever the scene image currently is.
    // Every resize and every render-scale change destroys the old image and its view, so
    // the descriptor has to be pointed at the new one before the next frame reads it.
    void write_post_descriptor() {
        const VkDescriptorImageInfo image_info{
            .sampler = post_sampler_.handle(),
            .imageView = offscreen_colour_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = post_set_,
            .dstBinding = 0,
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

    // Takes an extent rather than a FrameInfo, for the first time in this series: there
    // are two targets in this frame and they are different sizes.
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

    // The projection's aspect ratio comes from the scene image, not the window. They are
    // the same shape here because the render scale is uniform, but the rule is that the
    // matrix belongs to the target being drawn into.
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
                                       vkc::asset_path("textures/lvk_ground.png"),
                                       VK_FORMAT_R8G8B8A8_SRGB);
        fern_map_ = vkc::load_texture(context(),
                                      vkc::asset_path("textures/lvk_foliage.png"),
                                      VK_FORMAT_R8G8B8A8_SRGB);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});
        clamp_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE});

        // The sampler the second pass reads the scene image with. Three deliberate
        // differences from the scene's own sampler:
        //
        //   CLAMP_TO_EDGE, because a 3x3 kernel on the outermost pixel reaches one texel
        //   past the edge, and REPEAT would fetch the opposite side of the screen;
        //
        //   anisotropy off, which is meaningless for an image sampled flat-on;
        //
        //   and it will be filtering LINEAR, which matters only when the render scale is
        //   not 1 -- at 25% it is what turns four screen pixels per scene texel from
        //   blocks into a smooth stretch.
        post_sampler_ = vkc::Sampler(
            context(),
            vkc::SamplerDesc{.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .anisotropy = false});
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
        // The second pass's set has the same shape as a material's -- one combined image
        // sampler -- and is a separate layout anyway, because it is a separate pipeline
        // layout with a different push-constant range behind it.
        VK_CHECK(vkCreateDescriptorSetLayout(device, &texture_layout_info, nullptr,
                                             &post_set_layout_));

        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + 4,
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

        // Allocated once and rewritten whenever the scene image is replaced. A
        // descriptor set is a handle to a slot, not a copy of what is in it.
        const VkDescriptorSetAllocateInfo post_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &post_set_layout_,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &post_alloc, &post_set_));

        context().name(crate_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "crate material");
        context().name(floor_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "floor material");
        context().name(fern_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "fern material");
        context().name(post_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "scene image");
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

        const VkShaderModule scene_shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "offscreen_rendering.slang");

        // Every scene pipeline is told the scene image's format, not the swapchain's.
        // The two happen to be equal here, and the pipeline's attachment formats have to
        // match the images it is actually used with -- so naming the right one is the
        // habit worth having, and the validation layers will say so if it drifts.
        const auto base = [&]() -> vkc::PipelineBuilder {
            return vkc::PipelineBuilder(device)
                .shaders(scene_shader)
                .vertex_input(std::span(&binding, 1), attributes)
                .colour_attachment(offscreen_format_)
                .layout(pipeline_layout_);
        };

        const VkPipelineColorBlendAttachmentState source_over{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            // 4.2's finding, and it matters more here than it did there. A swapchain
            // image's alpha is mostly ignored; a render target's alpha is read back by
            // the next pass, which is exactly what effect 6 does to check it.
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };

        opaque_pipeline_ = base()
                               .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                               .depth(depth_format_, /*test=*/true, /*write=*/true)
                               .build();

        fern_pipeline_ = base()
                             .shaders(scene_shader, "vertexMain", "cutoutFragmentMain")
                             .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                             .depth(depth_format_, /*test=*/true, /*write=*/true)
                             .build();

        pane_pipeline_ = base()
                             .shaders(scene_shader, "vertexMain", "flatFragmentMain")
                             .cull(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                             .depth(depth_format_, /*test=*/true, /*write=*/false)
                             .colour_blend(source_over)
                             .build();

        vkDestroyShaderModule(device, scene_shader, nullptr);

        create_post_pipeline();

        context().name(opaque_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene opaque");
        context().name(fern_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene ferns");
        context().name(pane_pipeline_, VK_OBJECT_TYPE_PIPELINE, "scene glass");
        context().name(post_pipeline_, VK_OBJECT_TYPE_PIPELINE, "post-processing");
    }

    void create_post_pipeline() {
        const VkDevice device = context().device();

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PostPushConstants),
        };
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &post_set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &post_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "post_process.slang");

        // No vertex_input(), no depth(), no blending: the simplest pipeline in the
        // series since 1.7's, which is what a fullscreen pass should be. It renders into
        // the swapchain, so it is the one pipeline here told the swapchain's format.
        post_pipeline_ = vkc::PipelineBuilder(device)
                             .shaders(shader)
                             .colour_attachment(swapchain().format())
                             .layout(post_layout_)
                             .build();

        vkDestroyShaderModule(device, shader, nullptr);
    }

    vkc::Camera camera_;

    vkc::Buffer cube_vertices_;
    vkc::Buffer cube_indices_;
    vkc::Buffer quad_vertices_;
    vkc::Buffer quad_indices_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkFormat offscreen_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D offscreen_extent_{};
    vkc::Image offscreen_colour_;
    vkc::Image offscreen_depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    vkc::Image crate_map_;
    vkc::Image floor_map_;
    vkc::Image fern_map_;
    vkc::Sampler sampler_;
    vkc::Sampler clamp_sampler_;
    vkc::Sampler post_sampler_;

    VkDescriptorSet crate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet floor_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fern_set_ = VK_NULL_HANDLE;
    VkDescriptorSet post_set_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout material_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout post_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline opaque_pipeline_ = VK_NULL_HANDLE;
    VkPipeline fern_pipeline_ = VK_NULL_HANDLE;
    VkPipeline pane_pipeline_ = VK_NULL_HANDLE;

    VkPipelineLayout post_layout_ = VK_NULL_HANDLE;
    VkPipeline post_pipeline_ = VK_NULL_HANDLE;

    int32_t effect_ = 0;
    size_t scale_index_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 4.3 - Offscreen Rendering";
        OffscreenRenderingApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
