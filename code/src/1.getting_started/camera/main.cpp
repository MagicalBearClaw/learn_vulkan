// 1.14 Camera
//
// Adds to 1.13: a camera you can fly. WASD to move, mouse to look, shift to go faster,
// Escape to quit.
//
// There is no Vulkan in this chapter at all. The view matrix was already a uniform in
// 1.13; the only change is that its contents now come from a position and two angles
// that the keyboard and mouse update. That is worth saying plainly, because "camera" is
// not a Vulkan concept and there is no VkCamera to look for: a camera is a matrix, and
// the matrix is the inverse of where the camera is.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
};

// A unit cube, four vertices per face.
//
// Twenty-four vertices rather than eight, because the eight corners each need three
// different texture coordinates -- one per face they belong to -- and a vertex is the
// whole bundle of attributes, not just a position. The same is true of normals, which
// is why a cube with hard edges always has 24 vertices in practice.
//
// Model space here is right-handed: +x right, +y up, +z toward the viewer. Each face is
// listed bottom-left, bottom-right, top-right, top-left as seen from outside, which
// makes every face wind counter-clockwise from outside.
constexpr float kH = 0.5F;
constexpr std::array<Vertex, 24> kVertices{{
    // front (+z)
    {{-kH, -kH, kH}, {0.0F, 1.0F}},
    {{kH, -kH, kH}, {1.0F, 1.0F}},
    {{kH, kH, kH}, {1.0F, 0.0F}},
    {{-kH, kH, kH}, {0.0F, 0.0F}},
    // back (-z)
    {{kH, -kH, -kH}, {0.0F, 1.0F}},
    {{-kH, -kH, -kH}, {1.0F, 1.0F}},
    {{-kH, kH, -kH}, {1.0F, 0.0F}},
    {{kH, kH, -kH}, {0.0F, 0.0F}},
    // left (-x)
    {{-kH, -kH, -kH}, {0.0F, 1.0F}},
    {{-kH, -kH, kH}, {1.0F, 1.0F}},
    {{-kH, kH, kH}, {1.0F, 0.0F}},
    {{-kH, kH, -kH}, {0.0F, 0.0F}},
    // right (+x)
    {{kH, -kH, kH}, {0.0F, 1.0F}},
    {{kH, -kH, -kH}, {1.0F, 1.0F}},
    {{kH, kH, -kH}, {1.0F, 0.0F}},
    {{kH, kH, kH}, {0.0F, 0.0F}},
    // top (+y)
    {{-kH, kH, kH}, {0.0F, 1.0F}},
    {{kH, kH, kH}, {1.0F, 1.0F}},
    {{kH, kH, -kH}, {1.0F, 0.0F}},
    {{-kH, kH, -kH}, {0.0F, 0.0F}},
    // bottom (-y)
    {{-kH, -kH, -kH}, {0.0F, 1.0F}},
    {{kH, -kH, -kH}, {1.0F, 1.0F}},
    {{kH, -kH, kH}, {1.0F, 0.0F}},
    {{-kH, -kH, kH}, {0.0F, 0.0F}},
}};

// Six faces, two triangles each, same 0-1-2 2-3-0 pattern the quad used.
constexpr std::array<uint16_t, 36> kIndices{
    0,  1,  2,  2,  3,  0,   // front
    4,  5,  6,  6,  7,  4,   // back
    8,  9,  10, 10, 11, 8,   // left
    12, 13, 14, 14, 15, 12,  // right
    16, 17, 18, 18, 19, 16,  // top
    20, 21, 22, 22, 23, 20,  // bottom
};

// Where each cube sits and how fast it turns. Three of them, arranged so they overlap
// on screen: without a depth buffer the last one drawn would simply win.
struct CubePlacement {
    glm::vec3 position;
    glm::vec3 axis;
    float speed;
};

constexpr std::array<CubePlacement, 10> kCubes{{
    {{0.0F, 0.0F, 0.0F}, {0.35F, 1.0F, 0.2F}, 0.7F},
    {{2.4F, 0.8F, -3.0F}, {1.0F, 0.3F, 0.0F}, -0.5F},
    {{-2.0F, -0.6F, -2.2F}, {0.2F, 0.4F, 1.0F}, 0.45F},
    {{1.2F, -1.4F, -5.5F}, {0.0F, 1.0F, 0.0F}, 0.9F},
    {{-3.1F, 1.6F, -6.0F}, {1.0F, 1.0F, 0.0F}, -0.3F},
    {{0.6F, 2.2F, -4.1F}, {0.5F, 0.0F, 1.0F}, 0.6F},
    {{-1.4F, 0.2F, -8.0F}, {1.0F, 0.0F, 0.3F}, -0.8F},
    {{3.4F, -1.0F, -7.2F}, {0.3F, 1.0F, 0.6F}, 0.35F},
    {{-4.0F, -1.8F, -4.6F}, {0.0F, 0.4F, 1.0F}, 0.55F},
    {{2.0F, 2.6F, -9.5F}, {1.0F, 0.6F, 0.2F}, -0.45F},
}};

// A first-person camera, kept as a position and two angles rather than as a matrix.
//
// Storing the matrix and multiplying rotations into it each frame accumulates floating
// point error and eventually shears; storing the angles and rebuilding the matrix from
// scratch every frame cannot drift. Two angles is also exactly the right number for a
// camera that should never roll, which is what you want for anything walking on ground.
struct Camera {
    glm::vec3 position{0.0F, 0.0F, 4.0F};

    // Degrees. Yaw turns left and right; pitch looks up and down. -90 faces down -z,
    // which is where the cubes are.
    float yaw = -90.0F;
    float pitch = 0.0F;

    float speed = 4.0F;        // world units per second
    float sensitivity = 0.1F;  // degrees per pixel of mouse motion

    // The direction the camera is facing, from the two angles. This is just spherical
    // coordinates: yaw sweeps around the vertical axis, pitch lifts out of the
    // horizontal plane.
    [[nodiscard]] glm::vec3 front() const {
        const float yaw_radians = glm::radians(yaw);
        const float pitch_radians = glm::radians(pitch);
        return glm::normalize(glm::vec3{
            std::cos(yaw_radians) * std::cos(pitch_radians),
            std::sin(pitch_radians),
            std::sin(yaw_radians) * std::cos(pitch_radians),
        });
    }

    // Right is perpendicular to both the facing direction and world up. Using *world*
    // up rather than the camera's own up is what stops the camera rolling when it
    // looks up or down.
    [[nodiscard]] glm::vec3 right() const {
        return glm::normalize(glm::cross(front(), glm::vec3{0.0F, 1.0F, 0.0F}));
    }

    [[nodiscard]] glm::mat4 view() const {
        return glm::lookAt(position, position + front(), glm::vec3{0.0F, 1.0F, 0.0F});
    }
};

// Camera and projection, shared by every draw in the frame.
struct Globals {
    glm::mat4 view;
    glm::mat4 projection;
};
static_assert(sizeof(Globals) == 128);

struct PushConstants {
    glm::mat4 model;
};
static_assert(sizeof(PushConstants) == 64);

class CameraApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        vertex_buffer_ = vkc::upload_to_device_local(
            context(), kVertices.data(), sizeof(kVertices),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer_ = vkc::upload_to_device_local(
            context(), kIndices.data(), sizeof(kIndices),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        texture_ = vkc::load_texture(context(), vkc::asset_path("textures/lvk_grid.png"));
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{
                                               .address_mode =
                                                   VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                           });

        depth_format_ = choose_depth_format();
        create_depth_buffer(swapchain().extent());

        create_uniform_buffers();
        create_descriptors();
        create_pipeline();

        spdlog::info(
            "WASD to move, space/ctrl for up and down, shift to sprint. "
            "Right-click to capture the mouse and look around, right-click again to "
            "release it. Escape quits.");
    }

    // The depth buffer is exactly the size of the colour attachment, so it has to
    // follow the swapchain when the window changes.
    void on_resize(VkExtent2D extent) override { create_depth_buffer(extent); }

    void on_event(const SDL_Event& event) override {
        // Relative mouse mode hides the cursor and reports motion as deltas with no
        // edges to hit, which is what a look control needs. Toggled with the right
        // mouse button so the window stays usable.
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
            event.button.button == SDL_BUTTON_RIGHT) {
            mouse_captured_ = !mouse_captured_;
            SDL_SetWindowRelativeMouseMode(window().handle(), mouse_captured_);
            return;
        }

        if (event.type == SDL_EVENT_MOUSE_MOTION && mouse_captured_) {
            camera_.yaw += event.motion.xrel * camera_.sensitivity;

            // Subtracted, not added: SDL's y grows downwards, and moving the mouse up
            // should raise the pitch.
            camera_.pitch -= event.motion.yrel * camera_.sensitivity;

            // Clamped just short of straight up and straight down. At exactly 90
            // degrees the facing direction becomes parallel to world up, the cross
            // product that gives `right` collapses to zero, and the view matrix is
            // undefined -- the camera flips over.
            camera_.pitch = std::clamp(camera_.pitch, -89.0F, 89.0F);
        }
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        depth_.destroy();
        sampler_.destroy();
        texture_.destroy();
        index_buffer_.destroy();
        vertex_buffer_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        move_camera();

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        const VkClearValue colour_clear{.color = {{0.02F, 0.02F, 0.04F, 1.0F}}};

        const VkRenderingAttachmentInfo colour_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = frame.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = colour_clear,
        };

        // Cleared to 1.0, the far plane, because the depth test keeps the *smaller*
        // value and everything must be closer than "infinitely far away".
        //
        // storeOp is DONT_CARE: nothing reads the depth buffer after the frame ends, so
        // telling the driver that lets it skip writing the whole thing back to memory.
        // On a tiled GPU that is a large saving, and it costs one word to say.
        const VkClearValue depth_clear{.depthStencil = {1.0F, 0}};

        const VkRenderingAttachmentInfo depth_attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = depth_.view(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue = depth_clear,
        };

        const VkRenderingInfo rendering{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {{0, 0}, frame.extent},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour_attachment,
            .pDepthAttachment = &depth_attachment,
            .pStencilAttachment = nullptr,
        };

        vkCmdBeginRendering(frame.cmd, &rendering);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

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

        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &descriptor_sets_[slot], 0,
                                nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        const float time = static_cast<float>(frame.frame_number) / 60.0F;

        for (const CubePlacement& cube : kCubes) {
            const glm::mat4 model =
                glm::translate(glm::mat4(1.0F), cube.position) *
                glm::rotate(glm::mat4(1.0F), time * cube.speed,
                            glm::normalize(cube.axis));

            const PushConstants push{.model = model};
            vkCmdPushConstants(frame.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(push), &push);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0,
                             0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Depth
    // ---------------------------------------------------------------------------

    // Depth formats are not all mandatory, so the first supported one wins.
    //
    // D32_SFLOAT is the one to want: a full float has plenty of precision and no
    // stencil bits to pay for. D24_UNORM_S8_UINT is the traditional alternative and is
    // what much older hardware prefers. D16 is genuinely too coarse for a scene of any
    // depth -- it is where z-fighting comes from.
    [[nodiscard]] VkFormat choose_depth_format() {
        constexpr std::array candidates{
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        };

        for (const VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(context().physical_device(), format,
                                                &properties);
            if ((properties.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                spdlog::info("Depth format: {}",
                             format == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "other");
                return format;
            }
        }
        throw std::runtime_error("No supported depth attachment format was found.");
    }

    void create_depth_buffer(VkExtent2D extent) {
        // The GPU must have finished with the old one. A resize already waits for idle,
        // but on_start has nothing in flight and this costs nothing there.
        context().wait_idle();

        // Same vkc::Image as the texture, with three things different: a depth format,
        // DEPTH_STENCIL_ATTACHMENT usage instead of SAMPLED, and a DEPTH aspect. The
        // aspect is the one that is easy to forget, and it has to match everywhere --
        // in the view, in every barrier, in every copy.
        depth_ = vkc::Image(context(), vkc::ImageDesc{
                                           .format = depth_format_,
                                           .extent = extent,
                                           .usage =
                                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                           .mip_levels = 1,
                                           .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                                       });

        // A new image is in UNDEFINED. Moving it to the attachment layout once here
        // means the render pass never has to, and since loadOp is CLEAR the previous
        // contents are discarded every frame anyway.
        vkc::immediate_submit(context(), [&](VkCommandBuffer cmd) {
            depth_.transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        });

        context().name(depth_.handle(), VK_OBJECT_TYPE_IMAGE, "depth buffer");
    }

    // ---------------------------------------------------------------------------
    // Matrices
    // ---------------------------------------------------------------------------

    [[nodiscard]] static glm::mat4 build_projection(VkExtent2D extent) {
        const float aspect = static_cast<float>(extent.width) /
                             static_cast<float>(extent.height);

        // 45 degrees vertical field of view, and a near plane at 0.1 rather than
        // something smaller. Precision in a depth buffer is concentrated near the near
        // plane, and halving `near` costs precision everywhere else -- pushing it out
        // is the first thing to try when surfaces start flickering against each other.
        glm::mat4 projection = glm::perspective(glm::radians(45.0F), aspect, 0.1F, 100.0F);

        // The Vulkan y flip.
        //
        // glm produces an OpenGL projection, whose clip-space y points up. Vulkan's
        // points down. Negating the matrix entry that scales y turns one into the
        // other, and without it everything renders upside down.
        //
        // The z range is already right because the project defines
        // GLM_FORCE_DEPTH_ZERO_TO_ONE, which makes glm emit 0..1 depth instead of
        // OpenGL's -1..1. Without it, everything nearer than the midpoint of the
        // frustum is clipped away as being behind the viewer.
        projection[1][1] *= -1.0F;

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

    // Reads the keyboard directly rather than reacting to key events.
    //
    // A key *event* fires once when the key goes down and once when it comes up, with
    // an OS-controlled repeat in between; that is right for typing and wrong for
    // movement. What movement wants is "is this key held down right now", which is what
    // the keyboard state array answers.
    void move_camera() {
        const bool* keys = SDL_GetKeyboardState(nullptr);

        // Scaled by the frame time, so the camera moves at the same speed whether the
        // machine renders at 60 frames per second or 300. Forgetting this is why some
        // old games run at double speed on modern hardware.
        float distance = camera_.speed * delta_time();
        if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) {
            distance *= 4.0F;
        }

        const glm::vec3 front = camera_.front();
        const glm::vec3 right = camera_.right();

        if (keys[SDL_SCANCODE_W]) {
            camera_.position += front * distance;
        }
        if (keys[SDL_SCANCODE_S]) {
            camera_.position -= front * distance;
        }
        if (keys[SDL_SCANCODE_A]) {
            camera_.position -= right * distance;
        }
        if (keys[SDL_SCANCODE_D]) {
            camera_.position += right * distance;
        }
        // Up and down along *world* up, not the camera's up, which is what makes a fly
        // camera feel controllable rather than disorientating.
        if (keys[SDL_SCANCODE_SPACE]) {
            camera_.position.y += distance;
        }
        if (keys[SDL_SCANCODE_LCTRL]) {
            camera_.position.y -= distance;
        }
    }

    void update_globals(size_t slot, VkExtent2D extent) {
        const Globals globals{
            .view = camera_.view(),
            .projection = build_projection(extent),
        };

        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    // ---------------------------------------------------------------------------
    // Descriptors and pipeline
    // ---------------------------------------------------------------------------

    void create_descriptors() {
        const VkDevice device = context().device();

        // Two bindings this time: the per-frame matrices, and the texture.
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
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

        const VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                             &set_layout_));

        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, vkc::kFramesInFlight},
        }};

        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight,
            .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const std::array<VkDescriptorSetLayout, vkc::kFramesInFlight> layouts{
            set_layout_, set_layout_};

        const VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = vkc::kFramesInFlight,
            .pSetLayouts = layouts.data(),
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets_.data()));

        for (size_t i = 0; i < vkc::kFramesInFlight; ++i) {
            const VkDescriptorBufferInfo buffer_info{
                .buffer = uniform_buffers_[i].handle(),
                .offset = 0,
                .range = sizeof(Globals),
            };

            const VkDescriptorImageInfo image_info{
                .sampler = sampler_.handle(),
                .imageView = texture_.view(),
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            const std::array<VkWriteDescriptorSet, 2> writes{{
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = descriptor_sets_[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer_info,
                    .pTexelBufferView = nullptr,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = descriptor_sets_[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
            }};

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }
    }

    void create_pipeline() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        const std::array<VkVertexInputAttributeDescription, 2> attributes{{
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
        }};

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr,
                                        &pipeline_layout_));

        // One module, both stages: cube.slang compiles to a single SPIR-V
        // binary with a vertexMain and a fragmentMain entry point in it.
        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "cube.slang");

        pipeline_ =
            vkc::PipelineBuilder(device)
                .shaders(shader)
                .vertex_input(std::span(&binding, 1), attributes)
                // Throw away faces pointing away from the camera: half the triangles of
                // a closed object, discarded before they are rasterised.
                //
                // COUNTER_CLOCKWISE, matching how the cube is wound when seen from
                // outside -- which is not obvious, because the y flip in the projection
                // is a mirror and a mirror reverses winding. It works out because there
                // are *two* mirrors: glm::perspective is right-handed and negates z on
                // the way to clip space as well. Two reflections compose to a rotation,
                // so the winding on screen is unchanged.
                //
                // If your cube looks like an open box seen from inside, this is the
                // line to flip. Do not guess -- try both and look.
                .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .depth(depth_format_, /*test=*/true, /*write=*/true, VK_COMPARE_OP_LESS)
                .colour_attachment(swapchain().format())
                .layout(pipeline_layout_)
                .build();

        vkDestroyShaderModule(device, shader, nullptr);
    }

    Camera camera_;
    bool mouse_captured_ = false;

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    vkc::Image texture_;
    vkc::Sampler sampler_;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> descriptor_sets_{};

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.14 - Camera";
        CameraApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
