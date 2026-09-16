// 1.11 Textures
//
// Adds to 1.10: an image sampled by the fragment shader.
//
// A texture is three objects, not one, and keeping them apart is the thing to take away
// from this chapter:
//
//   VkImage        the pixels, in a layout chosen by the driver for speed
//   VkImageView    how a shader interprets them: format, mip range, array layers
//   VkSampler      how to read between them: filtering, mip selection, wrapping
//
// OpenGL merged all three into one texture object, which is why its sampler state was
// stuck to the texture and why binding the same image with two different filters meant
// two copies. In Vulkan a sampler is independent, and a handful of them serve a whole
// scene.
//
// Getting pixels in is the buffer story from 1.8 with one extra idea: an image has a
// *layout*, and the layout that is fast to copy into is not the layout that is fast to
// sample from, so the transitions are explicit.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/check.hpp>
#include <vkc/frame.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

// stb is a header-only library: one translation unit has to compile its implementation.
// STB_IMAGE_STATIC keeps the result private to this file, which matters from chapter 3.1
// onward -- Assimp bundles its own stb_image and exports the same names.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>

namespace {

struct Vertex {
    glm::vec2 position;
    glm::vec2 uv;
};

// The quad, now with texture coordinates.
//
// u runs 0 to 1 left to right and v runs 0 to 1 top to bottom. That top-to-bottom v is
// the same downward y as everywhere else in Vulkan, and it means an image loaded in the
// usual top-row-first order maps on the right way up with no flip. OpenGL's v points
// up, which is why so much OpenGL code calls stbi_set_flip_vertically_on_load.
constexpr std::array<Vertex, 4> kVertices{{
    {{-1.0F, -1.0F}, {0.0F, 0.0F}},  // top left
    {{1.0F, -1.0F}, {1.0F, 0.0F}},   // top right
    {{1.0F, 1.0F}, {1.0F, 1.0F}},    // bottom right
    {{-1.0F, 1.0F}, {0.0F, 1.0F}},   // bottom left
}};

constexpr std::array<uint16_t, 6> kIndices{0, 1, 2, 2, 3, 0};

struct PushConstants {
    glm::vec2 offset;
    float scale;
    float uv_scale;  // >1 tiles the texture, which is what the sampler's address mode
                     // decides how to handle
};
static_assert(sizeof(PushConstants) == 16);

// Two copies of the quad: one showing the texture once, one tiling it four times over
// the same area, which minifies it enough that the mip chain matters.
constexpr std::array<PushConstants, 2> kQuads{{
    {{-0.5F, 0.0F}, 0.45F, 1.0F},
    {{0.5F, 0.0F}, 0.45F, 4.0F},
}};

class TextureApp : public vkc::App {
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

        load_texture(vkc::asset_path("textures/lvk_grid.png"));
        create_sampler();
        create_descriptors();
        create_pipeline();
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);

        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);

        vkDestroySampler(device, sampler_, nullptr);
        vkDestroyImageView(device, texture_view_, nullptr);
        vmaDestroyImage(context().allocator(), texture_, texture_allocation_);

        index_buffer_.destroy();
        vertex_buffer_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        const VkClearValue clear{.color = {{0.02F, 0.02F, 0.04F, 1.0F}}};

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
            .clearValue = clear,
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
            .pDepthAttachment = nullptr,
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

        // One descriptor set, bound once. Nothing in it is per-frame this time: the
        // texture and the sampler are the same every frame, so unlike 1.10's uniform
        // buffer there is no need for one copy per frame in flight.
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0,
                             VK_INDEX_TYPE_UINT16);

        for (const PushConstants& quad : kQuads) {
            vkCmdPushConstants(frame.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(PushConstants), &quad);
            vkCmdDrawIndexed(frame.cmd, static_cast<uint32_t>(kIndices.size()), 1, 0, 0,
                             0);
        }

        vkCmdEndRendering(frame.cmd);
    }

private:
    // ---------------------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------------------

    void load_texture(const std::filesystem::path& path) {
        int width = 0;
        int height = 0;
        int channels = 0;

        // Forcing 4 channels regardless of what the file has. Three-channel formats are
        // widely supported for *buffers* and poorly supported for *images*, and one
        // wasted byte per texel is a much better trade than a format-support matrix.
        stbi_uc* pixels =
            stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (pixels == nullptr) {
            throw std::runtime_error(std::format(
                "Could not load '{}': {}. Did you run tools/make_textures.py?",
                path.string(), stbi_failure_reason()));
        }

        const VkDeviceSize size =
            static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

        // How many times the image can be halved before it reaches 1x1, inclusive.
        // A 512x512 texture has 10 levels: 512, 256, 128 ... 2, 1.
        mip_levels_ = static_cast<uint32_t>(std::bit_width(static_cast<uint32_t>(
                          std::max(width, height))));

        spdlog::info("Loaded {}x{} texture ({} channels in the file), {} mip levels",
                     width, height, channels, mip_levels_);

        // Same staging pattern as a vertex buffer: a host-visible buffer the CPU writes
        // and the GPU copies out of.
        vkc::Buffer staging(context().allocator(), size,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
        std::memcpy(staging.mapped(), pixels, static_cast<size_t>(size));
        stbi_image_free(pixels);

        create_image(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

        vkc::immediate_submit(context(), [&](VkCommandBuffer cmd) {
            // UNDEFINED means "I do not care what is in there", which is exactly true
            // for a brand new image, and lets the driver skip preserving its contents.
            transition(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, mip_levels_);

            const VkBufferImageCopy region{
                .bufferOffset = 0,
                // Zero means "tightly packed", which is what stb gives us. Non-zero
                // values describe a sub-rectangle of a larger source image.
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height), 1},
            };
            vkCmdCopyBufferToImage(cmd, staging.handle(), texture_,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            generate_mipmaps(cmd, width, height);
        });

        context().name(texture_, VK_OBJECT_TYPE_IMAGE, "lvk_grid");

        create_texture_view();
    }

    void create_image(uint32_t width, uint32_t height) {
        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            // _SRGB, not _UNORM. The hardware converts each texel from sRGB to linear
            // as it is sampled, for free, and the swapchain converts back on the way
            // out. Shading happens in between, in linear space, which is the only
            // space in which adding two lights together is meaningful. Chapter 5.2
            // takes this apart properly.
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .extent = {width, height, 1},
            .mipLevels = mip_levels_,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            // OPTIMAL lets the driver swizzle the texels into whatever order its
            // texture units like -- usually some tiled or Morton order, never plain
            // rows. LINEAR would keep them in row order and is dramatically slower to
            // sample; it exists for images the CPU has to read directly.
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            // TRANSFER_SRC as well as DST, because generating mipmaps reads from level
            // n to write level n+1.
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
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

        VK_CHECK(vmaCreateImage(context().allocator(), &image_info, &alloc_info,
                                &texture_, &texture_allocation_, nullptr));
    }

    // Moves some mip levels of the texture from one layout to another.
    //
    // This is 1.5's transition_image with a mip range added, and with real stage and
    // access masks instead of ALL_COMMANDS: mip generation issues one of these between
    // every pair of levels, so the sloppy version would serialise the whole chain.
    void transition(VkCommandBuffer cmd, VkImageLayout from, VkImageLayout to,
                    uint32_t base_mip, uint32_t mip_count) {
        const VkImageMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = from,
            .newLayout = to,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = texture_,
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = base_mip,
                    .levelCount = mip_count,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };

        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    // Builds the mip chain on the GPU by repeatedly halving the previous level.
    //
    // Vulkan has no vkCmdGenerateMipmaps. That is deliberate: a box filter is rarely
    // the best answer, and a real pipeline generates its mips offline with a better
    // one. Doing it at load time with vkCmdBlitImage is the pragmatic middle ground and
    // it is worth writing once to see what "a layout per mip level" really means.
    void generate_mipmaps(VkCommandBuffer cmd, int width, int height) {
        // Blitting with a linear filter is an optional format feature. It is supported
        // for the usual 8-bit formats everywhere in practice, but "in practice" is not
        // a guarantee, so it is checked.
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(context().physical_device(),
                                            VK_FORMAT_R8G8B8A8_SRGB, &properties);
        if ((properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
            throw std::runtime_error(
                "This GPU cannot linearly filter R8G8B8A8_SRGB blits, so mipmaps "
                "cannot be generated this way.");
        }

        int mip_width = width;
        int mip_height = height;

        for (uint32_t level = 1; level < mip_levels_; ++level) {
            // The level we are about to read from was just written, either by the
            // buffer copy (level 0) or by the previous blit. Move it to TRANSFER_SRC
            // and let the barrier order the write before the read.
            transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, level - 1, 1);

            const int next_width = std::max(mip_width / 2, 1);
            const int next_height = std::max(mip_height / 2, 1);

            const VkImageBlit blit{
                .srcSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = level - 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
                .dstSubresource =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = level,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .dstOffsets = {{0, 0, 0}, {next_width, next_height, 1}},
            };

            vkCmdBlitImage(cmd, texture_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texture_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                           VK_FILTER_LINEAR);

            // That level is finished; it will never be written again, so put it in the
            // layout the shader wants and forget about it.
            transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, level - 1, 1);

            mip_width = next_width;
            mip_height = next_height;
        }

        // The last level was only ever a blit destination, so it is still TRANSFER_DST.
        transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mip_levels_ - 1, 1);
    }

    void create_texture_view() {
        const VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = texture_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .components = {},  // VK_COMPONENT_SWIZZLE_IDENTITY is zero
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    // All of them. A view that exposes a subset of the mip chain is how
                    // you sample one specific level.
                    .levelCount = mip_levels_,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        VK_CHECK(vkCreateImageView(context().device(), &view_info, nullptr,
                                   &texture_view_));
    }

    void create_sampler() {
        const float max_anisotropy =
            context().gpu_properties().limits.maxSamplerAnisotropy;

        const VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            // Magnification: what happens when one texel covers many pixels. LINEAR
            // blends the four nearest texels; NEAREST gives you visible squares, which
            // is right for pixel art and wrong for almost everything else.
            .magFilter = VK_FILTER_LINEAR,
            // Minification: many texels per pixel. This is where aliasing lives.
            .minFilter = VK_FILTER_LINEAR,
            // How to blend between two mip levels. LINEAR gives trilinear filtering;
            // NEAREST picks one level and shows a visible seam where it changes.
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            // What u or v outside 0..1 means. REPEAT tiles, which is what the
            // right-hand quad relies on.
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0F,
            // Anisotropic filtering: sample along the direction the texture is
            // stretched rather than taking one square footprint. It costs little and it
            // is the single biggest visual improvement available on surfaces seen at a
            // glancing angle -- floors, roads, walls.
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = max_anisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0F,
            // Not mip_levels_: VK_LOD_CLAMP_NONE means "no limit", and it keeps the
            // sampler correct if the view later exposes a different number of levels.
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        VK_CHECK(vkCreateSampler(context().device(), &sampler_info, nullptr, &sampler_));

        spdlog::info("Sampler created with maxAnisotropy={}", max_anisotropy);
    }

    void create_descriptors() {
        const VkDevice device = context().device();

        // COMBINED_IMAGE_SAMPLER is one descriptor holding both a view and a sampler.
        // Vulkan also has SAMPLED_IMAGE and SAMPLER as separate types, which is how you
        // pair one image with several samplers without duplicating anything. Combined
        // is simpler and is what most code starts with.
        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };

        const VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                             &set_layout_));

        const VkDescriptorPoolSize pool_size{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
        };

        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout_,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_));

        const VkDescriptorImageInfo image_info{
            .sampler = sampler_,
            .imageView = texture_view_,
            // The layout the image will be in when it is sampled -- a promise, checked
            // by the validation layers at draw time. Ours got there at the end of mip
            // generation and stays there.
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptor_set_,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
                .format = VK_FORMAT_R32G32_SFLOAT,
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

        // One module, both stages: textured.slang compiles to a single SPIR-V
        // binary with a vertexMain and a fragmentMain entry point in it.
        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "textured.slang");

        pipeline_ = vkc::PipelineBuilder(device)
                        .shaders(shader)
                        .vertex_input(std::span(&binding, 1), attributes)
                        .colour_attachment(swapchain().format())
                        .layout(pipeline_layout_)
                        .build();

        vkDestroyShaderModule(device, shader, nullptr);
    }

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;

    VkImage texture_ = VK_NULL_HANDLE;
    VmaAllocation texture_allocation_ = VK_NULL_HANDLE;
    VkImageView texture_view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    uint32_t mip_levels_ = 1;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 1.11 - Textures";
        TextureApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
