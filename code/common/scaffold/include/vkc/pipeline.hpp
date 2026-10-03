#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vkc {

// Compiles one of this chapter's Slang shaders to SPIR-V and wraps it in a shader
// module. Chapter 1.7 wrote this out by hand; nothing here is new.
//
// The compile happens now, at run time, not during the build -- libslang is linked into
// the sample. One Slang file becomes one SPIR-V module holding every stage, so
// `shader_name` is the .slang file the chapter's CMakeLists lists ("quad.slang"), and
// one call here covers the whole pipeline.
//
// A compile error throws, with Slang's diagnostics in the message. The returned module
// is a throwaway: it is consumed by pipeline creation and can be destroyed as soon as
// the pipeline exists.
[[nodiscard]] VkShaderModule load_shader(VkDevice device, std::string_view chapter_id,
                                         std::string_view shader_name);

// Builds a VkGraphicsPipeline without the eleven-struct preamble.
//
// Chapter 1.7 filled in every one of those structs and explained what each field means;
// this class holds exactly the same structs with the same defaults, and lets a chapter
// override only the parts it is actually teaching. Nothing is hidden -- read
// pipeline.cpp and you will find 1.7's code, one member at a time.
//
// The defaults are the ones 1.7 used: triangle list, filled polygons, no culling, no
// depth test, no blending, dynamic viewport and scissor, no multisampling.
class PipelineBuilder {
public:
    explicit PipelineBuilder(VkDevice device) noexcept : device_(device) {}

    // The module holds both stages. The entry point names are the Slang function
    // names -- this project marks them [shader("vertex")] vertexMain and
    // [shader("fragment")] fragmentMain, and the build passes
    // -fvk-use-entrypoint-name so those names survive into the SPIR-V.
    //
    // An empty fragment_entry builds a pipeline with no fragment stage. That is legal
    // in any pipeline, but colour outputs are then undefined, so it is only useful
    // when depth is all the pipeline writes. Chapter 5.3's shadow map is the first.
    PipelineBuilder& shaders(VkShaderModule module,
                             std::string vertex_entry = "vertexMain",
                             std::string fragment_entry = "fragmentMain");

    // Describes the vertex buffer layout. Leave unset for shaders that generate their
    // own vertices from the vertex index, as 1.7 did.
    PipelineBuilder& vertex_input(
        std::span<const VkVertexInputBindingDescription> bindings,
        std::span<const VkVertexInputAttributeDescription> attributes);

    PipelineBuilder& topology(VkPrimitiveTopology topology);
    PipelineBuilder& polygon_mode(VkPolygonMode mode);
    PipelineBuilder& cull(VkCullModeFlags mode, VkFrontFace front_face);

    // Enables depth testing against an attachment of the given format. Passing
    // VK_FORMAT_UNDEFINED (the default) means the pipeline renders colour only.
    PipelineBuilder& depth(VkFormat format, bool test, bool write,
                           VkCompareOp compare = VK_COMPARE_OP_LESS);

    // Offsets every depth value this pipeline produces before the depth test compares
    // it, in multiples of the smallest difference the depth buffer can resolve.
    // Negative moves toward the viewer. Chapter 4.1 explains what the two factors do
    // and why a distance in world units would be the wrong unit for the job.
    PipelineBuilder& depth_bias(float constant_factor, float slope_factor);

    // Declares that the render pass instance has a stencil attachment of this format.
    // Required of *every* pipeline drawn into such an instance, including ones that
    // never test the stencil: the pipeline's attachment description has to match the
    // rendering instance's. This does not enable the stencil test.
    PipelineBuilder& stencil_attachment(VkFormat format);

    // Enables the stencil test and sets the op state for front- and back-facing
    // triangles. Chapter 4.1 writes both VkStencilOpState structs out in full and
    // explains every field; this only carries them through.
    PipelineBuilder& stencil_test(VkStencilOpState front, VkStencilOpState back);

    // Dynamic rendering has no render pass to describe the attachments, so the
    // pipeline is told their formats directly. These must match the images the
    // chapter actually renders into. Leaving it unset declares no colour attachment
    // at all, for a depth-only pass like chapter 5.3's shadow map.
    PipelineBuilder& colour_attachment(VkFormat format);

    // The views a multiview rendering pass draws, one bit per layer, which must match
    // VkRenderingInfo::viewMask of every pass the pipeline is used in. 0, the default, is
    // an ordinary pass. Chapter 5.4 is the first to set it, for the six faces of a cube.
    PipelineBuilder& view_mask(uint32_t mask);

    // Standard source-over alpha blending. Off by default: an opaque fragment shader
    // output simply replaces whatever was in the attachment.
    PipelineBuilder& alpha_blending(bool enabled);

    // The whole blend attachment state, for chapters that need a mode alpha_blending()
    // does not cover -- additive, premultiplied, a write mask that drops a channel.
    // Chapter 4.2 writes the struct out in full and explains every field; setting this
    // overrides alpha_blending().
    PipelineBuilder& colour_blend(const VkPipelineColorBlendAttachmentState& state);

    // Bakes constant values into the fragment shader at pipeline-creation time.
    // Chapter 1.9 builds the entries and the data block; this only carries them
    // through to VkPipelineShaderStageCreateInfo::pSpecializationInfo.
    // Samples per pixel, which must match the attachments the pipeline renders into, and
    // the fraction of them the fragment shader runs for. 0 leaves sample shading off:
    // the shader runs once per pixel however many samples there are. Chapter 4.7
    // explains both.
    PipelineBuilder& multisample(VkSampleCountFlagBits samples,
                                 float min_sample_shading = 0.0F);

    PipelineBuilder& fragment_specialisation(
        std::span<const VkSpecializationMapEntry> entries,
        std::span<const std::byte> data);

    PipelineBuilder& layout(VkPipelineLayout layout);

    [[nodiscard]] VkPipeline build() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;

    VkShaderModule module_ = VK_NULL_HANDLE;
    std::string vertex_entry_ = "vertexMain";
    std::string fragment_entry_ = "fragmentMain";

    std::vector<VkVertexInputBindingDescription> bindings_;
    std::vector<VkVertexInputAttributeDescription> attributes_;

    VkPrimitiveTopology topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode_ = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode_ = VK_CULL_MODE_NONE;
    VkFrontFace front_face_ = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkFormat colour_format_ = VK_FORMAT_UNDEFINED;
    uint32_t view_mask_ = 0;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    bool depth_test_ = false;
    bool depth_write_ = false;
    VkCompareOp depth_compare_ = VK_COMPARE_OP_LESS;

    bool depth_bias_ = false;
    float depth_bias_constant_ = 0.0F;
    float depth_bias_slope_ = 0.0F;

    VkFormat stencil_format_ = VK_FORMAT_UNDEFINED;
    bool stencil_test_ = false;
    VkStencilOpState stencil_front_{};
    VkStencilOpState stencil_back_{};

    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    float min_sample_shading_ = 0.0F;

    bool blending_ = false;
    bool custom_blend_ = false;
    VkPipelineColorBlendAttachmentState blend_state_{};

    std::vector<VkSpecializationMapEntry> specialisation_entries_;
    std::vector<std::byte> specialisation_data_;

    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

}  // namespace vkc
