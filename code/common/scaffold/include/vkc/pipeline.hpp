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

    // Dynamic rendering has no render pass to describe the attachments, so the
    // pipeline is told their formats directly. These must match the images the
    // chapter actually renders into.
    PipelineBuilder& colour_attachment(VkFormat format);

    // Standard source-over alpha blending. Off by default: an opaque fragment shader
    // output simply replaces whatever was in the attachment.
    PipelineBuilder& alpha_blending(bool enabled);

    // Bakes constant values into the fragment shader at pipeline-creation time.
    // Chapter 1.9 builds the entries and the data block; this only carries them
    // through to VkPipelineShaderStageCreateInfo::pSpecializationInfo.
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
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    bool depth_test_ = false;
    bool depth_write_ = false;
    VkCompareOp depth_compare_ = VK_COMPARE_OP_LESS;

    bool blending_ = false;

    std::vector<VkSpecializationMapEntry> specialisation_entries_;
    std::vector<std::byte> specialisation_data_;

    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

}  // namespace vkc
