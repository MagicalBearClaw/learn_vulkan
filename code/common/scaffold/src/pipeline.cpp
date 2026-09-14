#include "vkc/pipeline.hpp"

#include "vkc/check.hpp"
#include "vkc/paths.hpp"

#include <stdexcept>
#include <utility>

namespace vkc {

VkShaderModule load_shader(VkDevice device, std::string_view chapter_id,
                           std::string_view shader_name) {
    const std::vector<uint32_t> code =
        read_spirv(shader_path(chapter_id, shader_name));

    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

PipelineBuilder& PipelineBuilder::shaders(VkShaderModule module,
                                          std::string vertex_entry,
                                          std::string fragment_entry) {
    module_ = module;
    vertex_entry_ = std::move(vertex_entry);
    fragment_entry_ = std::move(fragment_entry);
    return *this;
}

PipelineBuilder& PipelineBuilder::vertex_input(
    std::span<const VkVertexInputBindingDescription> bindings,
    std::span<const VkVertexInputAttributeDescription> attributes) {
    // Copied rather than referenced: a builder is usually filled in from temporaries
    // and then built a few lines later, and dangling pointers into a dead array are
    // exactly the kind of bug that shows up as a driver crash with no message.
    bindings_.assign(bindings.begin(), bindings.end());
    attributes_.assign(attributes.begin(), attributes.end());
    return *this;
}

PipelineBuilder& PipelineBuilder::topology(VkPrimitiveTopology topology) {
    topology_ = topology;
    return *this;
}

PipelineBuilder& PipelineBuilder::polygon_mode(VkPolygonMode mode) {
    polygon_mode_ = mode;
    return *this;
}

PipelineBuilder& PipelineBuilder::cull(VkCullModeFlags mode, VkFrontFace front_face) {
    cull_mode_ = mode;
    front_face_ = front_face;
    return *this;
}

PipelineBuilder& PipelineBuilder::depth(VkFormat format, bool test, bool write,
                                        VkCompareOp compare) {
    depth_format_ = format;
    depth_test_ = test;
    depth_write_ = write;
    depth_compare_ = compare;
    return *this;
}

PipelineBuilder& PipelineBuilder::colour_attachment(VkFormat format) {
    colour_format_ = format;
    return *this;
}

PipelineBuilder& PipelineBuilder::alpha_blending(bool enabled) {
    blending_ = enabled;
    return *this;
}

PipelineBuilder& PipelineBuilder::fragment_specialisation(
    std::span<const VkSpecializationMapEntry> entries,
    std::span<const std::byte> data) {
    specialisation_entries_.assign(entries.begin(), entries.end());
    specialisation_data_.assign(data.begin(), data.end());
    return *this;
}

PipelineBuilder& PipelineBuilder::layout(VkPipelineLayout layout) {
    layout_ = layout;
    return *this;
}

VkPipeline PipelineBuilder::build() const {
    if (module_ == VK_NULL_HANDLE) {
        throw std::runtime_error("PipelineBuilder::build: shaders() was never called");
    }
    if (layout_ == VK_NULL_HANDLE) {
        throw std::runtime_error("PipelineBuilder::build: layout() was never called");
    }
    if (colour_format_ == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error(
            "PipelineBuilder::build: colour_attachment() was never called");
    }

    const VkSpecializationInfo specialisation{
        .mapEntryCount = static_cast<uint32_t>(specialisation_entries_.size()),
        .pMapEntries = specialisation_entries_.data(),
        .dataSize = specialisation_data_.size(),
        .pData = specialisation_data_.data(),
    };
    const VkSpecializationInfo* fragment_specialisation =
        specialisation_entries_.empty() ? nullptr : &specialisation;

    const VkPipelineShaderStageCreateInfo stages[2]{
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            // Both stages come from the same module; pName is what picks one of the
            // two entry points out of it.
            .module = module_,
            .pName = vertex_entry_.c_str(),
            .pSpecializationInfo = nullptr,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = module_,
            .pName = fragment_entry_.c_str(),
            .pSpecializationInfo = fragment_specialisation,
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<uint32_t>(bindings_.size()),
        .pVertexBindingDescriptions = bindings_.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes_.size()),
        .pVertexAttributeDescriptions = attributes_.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = topology_,
        .primitiveRestartEnable = VK_FALSE,
    };

    // The counts matter even though the pointers are null: the pipeline needs to know
    // there is one viewport and one scissor, and the values arrive at record time.
    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };

    const VkPipelineRasterizationStateCreateInfo rasterisation{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = polygon_mode_,
        .cullMode = cull_mode_,
        .frontFace = front_face_,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0F,
        .depthBiasClamp = 0.0F,
        .depthBiasSlopeFactor = 0.0F,
        .lineWidth = 1.0F,
    };

    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0F,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = depth_test_ ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = depth_write_ ? VK_TRUE : VK_FALSE,
        .depthCompareOp = depth_compare_,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0F,
        .maxDepthBounds = 1.0F,
    };

    const VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = blending_ ? VK_TRUE : VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
        .blendConstants = {0.0F, 0.0F, 0.0F, 0.0F},
    };

    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkPipelineRenderingCreateInfo rendering_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colour_format_,
        .depthAttachmentFormat = depth_format_,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering_info,
        .flags = 0,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterisation,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic_state,
        .layout = layout_,
        // No render pass: that is what VkPipelineRenderingCreateInfo replaces.
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                       nullptr, &pipeline));
    return pipeline;
}

}  // namespace vkc
