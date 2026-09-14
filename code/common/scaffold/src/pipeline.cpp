#include "vkc/pipeline.hpp"

#include "vkc/check.hpp"
#include "vkc/paths.hpp"

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstddef>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkc {
namespace {

// SPIR-V 1.6 is the version Vulkan 1.3 consumes, and 1.3 is this project's baseline.
constexpr const char* kSpirvProfile = "spirv_1_6";

// Slang reports errors and warnings through a blob rather than a return code, so every
// call that can produce diagnostics gets checked through here.
[[nodiscard]] std::string blob_text(slang::IBlob* blob) {
    if (blob == nullptr || blob->getBufferSize() == 0) {
        return {};
    }
    return std::string(static_cast<const char*>(blob->getBufferPointer()),
                       blob->getBufferSize());
}

// One global session for the life of the process.
//
// Creating it is the expensive part of using Slang -- it loads the standard module and
// sets up the compiler -- and it is designed to be created once and shared. Everything
// after it is cheap enough to do per shader.
[[nodiscard]] slang::IGlobalSession& global_session() {
    static Slang::ComPtr<slang::IGlobalSession> session = [] {
        Slang::ComPtr<slang::IGlobalSession> created;
        if (SLANG_FAILED(slang::createGlobalSession(created.writeRef()))) {
            throw std::runtime_error("slang::createGlobalSession failed");
        }
        return created;
    }();
    return *session;
}

}  // namespace

VkShaderModule load_shader(VkDevice device, std::string_view chapter_id,
                           std::string_view shader_name) {
    // Chapter 1.7 wrote all of this out by hand and explained it line by line; this is
    // that code with the error handling filled in and the global session cached.
    slang::IGlobalSession& global = global_session();

    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.profile = global.findProfile(kSpirvProfile);
    if (target.profile == SLANG_PROFILE_UNKNOWN) {
        throw std::runtime_error(
            std::format("Slang does not know the profile '{}'", kSpirvProfile));
    }

    const slang::CompilerOptionEntry options[]{
        // Use Slang's own SPIR-V backend rather than routing through generated GLSL.
        {slang::CompilerOptionName::EmitSpirvDirectly,
         {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
        // Keep the SPIR-V entry points named after the Slang functions. Without
        // this, a module with one entry point gets it renamed to "main" and the
        // names stop matching what pName asks for.
        {slang::CompilerOptionName::VulkanUseEntryPointName,
         {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
        // Debug info, so RenderDoc can show the original Slang beside the disassembly.
        {slang::CompilerOptionName::DebugInformation,
         {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD, 0,
          nullptr, nullptr}},
    };

    const std::string search_path = shader_dir(chapter_id).string();
    const char* search_paths[]{search_path.c_str()};

    slang::SessionDesc session_desc{};
    session_desc.targets = &target;
    session_desc.targetCount = 1;
    session_desc.searchPaths = search_paths;
    session_desc.searchPathCount = 1;
    session_desc.compilerOptionEntries = options;
    session_desc.compilerOptionEntryCount =
        static_cast<uint32_t>(std::size(options));

    // Matrix layout, and this is the one that will bite you.
    //
    // The session default is SLANG_MATRIX_LAYOUT_ROW_MAJOR, which does not match glm --
    // a glm::mat4 is sixteen floats stored column by column, and it is memcpy'd into a
    // uniform buffer or a push constant with no transpose on the way. Leave the default
    // and every transform comes out transposed: the image still draws, so nothing
    // errors, it is just wrong.
    //
    // Setting COLUMN_MAJOR makes Slang emit a RowMajor decoration in the SPIR-V, which
    // looks like the opposite of what was asked for and is not. Slang pairs that
    // decoration with OpVectorTimesMatrix, and storing transposed while multiplying on
    // the other side is the same arithmetic as storing plainly and multiplying
    // normally. The two conventions cancel. What matters is that this line makes mul(M,
    // v) mean "apply M to v" for a matrix whose bytes came from glm.
    //
    // slangc's command line defaults to this; the API does not.
    session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(global.createSession(session_desc, session.writeRef()))) {
        throw std::runtime_error("slang::IGlobalSession::createSession failed");
    }

    // Slang addresses a file by module name, without the extension: "quad.slang" on
    // disk is the module "quad", found by walking the search paths above.
    std::string module_name(shader_name);
    if (module_name.ends_with(".slang")) {
        module_name.resize(module_name.size() - 6);
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module =
        session->loadModule(module_name.c_str(), diagnostics.writeRef());
    if (module == nullptr) {
        throw std::runtime_error(std::format(
            "Could not compile shader module '{}' from '{}'.\n{}", module_name,
            search_path, blob_text(diagnostics)));
    }

    // Every [shader("...")] function in the file becomes an entry point of the SPIR-V
    // module. Composing them with the module itself and linking is what produces one
    // binary holding the whole pipeline.
    std::vector<slang::IComponentType*> components{module};
    const SlangInt entry_point_count = module->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entry_points(
        static_cast<size_t>(entry_point_count));
    for (SlangInt i = 0; i < entry_point_count; ++i) {
        const size_t index = static_cast<size_t>(i);
        if (SLANG_FAILED(
                module->getDefinedEntryPoint(i, entry_points[index].writeRef()))) {
            throw std::runtime_error(
                std::format("Could not read entry point {} of '{}'", i, module_name));
        }
        components.push_back(entry_points[index]);
    }
    if (entry_point_count == 0) {
        throw std::runtime_error(std::format(
            "'{}' declares no entry points. Mark one with [shader(\"vertex\")] or "
            "[shader(\"fragment\")].",
            module_name));
    }

    Slang::ComPtr<slang::IComponentType> composed;
    diagnostics = nullptr;
    if (SLANG_FAILED(session->createCompositeComponentType(
            components.data(), static_cast<SlangInt>(components.size()),
            composed.writeRef(), diagnostics.writeRef()))) {
        throw std::runtime_error(std::format("Composing '{}' failed.\n{}", module_name,
                                             blob_text(diagnostics)));
    }

    Slang::ComPtr<slang::IComponentType> linked;
    diagnostics = nullptr;
    if (SLANG_FAILED(composed->link(linked.writeRef(), diagnostics.writeRef()))) {
        throw std::runtime_error(std::format("Linking '{}' failed.\n{}", module_name,
                                             blob_text(diagnostics)));
    }

    Slang::ComPtr<slang::IBlob> spirv;
    diagnostics = nullptr;
    if (SLANG_FAILED(
            linked->getTargetCode(0, spirv.writeRef(), diagnostics.writeRef()))) {
        throw std::runtime_error(std::format("Generating SPIR-V for '{}' failed.\n{}",
                                             module_name, blob_text(diagnostics)));
    }

    // codeSize is in bytes; pCode is a uint32_t*. Slang's blob is already aligned for
    // it, which is the one thing worth checking if you ever swap the source of the
    // bytes.
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv->getBufferSize(),
        .pCode = static_cast<const uint32_t*>(spirv->getBufferPointer()),
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &shader_module));
    return shader_module;
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
