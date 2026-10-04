// 1.7 Hello Triangle
//
// Adds to 1.6: shader modules, a pipeline layout, a graphics pipeline, and a draw
// call. At the end of this chapter there is finally a triangle on the screen.
//
// The new idea is the *pipeline object*. In OpenGL you set state one call at a time --
// bind a shader, enable blending, set a depth function -- and the driver works out
// what that combination means at draw time, every draw. Vulkan makes you declare the
// entire configuration up front and compiles it into a VkPipeline. Shaders, blending,
// depth testing, the primitive topology and the vertex layout are all baked in
// together.
//
// That is why pipeline creation is such a large function, and why it is a one-off
// cost: the expensive work happens here rather than inside the frame loop.
//
// Almost all of it is fixed at creation. The two things deliberately left dynamic are
// the viewport and the scissor rectangle, so that resizing the window does not mean
// rebuilding the pipeline.
//
// The frame loop is no longer written here. Chapters 1.5 and 1.6 built it by hand and
// explained every fence and semaphore in it; it is now vkc::FrameContext, which hands
// out one open command buffer per frame and takes care of acquire, submit and present.

#include <vkc/capture.hpp>
#include <vkc/check.hpp>
#include <vkc/context.hpp>
#include <vkc/frame.hpp>
#include <vkc/paths.hpp>
#include <vkc/swapchain.hpp>
#include <vkc/window.hpp>

#include <slang-com-ptr.h>
#include <slang.h>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TriangleApp {
public:
    explicit TriangleApp(const vkc::Args& args)
        : args_(args),
          window_("LearnVulkan - Hello Triangle", args.width, args.height,
                  /*resizable=*/args.screenshot.empty()),
          context_(vkc::Context::Config{
                       .app_name = "LearnVulkan",
                       .enable_validation = args.validation,
                   },
                   window_),
          swapchain_(context_, window_),
          frames_(context_, swapchain_) {
        create_pipeline();
        capture_.init(context_.device(), context_.allocator());
    }

    ~TriangleApp() {
        context_.wait_idle();

        capture_.destroy();
        vkDestroyPipeline(context_.device(), pipeline_, nullptr);
        vkDestroyPipelineLayout(context_.device(), pipeline_layout_, nullptr);
    }

    TriangleApp(const TriangleApp&) = delete;
    TriangleApp& operator=(const TriangleApp&) = delete;
    TriangleApp(TriangleApp&&) = delete;
    TriangleApp& operator=(TriangleApp&&) = delete;

    int run() {
        spdlog::info("Drawing a triangle. Escape or close to quit.");

        uint64_t frames_rendered = 0;
        bool running = true;
        bool swapchain_dirty = false;

        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT ||
                    event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED ||
                    (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)) {
                    running = false;
                } else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                    swapchain_dirty = true;
                }
            }

            if (window_.is_minimised()) {
                SDL_WaitEvent(nullptr);
                continue;
            }

            if (swapchain_dirty) {
                swapchain_.recreate();
                frames_.on_swapchain_recreated();
                swapchain_dirty = false;
            }

            // begin() waits for this frame's slot, acquires an image and opens a
            // command buffer. It returns nothing when the swapchain went out of
            // date, in which case the frame is skipped rather than drawn wrong.
            std::optional<vkc::FrameInfo> frame = frames_.begin();
            if (!frame.has_value()) {
                swapchain_dirty = true;
                continue;
            }

            const bool capture_this_frame =
                !args_.screenshot.empty() && args_.frame_limit != 0 &&
                frames_rendered + 1 == args_.frame_limit;

            record(*frame, capture_this_frame);

            if (!frames_.end(*frame)) {
                swapchain_dirty = true;
            }

            ++frames_rendered;
            if (args_.frame_limit != 0 && frames_rendered >= args_.frame_limit) {
                running = false;
            }
        }

        context_.wait_idle();
        if (capture_.has_pending_capture()) {
            capture_.write(args_.screenshot, swapchain_.format());
        }
        return 0;
    }

private:
    void record(const vkc::FrameInfo& frame, bool capture_this_frame) {
        // Swapchain images come back in UNDEFINED layout and must reach
        // COLOR_ATTACHMENT_OPTIMAL before anything renders into them. Chapter 1.5
        // wrote this barrier out in full; it is vkc::transition_image now.
        vkc::transition_image(frame.cmd, frame.image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // A flat dark background now, rather than the cycling colour of 1.5 and 1.6.
        // The triangle is the subject of this chapter and a moving background would
        // only fight with it.
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

        // The two pieces of state left dynamic. The viewport maps clip space onto the
        // image; the scissor throws away anything outside a rectangle.
        //
        // Note minDepth/maxDepth of 0..1: that is Vulkan's depth range, and it is why
        // the project defines GLM_FORCE_DEPTH_ZERO_TO_ONE. OpenGL's is -1..1.
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

        // Three vertices, one instance. The vertex shader runs three times, with
        // SV_VulkanVertexID counting 0, 1, 2.
        vkCmdDraw(frame.cmd, 3, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);

        // The presentation engine will only accept an image in PRESENT_SRC_KHR. The
        // capture helper records its copy and performs that transition itself, so the
        // image still gets presented either way.
        if (capture_this_frame) {
            capture_.record(frame.cmd, frame.image, frame.extent);
        } else {
            vkc::transition_image(frame.cmd, frame.image,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
    }

    // Slang reports what went wrong through a blob rather than a return code, so the
    // calls below all pass one in and this turns it into something printable.
    [[nodiscard]] static std::string blob_text(slang::IBlob* blob) {
        if (blob == nullptr || blob->getBufferSize() == 0) {
            return {};
        }
        return std::string(static_cast<const char*>(blob->getBufferPointer()),
                           blob->getBufferSize());
    }

    // Compiles a .slang file to SPIR-V and hands the result to the driver.
    //
    // Vulkan will not take shader source -- it takes SPIR-V -- so something has to do
    // the translation. This project links libslang and does it here, when the program
    // starts, which means the .slang file beside the binary is the shader: edit it, run
    // again, see the change, with no build step in between.
    //
    // Everything from createGlobalSession down to getTargetCode is Slang's compiler
    // API. The last six lines are the Vulkan part, and they are the same six lines they
    // would be if the bytes had come from a .spv file on disk.
    [[nodiscard]] VkShaderModule load_shader(const char* name) const {
        // The global session owns the compiler and its standard library. It is the
        // expensive object -- create it once and share it. One shader here, so one
        // session; vkcommon caches it from 1.8 onward.
        Slang::ComPtr<slang::IGlobalSession> global;
        if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
            throw std::runtime_error("slang::createGlobalSession failed");
        }

        // What to generate: SPIR-V 1.6, which is the version Vulkan 1.3 consumes.
        slang::TargetDesc target{};
        target.format = SLANG_SPIRV;
        target.profile = global->findProfile("spirv_1_6");

        const slang::CompilerOptionEntry options[]{
            // Slang's own SPIR-V backend, rather than routing through generated GLSL.
            {slang::CompilerOptionName::EmitSpirvDirectly,
             {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
            // Keep the entry points named after the Slang functions. Without this, a
            // module gets its entry point renamed to "main" and the names stop matching
            // what VkPipelineShaderStageCreateInfo::pName asks for below.
            {slang::CompilerOptionName::VulkanUseEntryPointName,
             {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
            // Debug info, so RenderDoc can show the Slang source beside the SPIR-V.
            {slang::CompilerOptionName::DebugInformation,
             {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD, 0,
              nullptr, nullptr}},
        };

        // Where to look for the file. The build staged it next to the binary.
        const std::string search_path = vkc::shader_dir(LVK_CHAPTER_ID).string();
        const char* search_paths[]{search_path.c_str()};

        slang::SessionDesc session_desc{};
        session_desc.targets = &target;
        session_desc.targetCount = 1;
        session_desc.searchPaths = search_paths;
        session_desc.searchPathCount = 1;
        session_desc.compilerOptionEntries = options;
        session_desc.compilerOptionEntryCount = 3;

        // Matrix layout, and this is the one that will bite you.
        //
        // The session default is SLANG_MATRIX_LAYOUT_ROW_MAJOR, which does not match
        // glm -- a glm::mat4 is sixteen floats stored column by column, and it is
        // memcpy'd into a uniform buffer or a push constant with no transpose on the
        // way. Leave the default and every transform comes out transposed: the image
        // still draws, so nothing errors, it is just wrong.
        //
        // Setting COLUMN_MAJOR makes Slang emit a RowMajor decoration in the SPIR-V,
        // which looks like the opposite of what was asked for and is not. Slang pairs
        // that decoration with OpVectorTimesMatrix, and storing transposed while
        // multiplying on the other side is the same arithmetic as storing plainly and
        // multiplying normally. The two conventions cancel. What matters is that this
        // line makes mul(M, v) mean "apply M to v" for a matrix whose bytes came from
        // glm.
        //
        // slangc's command line defaults to this; the API does not.
        session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

        Slang::ComPtr<slang::ISession> session;
        if (SLANG_FAILED(global->createSession(session_desc, session.writeRef()))) {
            throw std::runtime_error("slang::IGlobalSession::createSession failed");
        }

        // Slang names a file by module, without the extension: triangle.slang on disk
        // is the module "triangle", found by walking the search paths.
        std::string module_name(name);
        if (module_name.ends_with(".slang")) {
            module_name.resize(module_name.size() - 6);
        }

        Slang::ComPtr<slang::IBlob> diagnostics;
        slang::IModule* module =
            session->loadModule(module_name.c_str(), diagnostics.writeRef());
        if (module == nullptr) {
            throw std::runtime_error(
                std::format("Could not compile '{}' from '{}'.\n{}", module_name,
                            search_path, blob_text(diagnostics)));
        }

        // Each [shader("...")] function in the file is an entry point. Composing them
        // with the module and linking is what produces one SPIR-V binary holding the
        // whole pipeline -- both stages, one module.
        std::vector<slang::IComponentType*> components{module};
        const SlangInt32 entry_point_count = module->getDefinedEntryPointCount();
        std::vector<Slang::ComPtr<slang::IEntryPoint>> entry_points(
            static_cast<size_t>(entry_point_count));
        for (SlangInt32 i = 0; i < entry_point_count; ++i) {
            const auto index = static_cast<size_t>(i);
            if (SLANG_FAILED(
                    module->getDefinedEntryPoint(i, entry_points[index].writeRef()))) {
                throw std::runtime_error("slang::IModule::getDefinedEntryPoint failed");
            }
            components.push_back(entry_points[index]);
        }

        Slang::ComPtr<slang::IComponentType> composed;
        diagnostics = nullptr;
        if (SLANG_FAILED(session->createCompositeComponentType(
                components.data(), static_cast<SlangInt>(components.size()),
                composed.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Composing '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        Slang::ComPtr<slang::IComponentType> linked;
        diagnostics = nullptr;
        if (SLANG_FAILED(composed->link(linked.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Linking '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        Slang::ComPtr<slang::IBlob> spirv;
        diagnostics = nullptr;
        if (SLANG_FAILED(
                linked->getTargetCode(0, spirv.writeRef(), diagnostics.writeRef()))) {
            throw std::runtime_error(std::format("Generating SPIR-V for '{}' failed.\n{}",
                                                 module_name, blob_text(diagnostics)));
        }

        // From here it is ordinary Vulkan. A shader module is just a SPIR-V blob handed
        // to the driver; it is not compiled to machine code here -- that happens when
        // the pipeline is created, which is when the driver finally knows the rest of
        // the state the shader runs under.
        //
        // codeSize is in bytes while pCode is a uint32_t*, because SPIR-V is a stream of
        // 32-bit words and must be 4-byte aligned. Slang's blob already is.
        const VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = spirv->getBufferSize(),
            .pCode = static_cast<const uint32_t*>(spirv->getBufferPointer()),
        };
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(context_.device(), &info, nullptr,
                                      &shader_module));
        return shader_module;
    }

    void create_pipeline() {
        const VkDevice device = context_.device();

        // One module for both stages. triangle.slang declares a vertexMain and a
        // fragmentMain, and the compile above produced a single SPIR-V binary holding
        // both of them.
        const VkShaderModule shader = load_shader("triangle.slang");

        const VkPipelineShaderStageCreateInfo stages[2]{
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = shader,
                // The entry point, by name. A SPIR-V module can hold as many as you
                // like, and pName is what selects one -- which is why the same module
                // appears twice here with a different name each time.
                .pName = "vertexMain",
                .pSpecializationInfo = nullptr,
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = shader,
                .pName = "fragmentMain",
                .pSpecializationInfo = nullptr,
            },
        };

        // No vertex buffers: the vertex shader generates its own positions from
        // the vertex index. Chapter 1.8 fills this struct in properly.
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
        };

        // Every three vertices form one triangle.
        const VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        // The counts matter even though the pointers are null: the pipeline needs to
        // know there is one of each, and the values arrive per-frame via
        // vkCmdSetViewport and vkCmdSetScissor.
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
            // VK_TRUE here would throw away all geometry before rasterisation, which
            // is occasionally useful and always confusing to leave on by accident.
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            // No culling yet. Face culling gets its own chapter, and turning it on now
            // would silently discard the triangle if its winding were the other way.
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0F,
            .depthBiasClamp = 0.0F,
            .depthBiasSlopeFactor = 0.0F,
            .lineWidth = 1.0F,
        };

        // One sample per pixel: no multisampling. The anti-aliasing chapter revisits
        // this, and it is one of the fields that must match the attachments exactly.
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

        // blendEnable VK_FALSE means the fragment shader's output replaces whatever
        // is in the attachment. colorWriteMask still has to list the channels to
        // write; leaving it at zero writes nothing at all, for a black screen and no
        // error message.
        const VkPipelineColorBlendAttachmentState blend_attachment{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
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

        const VkDynamicState dynamic_states[2]{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        const VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamic_states,
        };

        // The layout declares what external data the shaders can see: descriptor sets
        // and push constants. This shader reads nothing, so the layout is empty -- but
        // it still has to exist.
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr,
                                        &pipeline_layout_));

        // With dynamic rendering there is no VkRenderPass to be compatible with, so
        // the pipeline is told the attachment formats directly. These must match the
        // VkRenderingInfo used at draw time, or nothing will be drawn.
        const VkFormat colour_format = swapchain_.format();
        const VkPipelineRenderingCreateInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colour_format,
            .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
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
            .pDepthStencilState = nullptr,  // no depth attachment in this chapter
            .pColorBlendState = &blend,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout_,
            // Null, because dynamic rendering replaced them.
            .renderPass = VK_NULL_HANDLE,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                           nullptr, &pipeline_));

        // The module has been compiled into the pipeline and is no longer needed.
        vkDestroyShaderModule(device, shader, nullptr);
    }

    vkc::Args args_;
    vkc::Window window_;
    vkc::Context context_;
    vkc::Swapchain swapchain_;
    vkc::FrameContext frames_;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    vkc::Capture capture_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        TriangleApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
