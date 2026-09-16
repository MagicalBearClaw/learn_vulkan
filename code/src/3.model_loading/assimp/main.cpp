// 3.1 Assimp
//
// The first chapter whose geometry does not come from a C++ array.
//
// Everything drawn so far has been a cube written out as 24 vertices in this file. That
// stops here: the vertices now arrive from a file on disk, in a format nobody at this
// project chose, describing a shape nobody here modelled. Assimp is what reads it.
//
// The Vulkan lesson is small, and that is the point -- a vertex buffer does not care
// where its bytes came from. What changes is everything around it: the index type grows
// to 32 bits, the vertex count stops being a constant, and the program no longer knows
// how big the thing it is drawing is until it has read it.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The same layout Part 2 used. `uv` is filled in by the loader and nothing reads it yet
// -- 3.2 is where a material appears -- so the pipeline below declares two attributes
// while the vertex carries three fields.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// What one call to Assimp gives this chapter: one vertex buffer, one index buffer, and
// the box the result fits in.
struct ModelData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::vec3 min{};
    glm::vec3 max{};

    [[nodiscard]] glm::vec3 centre() const { return (min + max) * 0.5F; }
    [[nodiscard]] float radius() const { return glm::length(max - min) * 0.5F; }
};

// Post-processing steps. Assimp calls these flags, but each one is a real pass over the
// data, and choosing them is most of what using the library well consists of.
//
//   Triangulate           every face becomes a triangle. Vulkan draws triangles; a
//                         quad or an n-gon in the file is not something the pipeline
//                         can consume.
//   GenSmoothNormals      generate normals if the file has none, averaged across the
//                         faces meeting at each vertex. A no-op when they are present.
//   JoinIdenticalVertices deduplicate, so a shared corner is one vertex referenced many
//                         times rather than many copies. This is what makes the index
//                         buffer worth having.
//   PreTransformVertices  collapse the node hierarchy: bake each node's transform into
//                         its vertices and hand back meshes already in model space.
//                         This one is a simplification the chapter is explicit about,
//                         and 3.3 removes it.
//   ImproveCacheLocality  reorder triangles so nearby ones share vertices, which the
//                         GPU's post-transform cache rewards.
//
// Not used, although most OpenGL material passes it: aiProcess_FlipUVs. glTF puts (0,0)
// at the top-left of the image and so does Vulkan, so flipping here would be flipping
// twice. It costs nothing until 3.2 samples a texture, and then it costs an afternoon.
constexpr unsigned int kImportFlags =
    aiProcess_Triangulate | aiProcess_GenSmoothNormals |
    aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices |
    aiProcess_ImproveCacheLocality;

[[nodiscard]] glm::vec3 to_glm(const aiVector3D& v) { return {v.x, v.y, v.z}; }

// Reads the file and flattens every mesh in it into one pair of buffers.
//
// Concatenating the meshes is the other simplification here. It works because this
// chapter draws the whole model with one material and one draw call; the moment each
// mesh needs its own texture -- which is 3.2 -- they have to stay apart.
[[nodiscard]] ModelData load_model(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Model not found: " + path.string() +
            "\nFetch the tutorial assets first:  python3 tools/bootstrap.py");
    }

    // The Importer owns the aiScene. When it goes out of scope the scene is freed, which
    // is why everything needed is copied out before this function returns.
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path.string(), kImportFlags);

    // Three failures, not one: a null scene, a scene Assimp only partly understood, and
    // a scene with no root node. The middle case is the dangerous one -- it returns
    // something that looks usable and is missing data.
    if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0 ||
        scene->mRootNode == nullptr) {
        throw std::runtime_error("Assimp could not read " + path.string() + ": " +
                                 importer.GetErrorString());
    }

    spdlog::info("{}: {} mesh(es), {} material(s), {} embedded texture(s)",
                 path.filename().string(), scene->mNumMeshes, scene->mNumMaterials,
                 scene->mNumTextures);

    ModelData model;
    model.min = glm::vec3(std::numeric_limits<float>::max());
    model.max = glm::vec3(std::numeric_limits<float>::lowest());

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        // Indices are per-mesh, and the buffers are shared, so every index from this
        // mesh shifts up by however many vertices are already in.
        const auto base = static_cast<uint32_t>(model.vertices.size());

        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            Vertex vertex{};
            vertex.position = to_glm(mesh->mVertices[v]);
            vertex.normal = mesh->HasNormals() ? to_glm(mesh->mNormals[v])
                                               : glm::vec3(0.0F, 1.0F, 0.0F);
            // A mesh can carry up to eight UV sets. Set 0 is the one materials use by
            // default, and it is the only one anything in this series reads.
            vertex.uv = mesh->HasTextureCoords(0)
                            ? glm::vec2(mesh->mTextureCoords[0][v].x,
                                        mesh->mTextureCoords[0][v].y)
                            : glm::vec2(0.0F);

            model.min = glm::min(model.min, vertex.position);
            model.max = glm::max(model.max, vertex.position);
            model.vertices.push_back(vertex);
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            // aiProcess_Triangulate guarantees three, but reading the count rather than
            // assuming it is the difference between a loader and a loader for one file.
            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                model.indices.push_back(base + face.mIndices[i]);
            }
        }

        spdlog::info("  mesh {} \"{}\": {} vertices, {} faces, material {}", m,
                     mesh->mName.C_Str(), mesh->mNumVertices, mesh->mNumFaces,
                     mesh->mMaterialIndex);
    }

    if (model.vertices.empty()) {
        throw std::runtime_error("Model has no vertices: " + path.string());
    }

    spdlog::info("  total: {} vertices, {} indices, bounds ({:.2f} {:.2f} {:.2f}) to "
                 "({:.2f} {:.2f} {:.2f})",
                 model.vertices.size(), model.indices.size(), model.min.x, model.min.y,
                 model.min.z, model.max.x, model.max.y, model.max.z);
    return model;
}

// 2.6's block, with the light array replaced by two directional lights. `ambient.w` is
// the shininess, riding in padding that exists whether or not anything uses it.
struct Globals {
    glm::mat4 view;                // 0
    glm::mat4 projection;          // 64
    glm::vec4 view_position;       // 128
    glm::vec4 key_direction;       // 144
    glm::vec4 key_colour;          // 160
    glm::vec4 fill_direction;      // 176
    glm::vec4 fill_colour;         // 192
    glm::vec4 ambient;             // 208, w is the shininess
};
static_assert(offsetof(Globals, ambient) == 208);
static_assert(sizeof(Globals) == 224);

// Unchanged from 2.6, down to the byte.
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

constexpr float kShininess = 48.0F;

// A cool key from the front left and a warm fill from behind right, so that the side the
// key misses is described rather than black. Both are directional: the model is one
// object with nothing around it, and distance falloff would say nothing here.
constexpr glm::vec3 kKeyDirection{-0.5F, -0.8F, -0.6F};
constexpr glm::vec3 kKeyColour{0.95F, 0.96F, 1.00F};
constexpr glm::vec3 kFillDirection{0.7F, -0.2F, 0.7F};
constexpr glm::vec3 kFillColour{0.35F, 0.28F, 0.22F};
constexpr glm::vec3 kAmbient{0.06F, 0.06F, 0.08F};

// Flat mid-grey. There is a full set of PBR textures next to this model in the file and
// this chapter reads none of them -- the point is the geometry, and an untextured surface
// is where you can actually see it.
constexpr glm::vec4 kAlbedo{0.62F, 0.62F, 0.64F, 1.0F};

class AssimpApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        const ModelData model = load_model(vkc::asset_path("models/damaged_helmet/DamagedHelmet.glb"));

        index_count_ = static_cast<uint32_t>(model.indices.size());

        // The same upload path as every chapter since 1.8. A model is bigger than a cube
        // and otherwise no different: staging buffer, copy, device-local memory.
        vertex_buffer_ = vkc::upload_to_device_local(
            context(), model.vertices.data(), model.vertices.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer_ = vkc::upload_to_device_local(
            context(), model.indices.data(), model.indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        context().name(vertex_buffer_.handle(), VK_OBJECT_TYPE_BUFFER, "helmet vertices");
        context().name(index_buffer_.handle(), VK_OBJECT_TYPE_BUFFER, "helmet indices");

        // A file says nothing about how big its contents are in any unit you care about,
        // so the model is centred on the origin and the camera is placed off its bounding
        // sphere. This is why the loader returns the bounds: without them the first run
        // of any new model is either an empty screen or a wall of polygons.
        // Centred on the origin, then turned to face the camera. A file carries no
        // convention for which way "forward" is -- glTF fixes the axes and says nothing
        // about orientation -- so this half-turn was chosen by looking at the result.
        model_matrix_ = glm::rotate(glm::mat4(1.0F), glm::radians(215.0F),
                                    glm::vec3(0.0F, 1.0F, 0.0F)) *
                        glm::translate(glm::mat4(1.0F), -model.centre());
        camera_.position = {0.0F, 0.0F, model.radius() * 2.6F};

        depth_format_ = vkc::choose_depth_format(context().physical_device());
        depth_ = vkc::create_depth_buffer(context(), depth_format_, swapchain().extent());

        create_uniform_buffers();
        create_descriptors();
        create_pipeline();

        spdlog::info("Right-click to capture the mouse, WASD to fly, Escape to quit.");
    }

    void on_resize(VkExtent2D extent) override {
        depth_ = vkc::create_depth_buffer(context(), depth_format_, extent);
    }

    void on_event(const SDL_Event& event) override {
        camera_.handle_event(event, window().handle());
    }

    void on_shutdown() override {
        const VkDevice device = context().device();

        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        depth_.destroy();
        index_buffer_.destroy();
        vertex_buffer_.destroy();
    }

    void on_render(const vkc::FrameInfo& frame) override {
        camera_.update(delta_time());

        const size_t slot = frame.frame_number % vkc::kFramesInFlight;
        update_globals(slot, frame.extent);

        vkc::begin_rendering(frame.cmd, frame.view, depth_.view(), frame.extent,
                             {{0.02F, 0.02F, 0.04F, 1.0F}});

        set_viewport(frame);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &frame_sets_[slot], 0, nullptr);

        const VkDeviceSize offset = 0;
        const VkBuffer vertices = vertex_buffer_.handle();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);

        // UINT32, where every chapter so far used UINT16. A cube has 24 vertices and a
        // model has tens of thousands; 65 535 is not a ceiling worth living under, and
        // the cost is two bytes per index.
        vkCmdBindIndexBuffer(frame.cmd, index_buffer_.handle(), 0, VK_INDEX_TYPE_UINT32);

        const PushConstants push{
            .model = model_matrix_,
            .normal_matrix = normal_matrix_of(model_matrix_),
            .colour = kAlbedo,
        };
        vkCmdPushConstants(frame.cmd, pipeline_layout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);

        // One draw call for the whole model, because the whole model is one buffer and
        // one material. That holds for exactly one more chapter.
        vkCmdDrawIndexed(frame.cmd, index_count_, 1, 0, 0, 0);

        vkCmdEndRendering(frame.cmd);
    }

private:
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
            .key_direction = glm::vec4(glm::normalize(kKeyDirection), 0.0F),
            .key_colour = glm::vec4(kKeyColour, 1.0F),
            .fill_direction = glm::vec4(glm::normalize(kFillDirection), 0.0F),
            .fill_colour = glm::vec4(kFillColour, 1.0F),
            .ambient = glm::vec4(kAmbient, kShininess),
        };
        std::memcpy(uniform_buffers_[slot].mapped(), &globals, sizeof(globals));
    }

    void create_descriptors() {
        const VkDevice device = context().device();

        const VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
                                             &frame_set_layout_));

        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                             vkc::kFramesInFlight};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_));

        const std::array<VkDescriptorSetLayout, vkc::kFramesInFlight> layouts{
            frame_set_layout_, frame_set_layout_};
        const VkDescriptorSetAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = vkc::kFramesInFlight,
            .pSetLayouts = layouts.data(),
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &alloc, frame_sets_.data()));

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
    }

    void create_pipeline() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        // Two attributes out of three fields: the shader does not read uv, and an
        // attribute the shader does not declare is one the validation layers object to.
        // The stride still counts it, because the stride is what the buffer holds.
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
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
        }};

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &frame_set_layout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VK_CHECK(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_));

        const VkShaderModule shader =
            vkc::load_shader(device, LVK_CHAPTER_ID, "model.slang");

        pipeline_ = vkc::PipelineBuilder(device)
                        .shaders(shader, "vertexMain", "fragmentMain")
                        .vertex_input(std::span(&binding, 1), attributes)
                        .cull(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE)
                        .depth(depth_format_, /*test=*/true, /*write=*/true,
                               VK_COMPARE_OP_LESS)
                        .colour_attachment(swapchain().format())
                        .layout(pipeline_layout_)
                        .build();

        vkDestroyShaderModule(device, shader, nullptr);

        context().name(pipeline_, VK_OBJECT_TYPE_PIPELINE, "model pipeline");
    }

    vkc::Camera camera_;

    vkc::Buffer vertex_buffer_;
    vkc::Buffer index_buffer_;
    uint32_t index_count_ = 0;
    glm::mat4 model_matrix_{1.0F};

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        vkc::App::Options options{};
        options.title = "LearnVulkan 3.1 - Assimp";
        AssimpApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
