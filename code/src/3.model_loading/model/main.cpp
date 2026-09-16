// 3.3 Model
//
// 3.2 stopped concatenating meshes, and left one simplification standing:
// aiProcess_PreTransformVertices, which walks the node tree at import time, multiplies
// every vertex by its node's accumulated transform, and throws the tree away. That flag
// is gone in this chapter, and everything it was doing has to be done here instead.
//
// The reason is not tidiness. A node transform is data the file's author wrote down, and
// baking it in destroys it: once the vertices have moved, nothing can say where a part
// belongs, animate it, instance it, or cull it as a unit. A model is a tree, and this
// chapter is where it stops being flattened into a list.
//
// The measurable consequence is smaller than it sounds and larger than it looks. Drop the
// flag without walking the tree and DamagedHelmet -- one mesh, one node -- lies on its
// back, because that node carries a -90 degree rotation about X. FlightHelmet, six meshes
// and six nodes, does not move at all: every one of its transforms is identity. Both
// facts are in this chapter's log output, and both are the point.

#include <vkc/app.hpp>
#include <vkc/buffer.hpp>
#include <vkc/camera.hpp>
#include <vkc/check.hpp>
#include <vkc/image.hpp>
#include <vkc/paths.hpp>
#include <vkc/pipeline.hpp>

#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

// As in 3.2: Assimp's static library exports its own stbi_* symbols, so this chapter's
// copy has to be private. 3.1 has the whole story.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Unchanged since 3.1.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// Geometry and the material that describes it -- 3.2's type, and deliberately still
// without a transform.
//
// It would be easy to add one here, and 3.2's last exercise suggested exactly that. It is
// the wrong place. An aiNode refers to its meshes by index, so two nodes may name the
// same mesh, and a model that uses one bolt in forty places stores the bolt once. Put the
// transform on the mesh and that model cannot be expressed at all. Where something sits
// is a property of the node, not of the geometry.
struct Mesh {
    vkc::Buffer vertex_buffer;
    vkc::Buffer index_buffer;
    uint32_t index_count = 0;
    uint32_t material = 0;
    std::string name;
    // Local-space bounds, kept so the world-space bounds can be worked out per draw.
    glm::vec3 min{};
    glm::vec3 max{};
};

// 3.2's type, unchanged.
struct Material {
    vkc::Image base_colour;
    glm::vec4 base_colour_factor{1.0F};
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::string name;
};

// One mesh, drawn once, somewhere.
//
// This is what the tree walk produces and what the frame loop iterates: the hierarchy
// flattened into a draw list, with each node's accumulated transform carried along. The
// tree is walked once at load time because nothing in this chapter moves; an engine with
// animation walks it every frame and rebuilds exactly this list.
struct Draw {
    uint32_t mesh = 0;
    glm::mat4 transform{1.0F};
    std::string node;
};

struct Model {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Draw> draws;
    glm::vec3 min{};
    glm::vec3 max{};

    [[nodiscard]] glm::vec3 centre() const { return (min + max) * 0.5F; }
    [[nodiscard]] float radius() const { return glm::length(max - min) * 0.5F; }
};

// 3.2's flags, less one. aiProcess_PreTransformVertices is what this chapter replaces.
//
// Worth knowing before removing it from your own loader: the flag does more than bake
// transforms. It also merges meshes that share a material and discards the scene's
// cameras and lights. Dropping it therefore costs a few more draw calls as well as this
// tree walk -- and buys back everything the file said about structure.
constexpr unsigned int kImportFlags =
    aiProcess_Triangulate | aiProcess_GenSmoothNormals |
    aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;

[[nodiscard]] glm::vec3 to_glm(const aiVector3D& v) { return {v.x, v.y, v.z}; }

// aiMatrix4x4 to glm::mat4, transposed on the way through.
//
// This is the single most common bug in a hand-written Assimp loader, and it is silent:
// both types are sixteen floats and either will compile. Assimp stores rows (a1 a2 a3 a4
// is the first *row*); glm stores columns. Feed one to the other unchanged and every
// transform comes out transposed -- which for a pure rotation means rotating the opposite
// way, and for a translation means the offset lands in the wrong place entirely.
//
// glm's constructor takes columns, so column 0 is (a1 b1 c1 d1): the first column of
// Assimp's matrix, read down.
[[nodiscard]] glm::mat4 to_glm(const aiMatrix4x4& m) {
    return {m.a1, m.b1, m.c1, m.d1,   // column 0
            m.a2, m.b2, m.c2, m.d2,   // column 1
            m.a3, m.b3, m.c3, m.d3,   // column 2
            m.a4, m.b4, m.c4, m.d4};  // column 3
}

// 3.2's texture loading, unchanged and now exercising a branch it never reached.
//
// DamagedHelmet packs its textures into the .glb, so 3.2 only ever ran the embedded case.
// FlightHelmet ships PNGs beside the .gltf, so this chapter's default model runs the
// other one. The code did not need to change; it needed a second model to prove it.
[[nodiscard]] vkc::Image white_texture(vkc::Context& context) {
    constexpr std::array<uint8_t, 4> white{255, 255, 255, 255};
    return vkc::create_texture(context, white.data(), VkExtent2D{1, 1},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

[[nodiscard]] vkc::Image load_base_colour(vkc::Context& context, const aiScene* scene,
                                          const aiMaterial* material,
                                          const std::filesystem::path& model_path) {
    aiString texture_path;
    if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &texture_path) != AI_SUCCESS) {
        return white_texture(context);
    }

    const aiTexture* embedded = scene->GetEmbeddedTexture(texture_path.C_Str());
    if (embedded == nullptr) {
        spdlog::info("    base colour \"{}\" (file)", texture_path.C_Str());
        return vkc::load_texture(context, model_path.parent_path() / texture_path.C_Str(),
                                 VK_FORMAT_R8G8B8A8_SRGB);
    }

    if (embedded->mHeight != 0) {
        throw std::runtime_error(
            std::format("Embedded texture '{}' is an uncompressed aiTexel array, which "
                        "this chapter does not decode.",
                        texture_path.C_Str()));
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    const std::unique_ptr<stbi_uc, void (*)(void*)> pixels(
        stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(embedded->pcData),
                              static_cast<int>(embedded->mWidth), &width, &height,
                              &channels, STBI_rgb_alpha),
        stbi_image_free);
    if (pixels == nullptr) {
        throw std::runtime_error(std::format("Could not decode embedded texture '{}': {}",
                                             texture_path.C_Str(),
                                             stbi_failure_reason()));
    }

    spdlog::info("    base colour \"{}\": {}x{} {} ({} KiB compressed)",
                 texture_path.C_Str(), width, height, embedded->achFormatHint,
                 embedded->mWidth / 1024);

    return vkc::create_texture(context, pixels.get(),
                               VkExtent2D{static_cast<uint32_t>(width),
                                          static_cast<uint32_t>(height)},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

// The chapter, in fourteen lines.
//
// Depth-first, carrying the product of every transform from the root down. A node's own
// transform is relative to its parent, so the matrix that puts its meshes in world space
// is the whole chain multiplied together -- and multiplied in this order, parent on the
// left, because these are column vectors and the rightmost matrix is applied first.
//
// Nothing here touches the GPU or reads a vertex. It reads structure, which is the part
// PreTransformVertices was destroying.
void collect_draws(const aiNode* node, const glm::mat4& parent, Model& model,
                   int depth) {
    const glm::mat4 transform = parent * to_glm(node->mTransformation);

    if (node->mNumMeshes > 0) {
        spdlog::info("  {:>{}}node \"{}\": {} mesh(es){}", "", depth * 2,
                     node->mName.C_Str(), node->mNumMeshes,
                     to_glm(node->mTransformation) == glm::mat4(1.0F) ? ""
                                                                     : ", transformed");
    }

    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        model.draws.push_back(Draw{
            .mesh = node->mMeshes[i],
            .transform = transform,
            .node = node->mName.C_Str(),
        });
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        collect_draws(node->mChildren[i], transform, model, depth + 1);
    }
}

[[nodiscard]] Model load_model(vkc::Context& context,
                               const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Model not found: " + path.string() +
            "\nFetch the tutorial assets first:  python3 tools/bootstrap.py");
    }

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path.string(), kImportFlags);

    if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0 ||
        scene->mRootNode == nullptr) {
        throw std::runtime_error("Assimp could not read " + path.string() + ": " +
                                 importer.GetErrorString());
    }

    spdlog::info("{}: {} mesh(es), {} material(s), {} embedded texture(s)",
                 path.filename().string(), scene->mNumMeshes, scene->mNumMaterials,
                 scene->mNumTextures);

    Model model;

    // Materials first, then meshes, then the tree -- 3.2's order with one stage added.
    model.materials.reserve(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        const aiMaterial* source = scene->mMaterials[i];

        aiColor4D factor(1.0F, 1.0F, 1.0F, 1.0F);
        source->Get(AI_MATKEY_BASE_COLOR, factor);

        spdlog::info("  material {} \"{}\"", i, source->GetName().C_Str());

        model.materials.push_back(Material{
            .base_colour = load_base_colour(context, scene, source, path),
            .base_colour_factor = {factor.r, factor.g, factor.b, factor.a},
            .set = VK_NULL_HANDLE,
            .name = source->GetName().C_Str(),
        });
    }

    model.meshes.reserve(scene->mNumMeshes);
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* source = scene->mMeshes[m];

        glm::vec3 min(std::numeric_limits<float>::max());
        glm::vec3 max(std::numeric_limits<float>::lowest());

        std::vector<Vertex> vertices;
        vertices.reserve(source->mNumVertices);
        for (unsigned int v = 0; v < source->mNumVertices; ++v) {
            Vertex vertex{};
            vertex.position = to_glm(source->mVertices[v]);
            vertex.normal = source->HasNormals() ? to_glm(source->mNormals[v])
                                                 : glm::vec3(0.0F, 1.0F, 0.0F);
            vertex.uv = source->HasTextureCoords(0)
                            ? glm::vec2(source->mTextureCoords[0][v].x,
                                        source->mTextureCoords[0][v].y)
                            : glm::vec2(0.0F);

            // Local space now, not world space. 3.2 could accumulate straight into the
            // model's bounds because every vertex was already where it belonged; these
            // have not been placed yet.
            min = glm::min(min, vertex.position);
            max = glm::max(max, vertex.position);
            vertices.push_back(vertex);
        }

        std::vector<uint32_t> indices;
        indices.reserve(static_cast<size_t>(source->mNumFaces) * 3);
        for (unsigned int f = 0; f < source->mNumFaces; ++f) {
            const aiFace& face = source->mFaces[f];
            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                indices.push_back(face.mIndices[i]);
            }
        }

        if (vertices.empty() || indices.empty()) {
            throw std::runtime_error(
                std::format("Mesh {} of {} has no geometry", m, path.string()));
        }

        Mesh mesh{
            .vertex_buffer = vkc::upload_to_device_local(
                context, vertices.data(), vertices.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
            .index_buffer = vkc::upload_to_device_local(
                context, indices.data(), indices.size() * sizeof(uint32_t),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
            .index_count = static_cast<uint32_t>(indices.size()),
            .material = source->mMaterialIndex,
            .name = source->mName.C_Str(),
            .min = min,
            .max = max,
        };

        context.name(mesh.vertex_buffer.handle(), VK_OBJECT_TYPE_BUFFER,
                     std::format("{} vertices", mesh.name).c_str());
        context.name(mesh.index_buffer.handle(), VK_OBJECT_TYPE_BUFFER,
                     std::format("{} indices", mesh.name).c_str());

        spdlog::info("  mesh {} \"{}\": {} vertices, {} indices, material {}", m,
                     mesh.name, vertices.size(), indices.size(), mesh.material);

        model.meshes.push_back(std::move(mesh));
    }

    if (model.meshes.empty()) {
        throw std::runtime_error("Model has no meshes: " + path.string());
    }

    spdlog::info("  node tree:");
    collect_draws(scene->mRootNode, glm::mat4(1.0F), model, 1);

    if (model.draws.empty()) {
        throw std::runtime_error(
            "Model has meshes but no node refers to them: " + path.string() +
            "\nThe file's scene graph is empty, which no glTF this series uses should be.");
    }

    // World-space bounds, which now take a pass of their own.
    //
    // Each mesh's local box has eight corners; a transform moves all eight, and the box
    // around the moved corners contains the moved geometry. It can be looser than the box
    // around the transformed vertices -- a rotated box does not stay axis-aligned -- and
    // for framing a camera that does not matter. For culling it would.
    model.min = glm::vec3(std::numeric_limits<float>::max());
    model.max = glm::vec3(std::numeric_limits<float>::lowest());
    for (const Draw& draw : model.draws) {
        const Mesh& mesh = model.meshes[draw.mesh];
        for (int corner = 0; corner < 8; ++corner) {
            const glm::vec3 local((corner & 1) != 0 ? mesh.max.x : mesh.min.x,
                                  (corner & 2) != 0 ? mesh.max.y : mesh.min.y,
                                  (corner & 4) != 0 ? mesh.max.z : mesh.min.z);
            const glm::vec3 world = glm::vec3(draw.transform * glm::vec4(local, 1.0F));
            model.min = glm::min(model.min, world);
            model.max = glm::max(model.max, world);
        }
    }

    spdlog::info("  {} draw(s) from {} mesh(es)", model.draws.size(),
                 model.meshes.size());
    spdlog::info("  world bounds ({:.3f} {:.3f} {:.3f}) to ({:.3f} {:.3f} {:.3f})",
                 model.min.x, model.min.y, model.min.z, model.max.x, model.max.y,
                 model.max.z);
    return model;
}

// 3.1's block, still byte for byte.
struct Globals {
    glm::mat4 view;            // 0
    glm::mat4 projection;      // 64
    glm::vec4 view_position;   // 128
    glm::vec4 key_direction;   // 144
    glm::vec4 key_colour;      // 160
    glm::vec4 fill_direction;  // 176
    glm::vec4 fill_colour;     // 192
    glm::vec4 ambient;         // 208, w is the shininess
};
static_assert(offsetof(Globals, ambient) == 208);
static_assert(sizeof(Globals) == 224);

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

// 3.1's lighting rig, unchanged through three chapters.
constexpr glm::vec3 kKeyDirection{-0.5F, -0.8F, -0.6F};
constexpr glm::vec3 kKeyColour{0.95F, 0.96F, 1.00F};
constexpr glm::vec3 kFillDirection{0.7F, -0.2F, 0.7F};
constexpr glm::vec3 kFillColour{0.35F, 0.28F, 0.22F};
constexpr glm::vec3 kAmbient{0.06F, 0.06F, 0.08F};

class ModelApp : public vkc::App {
public:
    ModelApp(Options options, std::filesystem::path model_path)
        : vkc::App(std::move(options)), model_path_(std::move(model_path)) {}

protected:
    void on_start() override {
        model_ = load_model(context(), model_path_);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

        // Framing is computed from the world-space bounds rather than chosen, because
        // this sample has two models with quite different sizes: FlightHelmet stands
        // 0.72 units tall with its feet on y=0, DamagedHelmet is about 2 units across
        // and centred on the origin. Subtracting the centre puts either one in front of
        // the camera without a per-model constant.
        model_matrix_ = glm::rotate(glm::mat4(1.0F), glm::radians(200.0F),
                                    glm::vec3(0.0F, 1.0F, 0.0F)) *
                        glm::translate(glm::mat4(1.0F), -model_.centre());
        camera_.position = {0.0F, 0.0F, model_.radius() * 2.6F};

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
        vkDestroyDescriptorSetLayout(device, material_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        sampler_.destroy();
        model_.meshes.clear();
        model_.materials.clear();
        depth_.destroy();
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

        // 3.2 looped over meshes; this loops over draws. The difference is one line --
        // the multiplication below -- and it is the whole chapter.
        for (const Draw& draw : model_.draws) {
            const Mesh& mesh = model_.meshes[draw.mesh];
            const Material& material = model_.materials[mesh.material];

            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout_, 1, 1, &material.set, 0, nullptr);

            const VkDeviceSize offset = 0;
            const VkBuffer vertices = mesh.vertex_buffer.handle();
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
            vkCmdBindIndexBuffer(frame.cmd, mesh.index_buffer.handle(), 0,
                                 VK_INDEX_TYPE_UINT32);

            // The node's transform first, then the model matrix that frames the whole
            // thing. Reading right to left: a vertex is placed by its node, and the
            // result is placed by the sample.
            const glm::mat4 model_matrix = model_matrix_ * draw.transform;

            const PushConstants push{
                .model = model_matrix,
                .normal_matrix = normal_matrix_of(model_matrix),
                .colour = material.base_colour_factor,
            };
            vkCmdPushConstants(frame.cmd, pipeline_layout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);

            vkCmdDrawIndexed(frame.cmd, mesh.index_count, 1, 0, 0, 0);
        }

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

        const VkDescriptorSetLayoutBinding frame_binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo frame_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &frame_binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &frame_layout_info, nullptr,
                                             &frame_set_layout_));

        const VkDescriptorSetLayoutBinding material_binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        };
        const VkDescriptorSetLayoutCreateInfo material_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &material_binding,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &material_layout_info, nullptr,
                                             &material_set_layout_));

        // Still sized from the materials, not the draws. A descriptor set describes a
        // surface, and six draws sharing two materials need two sets, not six.
        const auto material_count = static_cast<uint32_t>(model_.materials.size());
        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, material_count},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + material_count,
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

        for (Material& material : model_.materials) {
            const VkDescriptorSetAllocateInfo alloc{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = descriptor_pool_,
                .descriptorSetCount = 1,
                .pSetLayouts = &material_set_layout_,
            };
            VK_CHECK(vkAllocateDescriptorSets(device, &alloc, &material.set));

            const VkDescriptorImageInfo image_info{
                .sampler = sampler_.handle(),
                .imageView = material.base_colour.view(),
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = material.set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            };
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

            context().name(material.set, VK_OBJECT_TYPE_DESCRIPTOR_SET,
                           std::format("material {}", material.name).c_str());
        }
    }

    void create_pipeline() {
        const VkDevice device = context().device();

        const VkVertexInputBindingDescription binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        const std::array<VkVertexInputAttributeDescription, 3> attributes{{
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
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
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

    std::filesystem::path model_path_;
    vkc::Camera camera_;

    Model model_;
    vkc::Sampler sampler_;
    glm::mat4 model_matrix_{1.0F};

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    vkc::Image depth_;

    std::array<vkc::Buffer, vkc::kFramesInFlight> uniform_buffers_;
    std::array<VkDescriptorSet, vkc::kFramesInFlight> frame_sets_{};

    VkDescriptorSetLayout frame_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout material_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        // --model is this chapter's own flag, so it is taken out here rather than added
        // to vkc::parse_args, which every sample in the book shares and which rejects
        // anything it does not recognise. Two models is a Part 3 concern, not a book-wide
        // one.
        std::filesystem::path model_path =
            vkc::asset_path("models/flight_helmet/FlightHelmet.gltf");

        std::vector<char*> forwarded{argv[0]};
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
                continue;
            }
            forwarded.push_back(argv[i]);
        }

        vkc::App::Options options{};
        options.title = "LearnVulkan 3.3 - Model";
        ModelApp app(vkc::App::parse_args(static_cast<int>(forwarded.size()),
                                          forwarded.data(), options),
                     std::move(model_path));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
