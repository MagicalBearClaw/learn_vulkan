// 3.4 Bindless materials
//
// 3.3 bound a descriptor set before every draw. Six meshes, six binds; a city scene,
// fifty thousand. That is the last per-draw cost in this series that exists for no reason
// other than how the descriptors were arranged, and this chapter removes it.
//
// The idea is one sentence long: put every texture the model uses into one array, put
// every material's parameters into one buffer, and let a draw name its material with a
// single integer in the push constants. Set 1 is then bound once for the whole frame and
// the draw loop stops calling vkCmdBindDescriptorSets entirely.
//
// This is the shape every modern renderer converges on, and it is the doorway to Part 9:
// once a draw is fully described by numbers in buffers rather than by binds, the CPU
// stops needing to be in the loop at all.

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

// As in 3.2 and 3.3: Assimp's static library exports its own stbi_* symbols, so this
// chapter's copy has to be private. 3.1 has the whole story.
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
#include <unordered_map>
#include <vector>

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// 3.3's Mesh, unchanged.
struct Mesh {
    vkc::Buffer vertex_buffer;
    vkc::Buffer index_buffer;
    uint32_t index_count = 0;
    uint32_t material = 0;
    std::string name;
    glm::vec3 min{};
    glm::vec3 max{};
};

// 3.3's Draw, unchanged.
struct Draw {
    uint32_t mesh = 0;
    glm::mat4 transform{1.0F};
    std::string node;
};

// A material as the GPU sees it, and the only form of material this chapter keeps.
//
// 3.3's Material owned a vkc::Image and a VkDescriptorSet. This owns neither: it is
// sixteen bytes of colour and an index into the texture array, destined for a storage
// buffer. The descriptor set it used to own does not exist any more, because there is
// now one set for the whole model.
//
// std430 lays a struct out on its largest member's alignment, so this must be a multiple
// of 16 bytes for the array stride to match what the shader reads. The padding is not
// decoration; drop it and every material after the first reads the wrong memory.
//
// The shader-side struct has to be padded the same way *and* with the same shapes. Three
// scalar uints there, not a uint3: std430 aligns a three-component vector to 16 bytes, so
// a uint3 would sit at offset 32 and make the GPU's stride 48 against this 32. Nothing
// warns about it -- the first material reads correctly and the rest read garbage.
struct MaterialGpu {
    glm::vec4 base_colour_factor{1.0F};  // 0
    uint32_t texture = 0;                // 16
    uint32_t pad0 = 0;                   // 20
    uint32_t pad1 = 0;                   // 24
    uint32_t pad2 = 0;                   // 28
};
static_assert(sizeof(MaterialGpu) == 32);
static_assert(offsetof(MaterialGpu, texture) == 16);

struct Model {
    std::vector<Mesh> meshes;
    std::vector<Draw> draws;

    // Every distinct image the model uses, in the order the array binding expects.
    std::vector<vkc::Image> textures;
    // One record per aiMaterial, uploaded to the GPU as one buffer.
    std::vector<MaterialGpu> materials;
    std::vector<std::string> material_names;

    vkc::Buffer material_buffer;

    glm::vec3 min{};
    glm::vec3 max{};

    [[nodiscard]] glm::vec3 centre() const { return (min + max) * 0.5F; }
    [[nodiscard]] float radius() const { return glm::length(max - min) * 0.5F; }
};

// 3.3's flags, unchanged. The node tree is still walked.
constexpr unsigned int kImportFlags =
    aiProcess_Triangulate | aiProcess_GenSmoothNormals |
    aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;

[[nodiscard]] glm::vec3 to_glm(const aiVector3D& v) { return {v.x, v.y, v.z}; }

// 3.3's transpose. Assimp stores rows, glm stores columns.
[[nodiscard]] glm::mat4 to_glm(const aiMatrix4x4& m) {
    return {m.a1, m.b1, m.c1, m.d1,   // column 0
            m.a2, m.b2, m.c2, m.d2,   // column 1
            m.a3, m.b3, m.c3, m.d3,   // column 2
            m.a4, m.b4, m.c4, m.d4};  // column 3
}

[[nodiscard]] vkc::Image white_texture(vkc::Context& context) {
    constexpr std::array<uint8_t, 4> white{255, 255, 255, 255};
    return vkc::create_texture(context, white.data(), VkExtent2D{1, 1},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

// 3.2's decode, returning an image rather than assigning one.
[[nodiscard]] vkc::Image decode_texture(vkc::Context& context, const aiScene* scene,
                                        const aiString& texture_path,
                                        const std::filesystem::path& model_path) {
    const aiTexture* embedded = scene->GetEmbeddedTexture(texture_path.C_Str());
    if (embedded == nullptr) {
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

    return vkc::create_texture(context, pixels.get(),
                               VkExtent2D{static_cast<uint32_t>(width),
                                          static_cast<uint32_t>(height)},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

// Loads a texture at most once, and returns its index in the array.
//
// This cache is not an optimisation bolted on at the end; it is the reason the shader has
// two indices instead of one. 3.3 gave every material its own vkc::Image, so a model with
// two materials sharing an image uploaded it twice and nothing noticed, because each
// material had a descriptor set of its own to put it in. With one shared array, the
// duplicate is visible: it is an extra 4 MB sitting at another index.
[[nodiscard]] uint32_t texture_index(vkc::Context& context, const aiScene* scene,
                                     const aiMaterial* material,
                                     const std::filesystem::path& model_path,
                                     std::unordered_map<std::string, uint32_t>& cache,
                                     std::vector<vkc::Image>& textures) {
    aiString texture_path;
    // The empty key is the fallback white texel, so a material with no base colour map
    // costs one array slot the first time and none after that.
    const std::string key =
        material->GetTexture(aiTextureType_BASE_COLOR, 0, &texture_path) == AI_SUCCESS
            ? texture_path.C_Str()
            : std::string{};

    const auto found = cache.find(key);
    if (found != cache.end()) {
        spdlog::info("    base colour \"{}\" -> texture {} (already loaded)",
                     key.empty() ? "(none)" : key, found->second);
        return found->second;
    }

    const auto index = static_cast<uint32_t>(textures.size());
    textures.push_back(key.empty() ? white_texture(context)
                                   : decode_texture(context, scene, texture_path,
                                                    model_path));
    cache.emplace(key, index);

    spdlog::info("    base colour \"{}\" -> texture {}", key.empty() ? "(none)" : key,
                 index);
    return index;
}

// 3.3's walk, unchanged.
void collect_draws(const aiNode* node, const glm::mat4& parent, Model& model) {
    const glm::mat4 transform = parent * to_glm(node->mTransformation);

    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        model.draws.push_back(Draw{
            .mesh = node->mMeshes[i],
            .transform = transform,
            .node = node->mName.C_Str(),
        });
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        collect_draws(node->mChildren[i], transform, model);
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

    std::unordered_map<std::string, uint32_t> texture_cache;
    model.materials.reserve(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        const aiMaterial* source = scene->mMaterials[i];

        aiColor4D factor(1.0F, 1.0F, 1.0F, 1.0F);
        source->Get(AI_MATKEY_BASE_COLOR, factor);

        spdlog::info("  material {} \"{}\"", i, source->GetName().C_Str());

        model.materials.push_back(MaterialGpu{
            .base_colour_factor = {factor.r, factor.g, factor.b, factor.a},
            .texture = texture_index(context, scene, source, path, texture_cache,
                                     model.textures),
        });
        model.material_names.emplace_back(source->GetName().C_Str());
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

    collect_draws(scene->mRootNode, glm::mat4(1.0F), model);

    if (model.draws.empty()) {
        throw std::runtime_error(
            "Model has meshes but no node refers to them: " + path.string());
    }

    // One upload for every material in the model, where 3.3 had one descriptor set each.
    model.material_buffer = vkc::upload_to_device_local(
        context, model.materials.data(), model.materials.size() * sizeof(MaterialGpu),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    context.name(model.material_buffer.handle(), VK_OBJECT_TYPE_BUFFER,
                 "material buffer");

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

    spdlog::info("  {} draw(s), {} material(s), {} distinct texture(s)",
                 model.draws.size(), model.materials.size(), model.textures.size());
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

// 116 bytes, down from 128, and the saving is the whole idea in miniature.
//
// `colour` carried a vec4 of base colour factor through three chapters and sat at offset
// 112. The factor lives in the material buffer now, and what travels in its place is the
// four-byte index that finds it -- twelve bytes less per draw, and, far more to the
// point, no descriptor bind.
struct PushConstants {
    glm::mat4 model;                         // 0
    std::array<glm::vec4, 3> normal_matrix;  // 64
    uint32_t material;                       // 112
};
static_assert(offsetof(PushConstants, material) == 112);
static_assert(sizeof(PushConstants) == 116);

[[nodiscard]] std::array<glm::vec4, 3> normal_matrix_of(const glm::mat4& model) {
    const glm::mat3 m = glm::inverseTranspose(glm::mat3(model));
    return {glm::vec4(m[0], 0.0F), glm::vec4(m[1], 0.0F), glm::vec4(m[2], 0.0F)};
}

constexpr float kShininess = 48.0F;

// 3.1's lighting rig, unchanged through four chapters.
constexpr glm::vec3 kKeyDirection{-0.5F, -0.8F, -0.6F};
constexpr glm::vec3 kKeyColour{0.95F, 0.96F, 1.00F};
constexpr glm::vec3 kFillDirection{0.7F, -0.2F, 0.7F};
constexpr glm::vec3 kFillColour{0.35F, 0.28F, 0.22F};
constexpr glm::vec3 kAmbient{0.06F, 0.06F, 0.08F};

class BindlessApp : public vkc::App {
public:
    BindlessApp(Options options, std::filesystem::path model_path)
        : vkc::App(std::move(options)), model_path_(std::move(model_path)) {}

protected:
    void on_start() override {
        model_ = load_model(context(), model_path_);

        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

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
        vkDestroyDescriptorSetLayout(device, bindless_set_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, frame_set_layout_, nullptr);

        for (vkc::Buffer& buffer : uniform_buffers_) {
            buffer.destroy();
        }
        sampler_.destroy();
        model_.material_buffer.destroy();
        model_.textures.clear();
        model_.meshes.clear();
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

        // Both sets, once, for the whole frame. Set 1 holds every material and every
        // texture in the model, so there is nothing left for the draw loop to rebind.
        const std::array<VkDescriptorSet, 2> sets{frame_sets_[slot], bindless_set_};
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0,
                                static_cast<uint32_t>(sets.size()), sets.data(), 0,
                                nullptr);

        for (const Draw& draw : model_.draws) {
            const Mesh& mesh = model_.meshes[draw.mesh];

            const VkDeviceSize offset = 0;
            const VkBuffer vertices = mesh.vertex_buffer.handle();
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
            vkCmdBindIndexBuffer(frame.cmd, mesh.index_buffer.handle(), 0,
                                 VK_INDEX_TYPE_UINT32);

            const glm::mat4 model_matrix = model_matrix_ * draw.transform;

            const PushConstants push{
                .model = model_matrix,
                .normal_matrix = normal_matrix_of(model_matrix),
                .material = mesh.material,
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

        const auto texture_count = static_cast<uint32_t>(model_.textures.size());

        // Set 1: the whole model's materials and textures.
        //
        // descriptorCount on binding 1 is where the array's size is decided. The shader
        // declares the array unbounded and takes its length from here, which is why the
        // layout cannot be built until the model has been read.
        const std::array<VkDescriptorSetLayoutBinding, 2> bindless_bindings{{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = texture_count,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = nullptr,
            },
        }};
        const VkDescriptorSetLayoutCreateInfo bindless_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindless_bindings.size()),
            .pBindings = bindless_bindings.data(),
        };
        VK_CHECK(vkCreateDescriptorSetLayout(device, &bindless_layout_info, nullptr,
                                             &bindless_set_layout_));

        // Three sets now, whatever the model: two per-frame and one bindless. 3.3 needed
        // one per material, so this pool no longer grows with the scene.
        const std::array<VkDescriptorPoolSize, 3> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkc::kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texture_count},
        }};
        const VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = vkc::kFramesInFlight + 1,
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

        const VkDescriptorSetAllocateInfo bindless_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &bindless_set_layout_,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &bindless_alloc, &bindless_set_));

        // Every texture in one write. descriptorCount is the array length and pImageInfo
        // points at that many structs: one vkUpdateDescriptorSets call fills the whole
        // binding, however many images the model turned out to have.
        std::vector<VkDescriptorImageInfo> image_infos;
        image_infos.reserve(model_.textures.size());
        for (const vkc::Image& texture : model_.textures) {
            image_infos.push_back(VkDescriptorImageInfo{
                .sampler = sampler_.handle(),
                .imageView = texture.view(),
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
        }

        const VkDescriptorBufferInfo material_info{
            .buffer = model_.material_buffer.handle(),
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        const std::array<VkWriteDescriptorSet, 2> writes{{
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = bindless_set_,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &material_info,
                .pTexelBufferView = nullptr,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = bindless_set_,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = texture_count,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = image_infos.data(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);

        context().name(bindless_set_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "bindless set");
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
                                                               bindless_set_layout_};

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
            vkc::load_shader(device, LVK_CHAPTER_ID, "bindless_materials.slang");

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

        context().name(pipeline_, VK_OBJECT_TYPE_PIPELINE, "bindless pipeline");
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
    VkDescriptorSetLayout bindless_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet bindless_set_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        // 3.3's --model flag, stripped here before vkc::parse_args sees it.
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
        options.title = "LearnVulkan 3.4 - Bindless Materials";
        BindlessApp app(vkc::App::parse_args(static_cast<int>(forwarded.size()),
                                             forwarded.data(), options),
                        std::move(model_path));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
