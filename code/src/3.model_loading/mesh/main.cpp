// 3.2 Mesh
//
// 3.1 flattened the whole file into one vertex buffer and painted it grey. That worked
// because a helmet is one mesh with one material, and it is the wrong answer the moment
// a model is two.
//
// This chapter takes the two simplifications apart. Each aiMesh becomes a Mesh that owns
// its own vertex and index buffers and names the material it wants; each aiMaterial
// becomes a Material that owns a texture and a descriptor set. The frame loop stops
// being one draw call and becomes a loop over meshes, binding set 1 as it goes.
//
// The textures come out of the .glb itself rather than from files beside it, which is
// the part with no OpenGL equivalent worth copying: Assimp hands back a pointer to a
// JPEG still in its compressed form, and turning that into a VkImage is this chapter's
// only genuinely new piece of Vulkan plumbing -- and even that is 1.11's, entered from
// the other end.

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

// The chapter decodes a JPEG that is already in memory, which vkcommon's loader cannot
// do for it -- that one takes a path. STB_IMAGE_STATIC is not optional here: Assimp's
// static library exports its own copy of every stbi_* symbol, and only a private one
// links. Chapter 3.1 has the whole story.
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

// Unchanged from 3.1, and this time all three fields are used.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// One run of vertices sharing one material, and the buffers holding it.
//
// This is the type the chapter is named after. It is deliberately small: a mesh is not
// an object, it has no transform and no opinion about where it sits. It is geometry plus
// a pointer at the material that describes it.
struct Mesh {
    vkc::Buffer vertex_buffer;
    vkc::Buffer index_buffer;
    uint32_t index_count = 0;
    uint32_t material = 0;
    std::string name;
};

// Everything the shader needs to know about a surface. One texture, one factor, and the
// descriptor set that presents them to the pipeline.
//
// The set is filled in after loading, because a descriptor set cannot be allocated until
// the pool knows how many there will be -- which is not known until the file has been
// read.
struct Material {
    vkc::Image base_colour;
    glm::vec4 base_colour_factor{1.0F};
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::string name;
};

struct Model {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    glm::vec3 min{};
    glm::vec3 max{};

    [[nodiscard]] glm::vec3 centre() const { return (min + max) * 0.5F; }
    [[nodiscard]] float radius() const { return glm::length(max - min) * 0.5F; }
};

// 3.1's flags exactly. PreTransformVertices is still here -- 3.3 is the chapter that
// walks the node tree instead -- and it is worth being clear about what it does and does
// not do: it bakes node transforms into vertices, and it leaves meshes with different
// materials separate, which is the only reason this chapter has anything to loop over.
constexpr unsigned int kImportFlags =
    aiProcess_Triangulate | aiProcess_GenSmoothNormals |
    aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices |
    aiProcess_ImproveCacheLocality;

[[nodiscard]] glm::vec3 to_glm(const aiVector3D& v) { return {v.x, v.y, v.z}; }

// A single opaque white texel.
//
// Every material gets a base colour texture whether the file supplied one or not, so the
// shader has one code path and the pipeline has one layout. White is the identity for a
// multiply, so a material with no map shows its base colour factor and nothing else.
[[nodiscard]] vkc::Image white_texture(vkc::Context& context) {
    constexpr std::array<uint8_t, 4> white{255, 255, 255, 255};
    return vkc::create_texture(context, white.data(), VkExtent2D{1, 1},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

// Turns whatever the material points at into a VkImage.
//
// Three cases, and glTF only ever produces the first:
//
//   "*3"        an index into scene->mTextures, an image packed into the model file
//   "brick.png" a path, relative to the model, of a file next to it
//   (none)      the material has no base colour map at all
[[nodiscard]] vkc::Image load_base_colour(vkc::Context& context, const aiScene* scene,
                                          const aiMaterial* material,
                                          const std::filesystem::path& model_path) {
    aiString texture_path;
    // BASE_COLOR is glTF's name for it. Assimp also reports the same image under
    // aiTextureType_DIFFUSE for compatibility with older formats, so asking for either
    // works here -- but asking for the one the format actually defines is what keeps a
    // loader honest when the two stop agreeing.
    if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &texture_path) != AI_SUCCESS) {
        return white_texture(context);
    }

    const aiTexture* embedded = scene->GetEmbeddedTexture(texture_path.C_Str());
    if (embedded == nullptr) {
        // A file beside the model. Paths in the file are relative to it, not to the
        // working directory, which is why this joins rather than using it as given.
        return vkc::load_texture(context, model_path.parent_path() / texture_path.C_Str(),
                                 VK_FORMAT_R8G8B8A8_SRGB);
    }

    // mHeight == 0 is Assimp's flag for "this is a compressed file, not a pixel array",
    // and then mWidth is a byte count rather than a width. The uncompressed form exists
    // and no format this series reads produces one, so rather than ship a path that has
    // never run, it says so.
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

    // And from here it is 1.11 again. create_texture is the half of vkc::load_texture
    // that runs after the decode: staging buffer, copy, mip chain, SHADER_READ_ONLY.
    return vkc::create_texture(context, pixels.get(),
                               VkExtent2D{static_cast<uint32_t>(width),
                                          static_cast<uint32_t>(height)},
                               VK_FORMAT_R8G8B8A8_SRGB);
}

// Reads the file and uploads it: one Mesh per aiMesh, one Material per aiMaterial.
//
// 3.1's loader copied vertices into std::vectors and left the GPU out of it. This one
// takes a Context, because the thing it returns is not data any more -- it is buffers and
// images that exist on the device.
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
    model.min = glm::vec3(std::numeric_limits<float>::max());
    model.max = glm::vec3(std::numeric_limits<float>::lowest());

    // Materials first, so that every mesh's material index already points at something.
    // Assimp's count includes a default material it synthesises for meshes that name
    // none; building a Material for it costs one white texel and removes a special case.
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

            model.min = glm::min(model.min, vertex.position);
            model.max = glm::max(model.max, vertex.position);
            vertices.push_back(vertex);
        }

        // No `base` offset any more, and that is the whole point of the chapter. Each
        // mesh has its own index buffer, so its indices are used exactly as the file
        // gives them -- numbered from zero, into its own vertices.
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

    spdlog::info("  bounds ({:.2f} {:.2f} {:.2f}) to ({:.2f} {:.2f} {:.2f})", model.min.x,
                 model.min.y, model.min.z, model.max.x, model.max.y, model.max.z);
    return model;
}

// 3.1's block, byte for byte.
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

// Also 3.1's, and `colour` still 128 bytes in -- but it now carries the material's base
// colour factor rather than a colour this sample chose.
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

// A glTF material has no Phong shininess to read -- it describes roughness and metalness
// instead, which Part 6 is where this series learns to use. Until then the number is
// chosen here, as it was in 3.1.
constexpr float kShininess = 48.0F;

// 3.1's lighting, unchanged, so that the only visible difference between the two
// chapters is where the surface colour comes from.
constexpr glm::vec3 kKeyDirection{-0.5F, -0.8F, -0.6F};
constexpr glm::vec3 kKeyColour{0.95F, 0.96F, 1.00F};
constexpr glm::vec3 kFillDirection{0.7F, -0.2F, 0.7F};
constexpr glm::vec3 kFillColour{0.35F, 0.28F, 0.22F};
constexpr glm::vec3 kAmbient{0.06F, 0.06F, 0.08F};

class MeshApp : public vkc::App {
public:
    using vkc::App::App;

protected:
    void on_start() override {
        model_ = load_model(context(),
                            vkc::asset_path("models/damaged_helmet/DamagedHelmet.glb"));

        // One sampler for every material in the model. A sampler says how to read a
        // texture, not which one, so a scene needs a handful of them and not one each.
        sampler_ = vkc::Sampler(context(), vkc::SamplerDesc{});

        model_matrix_ = glm::rotate(glm::mat4(1.0F), glm::radians(215.0F),
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
        // Clearing the vectors runs every Buffer and Image destructor in them. The sets
        // they held need no freeing of their own: destroying the pool above took them.
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

        // The frame loop 3.1 did not have. Everything inside it is per mesh: its
        // material's set, its two buffers, its push constants, its draw.
        for (const Mesh& mesh : model_.meshes) {
            const Material& material = model_.materials[mesh.material];

            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout_, 1, 1, &material.set, 0, nullptr);

            const VkDeviceSize offset = 0;
            const VkBuffer vertices = mesh.vertex_buffer.handle();
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertices, &offset);
            vkCmdBindIndexBuffer(frame.cmd, mesh.index_buffer.handle(), 0,
                                 VK_INDEX_TYPE_UINT32);

            const PushConstants push{
                .model = model_matrix_,
                .normal_matrix = normal_matrix_of(model_matrix_),
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

        // Set 1 is one combined image sampler. Every material in the model has the same
        // layout even where the file gave it no texture, because the layout is baked into
        // the pipeline and a second shape of material would mean a second pipeline.
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

        // The pool is sized from the model rather than from a constant, which is the
        // first time in this series that has been true. A pool cannot grow: ask for too
        // few and vkAllocateDescriptorSets fails with OUT_OF_POOL_MEMORY.
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

        // One set per material, allocated and written in a loop. This is the shape every
        // renderer ends up with, and 3.4 is where it gets replaced by a single set
        // holding every texture at once.
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

        // Three attributes for three fields. 3.1 declared two and left uv out because
        // nothing sampled it; the fragment shader reads it now, so it is described.
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
            vkc::load_shader(device, LVK_CHAPTER_ID, "mesh.slang");

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

        context().name(pipeline_, VK_OBJECT_TYPE_PIPELINE, "mesh pipeline");
    }

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
        vkc::App::Options options{};
        options.title = "LearnVulkan 3.2 - Mesh";
        MeshApp app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
