#include "vkc/paths.hpp"

#include <cstdio>
#include <format>
#include <fstream>
#include <stdexcept>

namespace vkc {

std::filesystem::path shader_path(std::string_view chapter_id,
                                  std::string_view shader_name) {
    return std::filesystem::path(LVK_SHADER_DIR) / chapter_id /
           std::format("{}.spv", shader_name);
}

std::filesystem::path asset_path(std::string_view relative) {
    return std::filesystem::path(LVK_ASSET_DIR) / relative;
}

std::vector<uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::format(
            "Could not open SPIR-V module '{}'. Did the build compile the shaders?",
            path.string()));
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        throw std::runtime_error(std::format(
            "'{}' is {} bytes, which is not a valid SPIR-V module (must be a "
            "non-zero multiple of 4).",
            path.string(), size));
    }

    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), size);
    if (!file) {
        throw std::runtime_error(
            std::format("Failed while reading '{}'.", path.string()));
    }
    return words;
}

}  // namespace vkc
