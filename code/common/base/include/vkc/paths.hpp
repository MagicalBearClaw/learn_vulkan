#pragma once

#include <filesystem>
#include <string_view>
#include <vector>

namespace vkc {

// Where the build put this chapter's compiled SPIR-V, and where bootstrap.py put the
// shared textures and models. Both are baked in at configure time so a sample runs
// correctly from any working directory, including from inside a debugger.
[[nodiscard]] std::filesystem::path shader_path(std::string_view chapter_id,
                                                std::string_view shader_name);

[[nodiscard]] std::filesystem::path asset_path(std::string_view relative);

// Reads a whole file as bytes. SPIR-V is the only consumer for now, and it must be
// 4-byte aligned, which a std::vector<uint32_t> gives us for free.
[[nodiscard]] std::vector<uint32_t> read_spirv(const std::filesystem::path& path);

}  // namespace vkc
