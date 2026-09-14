#pragma once

#include <filesystem>
#include <string_view>

namespace vkc {

// Where the build staged this chapter's shader sources, and where bootstrap.py put the
// shared textures and models. Both are baked in at configure time so a sample runs
// correctly from any working directory, including from inside a debugger.
//
// The shader directory holds .slang source, not SPIR-V: nothing compiles shaders at
// build time. The samples link libslang and compile them when they start.
[[nodiscard]] std::filesystem::path shader_dir(std::string_view chapter_id);

[[nodiscard]] std::filesystem::path asset_path(std::string_view relative);

}  // namespace vkc
