#include "vkc/paths.hpp"

namespace vkc {

std::filesystem::path shader_dir(std::string_view chapter_id) {
    return std::filesystem::path(LVK_SHADER_DIR) / chapter_id;
}

std::filesystem::path asset_path(std::string_view relative) {
    return std::filesystem::path(LVK_ASSET_DIR) / relative;
}

}  // namespace vkc
