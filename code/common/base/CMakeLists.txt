# vkbase - dependencies plus the handful of utilities that are not lessons.
#
# Nothing in here teaches Vulkan. VK_CHECK is error handling, paths.cpp is file
# system plumbing, and the two *_impl.cpp files exist only because VMA and stb are
# header-only. Chapter 1.1 links this on day one without owing the reader anything.

add_library(vkbase STATIC
    src/check.cpp
    src/paths.cpp
    src/capture.cpp
    src/vma_impl.cpp
    src/stb_impl.cpp
)

target_include_directories(vkbase PUBLIC include)

target_link_libraries(vkbase
    PUBLIC
        Vulkan::Headers
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        SDL3::SDL3
        glm::glm
        spdlog::spdlog
        lvk_stb
        # Slang is a link-time dependency because the samples compile their shaders
        # at startup rather than at build time. Chapter 1.7 calls this API by hand;
        # from 1.8 it is behind vkc::load_shader.
        slang::slang
)

# Slang's headers are third-party and are not written to our warning settings.
get_target_property(_slang_includes slang::slang INTERFACE_INCLUDE_DIRECTORIES)
if(_slang_includes)
    target_include_directories(vkbase SYSTEM PUBLIC ${_slang_includes})
endif()

# volk owns the Vulkan entry points, so the static prototypes must stay out of the
# headers. Every translation unit in the project needs this, hence PUBLIC.
target_compile_definitions(vkbase
    PUBLIC
        VK_NO_PROTOTYPES
        GLM_FORCE_DEPTH_ZERO_TO_ONE   # Vulkan clip space is z in [0,1], not [-1,1]
        GLM_FORCE_RADIANS
        GLM_ENABLE_EXPERIMENTAL
    PRIVATE
        LVK_ASSET_DIR="${LVK_ASSET_DIR}"
        LVK_SHADER_DIR="${LVK_BINARY_DIR}/shaders"
)

if(WIN32)
    target_compile_definitions(vkbase PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
endif()

if(MSVC)
    target_compile_options(vkbase PUBLIC /W4 /permissive-)
else()
    target_compile_options(vkbase PUBLIC -Wall -Wextra -Wpedantic)
endif()

# VMA and stb are third-party sources compiled into this target. They are not held to
# our warning settings, and the noise would bury warnings in our own code.
if(MSVC)
    set_source_files_properties(src/vma_impl.cpp src/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(src/vma_impl.cpp src/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()
