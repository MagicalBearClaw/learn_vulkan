# vkcommon — the teaching scaffold.
#
# Rule for this library: nothing goes in here until the chapter that teaches it has
# been written. Chapter N writes the code by hand and explains every line; from
# chapter N+1 onward the same code lives here and the prose says so explicitly.
# Nothing in vkcommon is ever magic to a reader who has followed the series in order.

add_library(vkcommon STATIC
    src/check.cpp
    src/window.cpp
    src/context.cpp
    src/swapchain.cpp
    src/frame.cpp
    src/app.cpp
    src/paths.cpp
    src/vma_impl.cpp
    src/stb_impl.cpp
)

target_include_directories(vkcommon PUBLIC include)

target_link_libraries(vkcommon
    PUBLIC
        Vulkan::Headers
        volk::volk
        GPUOpen::VulkanMemoryAllocator
        SDL3::SDL3
        glm::glm
        spdlog::spdlog
        lvk_stb
)

# volk owns the Vulkan entry points, so the static prototypes must stay out of the
# headers. Every translation unit in the project needs this, hence PUBLIC.
target_compile_definitions(vkcommon
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
    target_compile_definitions(vkcommon PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
endif()

if(MSVC)
    target_compile_options(vkcommon PUBLIC /W4 /permissive-)
else()
    target_compile_options(vkcommon PUBLIC -Wall -Wextra -Wpedantic)
endif()

# VMA and stb are third-party sources compiled into this target. They are not held to
# our warning settings, and the noise would bury warnings in our own code.
if(MSVC)
    set_source_files_properties(src/vma_impl.cpp src/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(src/vma_impl.cpp src/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()
