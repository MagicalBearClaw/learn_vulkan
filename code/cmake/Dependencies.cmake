# Dependencies.cmake
#
# Two supported ways to get the libraries, because the two platforms genuinely want
# different things:
#
#   LVK_DEPS=system   find_package against libraries already installed on the
#                     machine. The normal choice on Linux, where the distribution
#                     ships current SDL3, glm, spdlog, volk and the Vulkan headers.
#   LVK_DEPS=vcpkg    let the vcpkg toolchain provide everything. The normal choice
#                     on Windows, where nothing is installed by default.
#
# Either way the rest of the build sees the same imported targets, so no sample or
# CMakeLists anywhere else has to care which mode is active.
#
# Note for vcpkg users: several vcpkg ports build through autotools, which mishandles
# paths containing spaces. Keep the checkout somewhere without them.

if(WIN32)
    set(_lvk_default_deps "vcpkg")
else()
    set(_lvk_default_deps "system")
endif()
set(LVK_DEPS "${_lvk_default_deps}" CACHE STRING "Where to get dependencies: system or vcpkg")
set_property(CACHE LVK_DEPS PROPERTY STRINGS system vcpkg)

message(STATUS "Dependency mode: ${LVK_DEPS}")

find_package(VulkanHeaders CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include(FetchContent)

# --- Vulkan Memory Allocator -------------------------------------------------
# Header-only, and not packaged by most distributions. Fetch it if it is not
# already available so a Linux clone needs nothing beyond the distro packages.
find_package(VulkanMemoryAllocator CONFIG QUIET)
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    message(STATUS "VulkanMemoryAllocator not found locally; fetching it")
    FetchContent_Declare(vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.3.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(vma)
endif()

# --- stb ---------------------------------------------------------------------
# stb_image / stb_image_write, used for loading textures and writing screenshots.
find_path(Stb_INCLUDE_DIR NAMES stb_image.h PATH_SUFFIXES stb)
if(NOT Stb_INCLUDE_DIR)
    message(STATUS "stb not found locally; fetching it")
    FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG f58f558c120e9b32c217290b80bad1a0729fbb2c
        GIT_SHALLOW FALSE)
    FetchContent_MakeAvailable(stb)
    set(Stb_INCLUDE_DIR "${stb_SOURCE_DIR}" CACHE PATH "stb headers" FORCE)
endif()

# VMA's headers use Clang nullability annotations that -Wpedantic objects to. It is
# third-party code, so include it as SYSTEM and let its own authors' warnings be
# their business rather than ours.
if(TARGET VulkanMemoryAllocator)
    set_target_properties(VulkanMemoryAllocator PROPERTIES SYSTEM ON)
endif()

add_library(lvk_stb INTERFACE)
target_include_directories(lvk_stb SYSTEM INTERFACE "${Stb_INCLUDE_DIR}")
