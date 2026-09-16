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

# --- Slang ---------------------------------------------------------------------
# The shading language. Unlike VMA and stb this is not header-only: the samples link
# libslang and call its compiler at startup to turn .slang into SPIR-V, so the library
# has to be here at build time and beside the binary at run time.
#
# vcpkg's shader-slang port provides it on Windows. On Linux most distributions do not
# package it, so fall back to the official release build -- the same archive a human
# would download, pinned and hash-checked.
set(LVK_SLANG_VERSION "2026.17.1" CACHE STRING "Slang release to fetch if none is installed")

find_package(slang CONFIG QUIET)
if(NOT TARGET slang::slang)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_slang_archive "slang-${LVK_SLANG_VERSION}-linux-x86_64.tar.gz")
        set(_slang_hash "SHA512=2d8839a1933de8720cf66682ddfc2f69463e351bceff8c66ba2376cf056bd10cd122f17f01e4856ecfa7cec61f5604dbecc1c6caae6af03f6cb0357c75f7a803")
    elseif(WIN32)
        set(_slang_archive "slang-${LVK_SLANG_VERSION}-windows-x86_64.zip")
        set(_slang_hash "SHA512=6ca46a6e920596b2818d870663ea1c08c1ba0c40e600a6e6faf38784e3750f1a97d550f1c2b36fa0e2458166b9d128783cf0c413ecb6cd9cf98a675f3f92e33b")
    else()
        message(FATAL_ERROR
            "No prebuilt Slang is configured for this platform. Install Slang and make "
            "sure find_package(slang CONFIG) locates it, or point CMAKE_PREFIX_PATH at "
            "an unpacked release from https://github.com/shader-slang/slang/releases.")
    endif()

    message(STATUS "Slang not found locally; fetching ${LVK_SLANG_VERSION}")
    FetchContent_Declare(slang_bindist
        URL "https://github.com/shader-slang/slang/releases/download/v${LVK_SLANG_VERSION}/${_slang_archive}"
        URL_HASH "${_slang_hash}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(slang_bindist)

    # The archive ships its own CMake package, so the rest of the build sees exactly
    # the same slang::slang target either way.
    find_package(slang CONFIG REQUIRED
        PATHS "${slang_bindist_SOURCE_DIR}/lib/cmake/slang"
              "${slang_bindist_SOURCE_DIR}/cmake"
        NO_DEFAULT_PATH)
endif()

# --- Assimp --------------------------------------------------------------------
# Model loading, from chapter 3.1 onward.
#
# This one gets no FetchContent fallback, unlike VMA, stb and Slang. Assimp is a large
# C++ library that takes minutes to compile from source and drags in its own
# compression dependencies, and every platform this series targets already packages
# it -- so a clear message beats a long build.
# Not fatal when missing: only the Part 3 chapters need it, and the rest of the series
# should still configure and build without it. LVK_HAVE_ASSIMP is what those chapters
# will be gated on once they exist; nothing reads it yet.
find_package(assimp CONFIG QUIET)
if(TARGET assimp::assimp)
    set(LVK_HAVE_ASSIMP TRUE)
else()
    set(LVK_HAVE_ASSIMP FALSE)
    message(WARNING
        "Assimp was not found, so the model-loading chapters (3.1 onward) will be "
        "skipped. Everything else builds as usual.\n"
        "To get them, either configure with vcpkg, which installs nothing "
        "system-wide:\n"
        "    cmake --preset linux-vcpkg-debug\n"
        "or install assimp for the system build (LVK_DEPS=system):\n"
        "  Arch / CachyOS:   sudo pacman -S assimp\n"
        "  Debian / Ubuntu:  sudo apt install libassimp-dev\n"
        "  Fedora:           sudo dnf install assimp-devel")
endif()

# VMA's headers use Clang nullability annotations that -Wpedantic objects to. It is
# third-party code, so include it as SYSTEM and let its own authors' warnings be
# their business rather than ours.
if(TARGET VulkanMemoryAllocator)
    set_target_properties(VulkanMemoryAllocator PROPERTIES SYSTEM ON)
endif()

add_library(lvk_stb INTERFACE)
target_include_directories(lvk_stb SYSTEM INTERFACE "${Stb_INCLUDE_DIR}")
