# Dependencies.cmake
#
# One source for every library: vcpkg. `code/vcpkg.json` is the manifest, the toolchain
# file named by every preset installs those libraries into the build tree, and nothing
# is ever taken from the machine's own package manager.
#
# That is deliberate. A distribution package is a different version on every developer's
# machine, and this series' articles quote behaviour -- validation messages, Slang
# codegen, struct fields that exist in one release and not the last -- that a version
# change can invalidate. One pinned set of libraries is the only way the prose and the
# code can be checked against each other and stay checked.
#
# So every find_package here is REQUIRED, and there are no fallbacks. A library that is
# missing is a vcpkg problem, reported by vcpkg, rather than something for this file to
# paper over.

find_package(VulkanHeaders CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# Model loading, from chapter 3.1 onward. Large and slow to build from source, which is
# why it was once optional; vcpkg builds it once and caches it, so it is now simply a
# dependency like any other and Part 3 is never skipped.
find_package(assimp CONFIG REQUIRED)

# The shading language. Not header-only, unlike most of the list: the samples link
# libslang and call its compiler at startup to turn .slang into SPIR-V, so it has to be
# present at build time and beside the binary at run time.
find_package(slang CONFIG REQUIRED)

# stb_image / stb_image_write, for loading textures and writing screenshots. vcpkg's stb
# port installs headers under include/stb and ships no CMake config package, so this is a
# find_path rather than a find_package.
find_path(Stb_INCLUDE_DIR NAMES stb_image.h PATH_SUFFIXES stb REQUIRED)

# VMA's headers use Clang nullability annotations that -Wpedantic objects to. It is
# third-party code, so include it as SYSTEM and let its own authors' warnings be
# their business rather than ours.
if(TARGET VulkanMemoryAllocator)
    set_target_properties(VulkanMemoryAllocator PROPERTIES SYSTEM ON)
endif()

add_library(lvk_stb INTERFACE)
target_include_directories(lvk_stb SYSTEM INTERFACE "${Stb_INCLUDE_DIR}")
