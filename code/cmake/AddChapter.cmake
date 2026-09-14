# AddChapter.cmake
#
# Every tutorial sample is declared with a single call:
#
#   add_chapter(1.7.hello_triangle
#       SOURCES main.cpp
#       SHADERS triangle.slang)
#
# The target name is the chapter id with dots replaced by underscores, so
# `1.7.hello_triangle` builds `bin/1.7.hello_triangle` from target
# `ch_1_7_hello_triangle`. Shaders are compiled to SPIR-V next to the binary.
#
# Shaders are written in Slang. One .slang file holds every stage of a pipeline and
# compiles to one SPIR-V module with one entry point per stage, so a chapter lists a
# single shader file where a GLSL project would list a .vert and a .frag.

find_program(LVK_SLANGC
    NAMES slangc slangc.exe
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{SLANG_ROOT}/bin"
        "${CMAKE_CURRENT_LIST_DIR}/../out/build/${CMAKE_PRESET_NAME}/vcpkg_installed/x64-linux/tools/shader-slang"
        "${CMAKE_CURRENT_LIST_DIR}/../out/build/${CMAKE_PRESET_NAME}/vcpkg_installed/x64-windows/tools/shader-slang"
    DOC "slangc, used to compile Slang to SPIR-V at build time")

# Slang targets a SPIR-V version, not a Vulkan one. SPIR-V 1.6 is the version Vulkan
# 1.3 consumes, and 1.3 is this project's baseline.
set(LVK_SPIRV_PROFILE "spirv_1_6")

function(_lvk_compile_shaders target chapter_id)
    if(NOT ARGN)
        return()
    endif()
    if(NOT LVK_SLANGC)
        message(FATAL_ERROR
            "slangc was not found, so the shaders cannot be compiled.\n"
            "  Install it with one of:\n"
            "    - the Vulkan SDK 1.3.296 or newer, which bundles slangc\n"
            "    - a release from https://github.com/shader-slang/slang/releases\n"
            "    - your distribution's shader-slang package\n"
            "  Then re-run CMake, or point it at the binary directly with\n"
            "    cmake --preset <preset> -DLVK_SLANGC=/path/to/slangc")
    endif()

    set(spv_dir "${LVK_BINARY_DIR}/shaders/${chapter_id}")
    set(outputs "")

    foreach(shader IN LISTS ARGN)
        get_filename_component(shader_name "${shader}" NAME)
        set(spv "${spv_dir}/${shader_name}.spv")
        add_custom_command(
            OUTPUT "${spv}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${spv_dir}"
            # -emit-spirv-directly skips the intermediate GLSL that older Slang
            # versions went through; -fvk-use-entrypoint-name keeps the SPIR-V entry
            # points named after the Slang functions instead of renaming them to
            # "main", which is what lets one module hold both stages; -g2 emits the
            # debug info RenderDoc needs to show Slang source.
            COMMAND "${LVK_SLANGC}" "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                    -target spirv -profile ${LVK_SPIRV_PROFILE}
                    -emit-spirv-directly -fvk-use-entrypoint-name -g2
                    -o "${spv}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            COMMENT "Compiling ${chapter_id}/${shader_name} to SPIR-V"
            VERBATIM)
        list(APPEND outputs "${spv}")
    endforeach()

    add_custom_target(${target}_shaders DEPENDS ${outputs})
    add_dependencies(${target} ${target}_shaders)
endfunction()

# NO_SCAFFOLD marks a chapter that builds Vulkan by hand rather than using vkcommon.
# Chapters 1.1 to 1.7 all pass it: they are the chapters that write the scaffold, so
# linking it would be circular and would hide the very code the article is about.
function(add_chapter chapter_id)
    cmake_parse_arguments(ARG "NO_SCAFFOLD" "" "SOURCES;SHADERS;LIBS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_chapter(${chapter_id}) requires SOURCES")
    endif()

    string(REPLACE "." "_" target_suffix "${chapter_id}")
    set(target "ch_${target_suffix}")

    add_executable(${target} ${ARG_SOURCES})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${chapter_id}"
        FOLDER "chapters")

    if(ARG_NO_SCAFFOLD)
        target_link_libraries(${target} PRIVATE vkbase ${ARG_LIBS})
    else()
        target_link_libraries(${target} PRIVATE vkcommon ${ARG_LIBS})
    endif()

    # Each sample knows its own id so it can locate its compiled shaders.
    target_compile_definitions(${target} PRIVATE LVK_CHAPTER_ID="${chapter_id}")

    _lvk_compile_shaders(${target} "${chapter_id}" ${ARG_SHADERS})

    set_property(GLOBAL APPEND PROPERTY LVK_CHAPTERS "${chapter_id}")
endfunction()
