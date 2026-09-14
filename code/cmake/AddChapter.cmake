# AddChapter.cmake
#
# Every tutorial sample is declared with a single call:
#
#   add_chapter(1.8.hello_triangle
#       SOURCES main.cpp
#       SHADERS triangle.vert triangle.frag)
#
# The target name is the chapter id with dots replaced by underscores, so
# `1.8.hello_triangle` builds `bin/1.8.hello_triangle` from target
# `ch_1_8_hello_triangle`. Shaders are compiled to SPIR-V next to the binary.

find_program(LVK_GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS "$ENV{VULKAN_SDK}/bin" "${CMAKE_CURRENT_LIST_DIR}/../out/build/${CMAKE_PRESET_NAME}/vcpkg_installed/x64-linux/tools/glslang"
    DOC "glslangValidator, used to compile GLSL to SPIR-V at build time")

function(_lvk_compile_shaders target chapter_id)
    if(NOT ARGN)
        return()
    endif()
    if(NOT LVK_GLSLANG_VALIDATOR)
        message(FATAL_ERROR
            "glslangValidator was not found. Install the Vulkan SDK or the glslang "
            "package and re-run CMake.")
    endif()

    set(spv_dir "${LVK_BINARY_DIR}/shaders/${chapter_id}")
    set(outputs "")

    foreach(shader IN LISTS ARGN)
        get_filename_component(shader_name "${shader}" NAME)
        set(spv "${spv_dir}/${shader_name}.spv")
        add_custom_command(
            OUTPUT "${spv}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${spv_dir}"
            COMMAND "${LVK_GLSLANG_VALIDATOR}" --target-env vulkan1.3 -g
                    -o "${spv}" "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
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
