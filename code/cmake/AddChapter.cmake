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
# `ch_1_7_hello_triangle`.
#
# Shaders are written in Slang and are NOT compiled by the build. The samples link
# libslang and compile them at startup, so all this does is put the .slang source
# where the running binary can find it: bin/shaders/<chapter-id>/. Editing a shader
# and re-running is enough; there is no SPIR-V artefact anywhere.

function(_lvk_stage_shaders target chapter_id)
    if(NOT ARGN)
        return()
    endif()

    set(shader_dir "${LVK_BINARY_DIR}/shaders/${chapter_id}")
    set(outputs "")

    foreach(shader IN LISTS ARGN)
        get_filename_component(shader_name "${shader}" NAME)
        set(staged "${shader_dir}/${shader_name}")
        add_custom_command(
            OUTPUT "${staged}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_dir}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" "${staged}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            COMMENT "Staging ${chapter_id}/${shader_name}"
            VERBATIM)
        list(APPEND outputs "${staged}")
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

    # Each sample knows its own id so it can locate its shader sources at run time.
    target_compile_definitions(${target} PRIVATE LVK_CHAPTER_ID="${chapter_id}")

    _lvk_stage_shaders(${target} "${chapter_id}" ${ARG_SHADERS})

    set_property(GLOBAL APPEND PROPERTY LVK_CHAPTERS "${chapter_id}")
endfunction()
