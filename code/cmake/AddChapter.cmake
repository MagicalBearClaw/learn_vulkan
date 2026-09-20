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

# How much of the scaffold a chapter is allowed to see. Exactly one of these applies:
#
#   NO_SCAFFOLD          Links vkbase only. Chapter 1.1 passes this: it is the first
#                        chapter and has nothing to stand on yet.
#
#   SCAFFOLD <targets>   Links only those pieces of vkcommon. This is how the rule
#                        "a reader never meets a helper they have not built" is
#                        enforced mechanically: a chapter that names
#                        `vkc_window vkc_instance` and then calls into vkc::Swapchain
#                        fails to link with an undefined reference. The headers share
#                        one include tree, so this bites at link time rather than at
#                        compile time -- which is still a build that does not pass.
#                        Chapters 1.2 to 1.7 each name what earlier chapters taught.
#
#   neither              Links the whole scaffold. Correct from 1.8 onward, by which
#                        point every piece has been written out and explained.
function(add_chapter chapter_id)
    cmake_parse_arguments(ARG "NO_SCAFFOLD" "" "SOURCES;SHADERS;LIBS;SCAFFOLD" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_chapter(${chapter_id}) requires SOURCES")
    endif()

    if(ARG_NO_SCAFFOLD AND ARG_SCAFFOLD)
        message(FATAL_ERROR
            "add_chapter(${chapter_id}): NO_SCAFFOLD and SCAFFOLD are mutually "
            "exclusive -- a chapter either stands on nothing or names what it stands on.")
    endif()

    string(REPLACE "." "_" target_suffix "${chapter_id}")
    set(target "ch_${target_suffix}")

    add_executable(${target} ${ARG_SOURCES})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${chapter_id}"
        FOLDER "chapters")

    if(ARG_NO_SCAFFOLD)
        target_link_libraries(${target} PRIVATE vkbase ${ARG_LIBS})
    elseif(ARG_SCAFFOLD)
        target_link_libraries(${target} PRIVATE ${ARG_SCAFFOLD} ${ARG_LIBS})
    else()
        target_link_libraries(${target} PRIVATE vkcommon ${ARG_LIBS})
    endif()

    # Each sample knows its own id so it can locate its shader sources at run time.
    target_compile_definitions(${target} PRIVATE LVK_CHAPTER_ID="${chapter_id}")

    _lvk_stage_shaders(${target} "${chapter_id}" ${ARG_SHADERS})

    set_property(GLOBAL APPEND PROPERTY LVK_CHAPTERS "${chapter_id}")
endfunction()
