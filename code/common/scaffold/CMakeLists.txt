# vkcommon - the teaching scaffold, in the order the reader builds it.
#
# Rule for this library: nothing goes in here until the chapter that teaches it has
# been written. Chapter N writes the code by hand and explains every line; from
# chapter N+1 onward the same code lives here and the prose says so explicitly.
# Nothing in vkcommon is ever magic to a reader who has followed the series in order.
#
# That rule is enforced by the build rather than by good intentions. The scaffold is
# not one library but one per subject, and a chapter's CMakeLists names exactly the
# pieces it is allowed to use:
#
#   add_chapter(1.4.the_swapchain
#       SOURCES main.cpp
#       SCAFFOLD vkc_window vkc_instance vkc_device)
#
# Chapter 1.4 teaches the swapchain, so it may use the window, instance and device
# that 1.1 to 1.3 taught -- and a call into vkc::Swapchain fails to link, with an
# undefined reference, because that piece is not among the libraries it named.
#
# The check is at link time rather than compile time: every piece shares one include
# tree, so the declaration is still visible. That is enough to enforce the rule -- such
# a chapter cannot be built -- and it keeps the header layout simple.
#
#   vkc_window      1.1  Creating a window
#   vkc_instance    1.2  The instance -- the instance, the layers, the messenger
#   vkc_device      1.3  Devices and queues -- physical device, logical device, the
#                        queue, and the VMA allocator that serves it
#   vkc_context          the facade holding an Instance, the Window's surface and a
#                        Device together, since from 1.4 every chapter wants all three
#   vkc_swapchain   1.4  The swapchain
#   vkc_frame       1.5  Command buffers / 1.6 Synchronisation
#                   1.13 begin_rendering -- the colour and depth attachments
#                   4.3  image_barrier -- transition_image with the masks spelled out
#   vkc_pipeline    1.7  Hello triangle -- shader modules and the pipeline builder
#   vkc_app              the frame loop those chapters share, plus screenshot support
#   vkc_buffer      1.8  Memory and buffers -- VMA buffers, staging, immediate submit
#   vkc_image       1.11 Textures -- images, views, mip generation, samplers
#                   1.13 choose_depth_format and create_depth_buffer
#   vkc_camera      1.14 Camera -- a first-person fly camera, no Vulkan in it at all

# The headers and the third-party dependencies every piece shares.
add_library(vkc_headers INTERFACE)
target_include_directories(vkc_headers INTERFACE include)
target_link_libraries(vkc_headers INTERFACE vkbase)

function(_vkc_piece name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})
    add_library(${name} STATIC ${ARG_SOURCES})
    target_link_libraries(${name} PUBLIC vkc_headers ${ARG_DEPENDS})
    set_target_properties(${name} PROPERTIES FOLDER "vkcommon")
endfunction()

_vkc_piece(vkc_window    SOURCES src/window.cpp)
_vkc_piece(vkc_instance  SOURCES src/instance.cpp  DEPENDS vkc_window)
_vkc_piece(vkc_device    SOURCES src/device.cpp    DEPENDS vkc_instance)
_vkc_piece(vkc_context   SOURCES src/context.cpp   DEPENDS vkc_device vkc_window)
_vkc_piece(vkc_swapchain SOURCES src/swapchain.cpp DEPENDS vkc_context)
_vkc_piece(vkc_frame     SOURCES src/frame.cpp     DEPENDS vkc_swapchain)
_vkc_piece(vkc_app       SOURCES src/app.cpp       DEPENDS vkc_frame)

# Neither of these touches anything above: pipeline.cpp needs only VK_CHECK and the
# shader search path, and camera.cpp contains no Vulkan at all.
_vkc_piece(vkc_pipeline  SOURCES src/pipeline.cpp)
_vkc_piece(vkc_camera    SOURCES src/camera.cpp)

_vkc_piece(vkc_buffer    SOURCES src/buffer.cpp    DEPENDS vkc_context)
_vkc_piece(vkc_image     SOURCES src/image.cpp     DEPENDS vkc_buffer)

# The whole scaffold, for chapters past the point where all of it has been taught.
# add_chapter() links this when a chapter names no SCAFFOLD of its own.
add_library(vkcommon INTERFACE)
target_link_libraries(vkcommon
    INTERFACE
        vkc_window
        vkc_instance
        vkc_device
        vkc_context
        vkc_swapchain
        vkc_frame
        vkc_app
        vkc_pipeline
        vkc_buffer
        vkc_image
        vkc_camera
)
