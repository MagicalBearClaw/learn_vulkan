# vkcommon - the teaching scaffold.
#
# Rule for this library: nothing goes in here until the chapter that teaches it has
# been written. Chapter N writes the code by hand and explains every line; from
# chapter N+1 onward the same code lives here and the prose says so explicitly.
# Nothing in vkcommon is ever magic to a reader who has followed the series in order.
#
# Currently holds what chapters 1.1 through 1.14 build:
#   window.cpp      1.1  Creating a window
#   context.cpp     1.2  The instance / 1.3  Devices and queues
#   swapchain.cpp   1.4  The swapchain
#   frame.cpp       1.5  Command buffers / 1.6  Synchronisation
#                   1.13 begin_rendering -- the colour and depth attachments
#   app.cpp         the loop those chapters share, plus screenshot support
#   pipeline.cpp    1.7  Hello triangle -- shader modules and the pipeline builder
#   buffer.cpp      1.8  Memory and buffers -- VMA buffers, staging, immediate submit
#   image.cpp       1.11 Textures -- images, views, mip generation, samplers
#                   1.13 choose_depth_format and create_depth_buffer
#   camera.cpp      1.14 Camera -- a first-person fly camera, no Vulkan in it at all

add_library(vkcommon STATIC
    src/window.cpp
    src/context.cpp
    src/swapchain.cpp
    src/frame.cpp
    src/app.cpp
    src/pipeline.cpp
    src/buffer.cpp
    src/image.cpp
    src/camera.cpp
)

target_include_directories(vkcommon PUBLIC include)
target_link_libraries(vkcommon PUBLIC vkbase)
