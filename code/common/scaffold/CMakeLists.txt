# vkcommon - the teaching scaffold.
#
# Rule for this library: nothing goes in here until the chapter that teaches it has
# been written. Chapter N writes the code by hand and explains every line; from
# chapter N+1 onward the same code lives here and the prose says so explicitly.
# Nothing in vkcommon is ever magic to a reader who has followed the series in order.
#
# Currently holds what chapters 1.1 through 1.7 build:
#   window.cpp      1.1  Creating a window
#   context.cpp     1.2  The instance / 1.3  Devices and queues
#   swapchain.cpp   1.4  The swapchain
#   frame.cpp       1.5  Command buffers / 1.6  Synchronisation
#   app.cpp         the loop those chapters share, plus screenshot support

add_library(vkcommon STATIC
    src/window.cpp
    src/context.cpp
    src/swapchain.cpp
    src/frame.cpp
    src/app.cpp
)

target_include_directories(vkcommon PUBLIC include)
target_link_libraries(vkcommon PUBLIC vkbase)
