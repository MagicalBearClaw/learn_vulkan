# This chapter teaches the swapchain, so it writes the swapchain by hand. It stands on
# the window, the instance and the device that 1.1 to 1.3 taught -- and naming
# vkc_swapchain here would fail to link, which is the point.
add_chapter(1.4.the_swapchain
    SOURCES main.cpp
    SCAFFOLD vkc_window vkc_instance vkc_device
)
