# This chapter teaches command pools, command buffers, layout transitions and the
# frame, so it writes all of them by hand. It stands on 1.1 to 1.4: the window, the
# instance and the device arrive together as vkc::Context, and the swapchain is 1.4's.
# vkc_frame -- which is what this chapter and 1.6 become -- is deliberately not named.
add_chapter(1.5.command_buffers
    SOURCES main.cpp
    SCAFFOLD vkc_context vkc_swapchain
)
