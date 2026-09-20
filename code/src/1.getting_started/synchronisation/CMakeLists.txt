# This chapter and 1.5 are together what becomes vkc::FrameContext, so neither may
# link vkc_frame. It stands on the same pieces 1.5 did: vkc::Context (1.1 to 1.3) and
# the swapchain (1.4).
add_chapter(1.6.synchronisation
    SOURCES main.cpp
    SCAFFOLD vkc_context vkc_swapchain
)
