# This chapter teaches shader modules and the graphics pipeline, so it writes both by
# hand -- the Slang compiler calls and all eleven pipeline structs. It stands on
# everything 1.1 to 1.6 taught, including the frame loop that is now vkc::FrameContext.
# vkc_pipeline, which is what this chapter becomes, is deliberately not named.
add_chapter(1.7.hello_triangle
    SOURCES main.cpp
    SHADERS triangle.slang
    SCAFFOLD vkc_context vkc_swapchain vkc_frame
)
