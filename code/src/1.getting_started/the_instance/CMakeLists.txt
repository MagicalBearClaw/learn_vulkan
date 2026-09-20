# This chapter teaches the instance, so it writes the instance by hand. It stands on
# chapter 1.1's window, which is now vkc_window -- and on nothing else: a call into
# vkc::Device or vkc::Swapchain from here would fail to link.
add_chapter(1.2.the_instance
    SOURCES main.cpp
    SCAFFOLD vkc_window
)
