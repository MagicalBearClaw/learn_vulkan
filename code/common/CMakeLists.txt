# The shared code is deliberately in two pieces, and the split is pedagogical.
#
#   vkbase     Third-party dependencies and the two utilities that are not Vulkan
#              lessons in themselves (VK_CHECK, file paths). Every chapter links
#              this, including the very first.
#
#   vkcommon   The teaching scaffold: window, context, swapchain, frame loop. A
#              chapter may only link this once the series has taught what is in it.
#              Chapters 1.1 to 1.7 build all of it by hand instead, which is the
#              entire point of those chapters.
add_subdirectory(base)
add_subdirectory(scaffold)
