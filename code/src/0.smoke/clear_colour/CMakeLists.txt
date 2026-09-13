# Not a chapter: the build-system smoke test. If this runs, the toolchain, vcpkg
# dependencies, Vulkan driver and validation layers are all working, and any failure
# in a real chapter is the chapter's fault rather than the setup's.
add_chapter(0.smoke.clear_colour
    SOURCES main.cpp
)
