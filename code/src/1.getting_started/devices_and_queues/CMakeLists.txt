# This chapter teaches the device, the queue and the allocator, so it writes all three
# by hand. It stands on 1.1's window and 1.2's instance, and on nothing else -- naming
# vkc_device here would defeat the point of the chapter, and leaving it out is what
# makes that a build error rather than a matter of discipline.
add_chapter(1.3.devices_and_queues
    SOURCES main.cpp
    SCAFFOLD vkc_window vkc_instance
)
