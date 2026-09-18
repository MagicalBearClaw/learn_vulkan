# One add_subdirectory per chapter that ships code, in the order the reader meets
# them. tools/new_chapter.py appends to this list.

add_subdirectory(0.smoke/clear_colour)

add_subdirectory(1.getting_started/creating_a_window)
add_subdirectory(1.getting_started/the_instance)
add_subdirectory(1.getting_started/devices_and_queues)
add_subdirectory(1.getting_started/the_swapchain)
add_subdirectory(1.getting_started/command_buffers)
add_subdirectory(1.getting_started/synchronisation)
add_subdirectory(1.getting_started/hello_triangle)
add_subdirectory(1.getting_started/memory_and_buffers)
add_subdirectory(1.getting_started/shaders)
add_subdirectory(1.getting_started/descriptors)
add_subdirectory(1.getting_started/textures)
add_subdirectory(1.getting_started/transformations)
add_subdirectory(1.getting_started/coordinate_systems)
add_subdirectory(1.getting_started/camera)
add_subdirectory(2.lighting/colours)
add_subdirectory(2.lighting/basic_lighting)
add_subdirectory(2.lighting/materials)
add_subdirectory(2.lighting/lighting_maps)
add_subdirectory(2.lighting/light_casters)
add_subdirectory(2.lighting/multiple_lights)

add_subdirectory(3.model_loading/assimp)
add_subdirectory(3.model_loading/mesh)
add_subdirectory(3.model_loading/model)
add_subdirectory(3.model_loading/bindless_materials)


add_subdirectory(4.advanced_vulkan/depth_and_stencil)
add_subdirectory(4.advanced_vulkan/blending_and_culling)
add_subdirectory(4.advanced_vulkan/offscreen_rendering)
