// The Vulkan Memory Allocator is a header-only library, which means its function
// bodies have to be compiled into exactly one translation unit somewhere in the
// program. This file is that one place, and it deliberately contains nothing else.

#include <volk.h>

// volk fetches the entry points at run time, so VMA must not expect the linker to
// resolve them statically; it uses the pointers handed to vmaCreateAllocator instead.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
