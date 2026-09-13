# Credits

## LearnOpenGL

This series follows the chapter structure, topic ordering and teaching approach of
**[LearnOpenGL](https://learnopengl.com)** by **Joey de Vries**
([@JoeyDeVriez](https://twitter.com/JoeyDeVriez)). LearnOpenGL is the best
introduction to real-time graphics that exists, and the reason it works is the order
it puts things in. That order is what has been borrowed here.

What is *not* borrowed: the prose and the code. Every article in this series is
written from scratch against the Vulkan specification, and every line of C++ and GLSL
here is new. Vulkan is different enough from OpenGL that adapting sentence by
sentence would produce worse explanations than writing them fresh.

Where a figure from LearnOpenGL is used, it is reproduced under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to Joey de
Vries in the caption, as that licence requires.

## Libraries

| Library | Used for | Licence |
|---|---|---|
| [SDL3](https://libsdl.org) | Windowing, input, audio | Zlib |
| [volk](https://github.com/zeux/volk) | Vulkan entry-point loading | MIT |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory allocation | MIT |
| [glm](https://github.com/g-truc/glm) | Vector and matrix maths | MIT |
| [spdlog](https://github.com/gabime/spdlog) | Logging | MIT |
| [stb](https://github.com/nothings/stb) | Image loading and writing | MIT / public domain |
| [Dear ImGui](https://github.com/ocornut/imgui) | Debug UI (later chapters) | MIT |
| [Assimp](https://github.com/assimp/assimp) | Model loading (later chapters) | BSD-3-Clause |

## Reference material

- The [Vulkan specification](https://registry.khronos.org/vulkan/) and
  [Vulkan Documentation Project](https://docs.vulkan.org/), Khronos Group
- [Vulkan Guide](https://vkguide.dev/), Victor Blanco
- [Sascha Willems' Vulkan samples](https://github.com/SaschaWillems/Vulkan)
- [How to Vulkan in 2026](https://howtovulkan.com/), Sascha Willems
