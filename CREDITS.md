# Credits

## LearnOpenGL

This series follows the chapter structure, topic ordering and teaching approach of
**[LearnOpenGL](https://learnopengl.com)** by **Joey de Vries**
([@JoeyDeVriez](https://twitter.com/JoeyDeVriez)). LearnOpenGL is the best
introduction to real-time graphics that exists, and the reason it works is the order
it puts things in. That order is what has been borrowed here.

What is *not* borrowed: the prose and the code. Every article in this series is
written from scratch against the Vulkan specification, and every line of C++ and Slang
here is new. Vulkan is different enough from OpenGL that adapting sentence by
sentence would produce worse explanations than writing them fresh.

Where a figure from LearnOpenGL is used, it is reproduced under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with attribution to Joey de
Vries in the caption, as that licence requires.

## Assets

These textures are this project's own work, drawn by `tools/make_textures.py` and
committed so that a fresh clone can run the texture chapters without fetching anything:

| File | Used from | What it is |
|---|---|---|
| `assets/textures/lvk_grid.png` | 1.11 | Test grid: checkerboard, fine lines, orientation marker, border |
| `assets/textures/lvk_crate_diffuse.png` | 2.4 | Crate diffuse map: brushed-steel frame and rivets around wooden planks |
| `assets/textures/lvk_crate_specular.png` | 2.4 | The matching specular map: bright steel, near-black wood |
| `assets/textures/lvk_foliage.png` | 4.2 | Fern cutout: an intricate alpha silhouette, with leaf colour bled into every transparent texel |
| `assets/textures/lvk_ground.png` | 4.2 | Flagstone ground: deliberately low-contrast, for scenes where the floor is not the subject |

Regenerate them with:

```bash
python3 tools/make_textures.py
```

Everything else under `assets/` is downloaded by `tools/bootstrap.py` from the sources
listed in `bootstrap.json`, and each entry there records its licence.

### Models

Both come from the Khronos
[glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository, and
`bootstrap.json` fetches them file by file from a pinned commit rather than cloning it —
the repository is several gigabytes and these chapters use two models from it.

| Model | Used from | Author | Licence |
|---|---|---|---|
| Damaged Helmet | 3.1 | [ctxwing](https://github.com/ctxwing), 2018 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Flight Helmet | 3.3 | Gary Hsu, 2018 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

Damaged Helmet is ctxwing's rebuild and glTF conversion of an earlier model by
theblueturtle\_. The two are licensed separately: the rebuild — the `.glb` fetched here
and the only version this series uses — is CC BY 4.0, while theblueturtle\_'s earlier
version is CC BY-NC 4.0 and is not used.

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
| [Slang](https://github.com/shader-slang/slang) | Shading language; linked as a library and used to compile shaders at startup | Apache-2.0 with LLVM exception |

## Reference material

- The [Vulkan specification](https://registry.khronos.org/vulkan/) and
  [Vulkan Documentation Project](https://docs.vulkan.org/), Khronos Group
- [Vulkan Guide](https://vkguide.dev/), Victor Blanco
- [Sascha Willems' Vulkan samples](https://github.com/SaschaWillems/Vulkan)
- [How to Vulkan in 2026](https://howtovulkan.com/), Sascha Willems
