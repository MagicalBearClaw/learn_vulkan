# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A LearnOpenGL-shaped Vulkan course: articles in `site/`, one runnable sample per
chapter in `code/src/`, shared scaffolding in `code/common/` (`vkcommon`).

## Non-negotiable rules

**Prose and code are original.** The chapter *sequence* is borrowed from
LearnOpenGL; the writing and the code are not. Never adapt LearnOpenGL's text
sentence by sentence, and never port its C++. Write each explanation from the Vulkan
specification and the graphics literature. Assets must be openly licensed (Khronos
glTF samples, CC0 textures) or drawn by this project; record every one in
`bootstrap.json` and `CREDITS.md`. `assets/textures/` is ours and is committed
(see `tools/make_textures.py`); the rest of `assets/` is fetched and gitignored.

**Verify claims about behaviour before writing them down.** Several article passages
assert what the validation layers say, or what a change looks like on screen. Run the
change and look. Two claims in M2 were written from reasoning and were wrong -- the
front-face winding in 1.13 and a validation message in 1.9 -- and both were caught only
by testing them.

**The vkcommon rule.** Nothing enters `code/common/` until the chapter that teaches it
has been written. Chapter N writes the code out in full and explains every line; from
chapter N+1 it lives in `vkcommon` and the article says so explicitly. A reader who
has followed the series in order must never meet a helper they have not built.

**Zero validation messages.** A sample that trips the validation layers is broken,
even if it renders correctly. The samples are what readers copy.

**Vulkan 1.3 baseline.** Dynamic rendering, `synchronization2`, descriptor indexing,
buffer device address. No `VkRenderPass`, no `VkFramebuffer`, no subpasses anywhere
except the appendix that explains why older code has them.

**Shaders are Slang.** One `.slang` file per pipeline, holding every stage, compiled by
`slangc` to one SPIR-V module. Entry points are `vertexMain` / `fragmentMain` /
`computeMain`, and the build passes `-fvk-use-entrypoint-name` so those names survive to
`pName`. Use `mul(M, v)` and never `*` for a transform; use `SV_VulkanVertexID` and never
`SV_VertexID`. GLSL appears only in the *Slang for GLSL programmers* appendix and where an
article contrasts the two.

## Commands

```bash
# Build (Linux, system packages). CMakePresets.json lives in code/, so the
# configure step must run from there; the build path is relative to it.
cd code && cmake --preset linux-debug
cmake --build code/out/build/linux-debug

# Build (Windows, vcpkg)
cd code && cmake --preset x64-debug
cmake --build code/out/build/x64-debug

# Run a sample
./code/out/build/linux-debug/bin/<chapter-id>

# If slangc is not on PATH or in the Vulkan SDK
cd code && cmake --preset linux-debug -DLVK_SLANGC=/path/to/slangc

# Scaffold a chapter
python3 tools/new_chapter.py <part>.<n>.<slug> --title "..." --part <part-slug>

# Screenshot regression across every chapter
python3 tools/capture.py --all

# Regenerate the committed test textures (rarely needed)
python3 tools/make_textures.py

# Site
cd site && npm run dev
```

## Conventions

- **Chapter ids** are `<part>.<n>.<slug>`, e.g. `1.7.hello_triangle`. The binary is
  named after the id; the CMake target is `ch_1_7_hello_triangle`. A chapter's shaders
  compile to `bin/shaders/<chapter-id>/<name>.slang.spv`.
- **Docs directories** carry no numeric prefix (`getting-started`, not
  `1.getting-started`) because Starlight strips dots from URLs. Ordering comes from
  the explicit sidebar in `site/astro.config.mjs` plus `sidebar.order` in frontmatter.
- **C++23**, `-Wall -Wextra -Wpedantic`, and the build is warning-free. Third-party
  implementation TUs (`vma_impl.cpp`, `stb_impl.cpp`) are compiled with warnings off
  and contain nothing else.
- **volk owns the entry points.** `VK_NO_PROTOTYPES` is set project-wide. Call
  `volkLoadDevice` immediately after `vkCreateDevice` - device-level pointers are null
  until then.
- Comments explain *why*, not *what*. The article explains what.

## Platform notes

- Development happens on Linux; Windows must keep building. Do not add MSVC-only or
  Linux-only code without guarding it.
- The repository path contains spaces, which breaks vcpkg's autotools ports. Linux
  uses system packages (`LVK_DEPS=system`); Windows uses vcpkg.
