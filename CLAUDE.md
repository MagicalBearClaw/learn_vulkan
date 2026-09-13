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
glTF samples, CC0 textures); record every one in `bootstrap.json` and `CREDITS.md`.

**The vkcommon rule.** Nothing enters `code/common/` until the chapter that teaches it
has been written. Chapter N writes the code out in full and explains every line; from
chapter N+1 it lives in `vkcommon` and the article says so explicitly. A reader who
has followed the series in order must never meet a helper they have not built.

**Zero validation messages.** A sample that trips the validation layers is broken,
even if it renders correctly. The samples are what readers copy.

**Vulkan 1.3 baseline.** Dynamic rendering, `synchronization2`, descriptor indexing,
buffer device address. No `VkRenderPass`, no `VkFramebuffer`, no subpasses anywhere
except the appendix that explains why older code has them.

## Commands

```bash
# Build (Linux, system packages)
cmake --preset linux-debug
cmake --build code/out/build/linux-debug

# Build (Windows, vcpkg)
cmake --preset x64-debug
cmake --build code/out/build/x64-debug

# Run a sample
./code/out/build/linux-debug/bin/<chapter-id>

# Scaffold a chapter
python3 tools/new_chapter.py <part>.<n>.<slug> --title "..." --part <part-slug>

# Screenshot regression across every chapter
python3 tools/capture.py --all

# Site
cd site && npm run dev
```

## Conventions

- **Chapter ids** are `<part>.<n>.<slug>`, e.g. `1.8.hello_triangle`. The binary is
  named after the id; the CMake target is `ch_1_8_hello_triangle`.
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
