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

**Shaders are Slang, compiled at run time.** One `.slang` file per pipeline, holding
every stage, compiled to one SPIR-V module by `libslang` when the sample starts. The
build only copies the source to `bin/shaders/<chapter-id>/`; there is no `.spv` artefact
and no `slangc` invocation anywhere. Entry points are `vertexMain` / `fragmentMain` /
`computeMain`.

Three session settings are load-bearing and easy to lose:
`VulkanUseEntryPointName` (or entry points get renamed to `main`),
`EmitSpirvDirectly`, and `defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR`
(the API defaults to row-major, which silently transposes every glm matrix).

Use `mul(M, v)` and never `*` for a transform; use `SV_VulkanVertexID` and never
`SV_VertexID`. GLSL appears only in the *Slang for GLSL programmers* appendix and where
an article contrasts the two.

## Commands

```bash
# Build (Linux). CMakePresets.json lives in code/, so the configure step must run
# from there; the build path is relative to it.
cd code && cmake --preset linux-debug
cmake --build code/out/build/linux-debug

# Build (Windows)
cd code && cmake --preset x64-debug
cmake --build code/out/build/x64-debug

# Run a sample
./code/out/build/linux-debug/bin/<chapter-id>

# Scaffold a chapter
python3 tools/new_chapter.py <part>.<n>.<slug> --title "..." --part <part-slug>

# Screenshot regression across every chapter
python3 tools/capture.py --all

# Regenerate the committed test textures (rarely needed)
python3 tools/make_textures.py

# Site
cd site && npm run dev

# Site build. `npm run build` is `astro check && astro build`, and astro check
# prompts to install @astrojs/check when it is missing -- which hangs forever in a
# non-interactive shell. This is the unattended form:
cd site && npx astro build < /dev/null
```

## Conventions

- **Chapter ids** are `<part>.<n>.<slug>`, e.g. `1.7.hello_triangle`. The binary is
  named after the id; the CMake target is `ch_1_7_hello_triangle`. A chapter's `.slang`
  sources are staged to `bin/shaders/<chapter-id>/` and compiled from there at run time.
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
- **vcpkg is the only source of libraries.** There is no system-packages mode and no
  `find_package` fallback anywhere: `code/vcpkg.json` is the manifest, every preset names
  the vcpkg toolchain, and every `find_package` in `Dependencies.cmake` is `REQUIRED`.
  The reason is reproducibility -- the articles quote validation messages and Slang
  output that a distro's version bump can invalidate, so the library set has to be
  pinned. A missing library is a vcpkg error, not something CMake works around.
- **Two presets on each platform.** `linux-debug` / `linux-release` and
  `x64-debug` / `x64-release`. All of them build **all 33 chapters**, Part 3 included.
  Verified by building every chapter warning-free and running `tools/capture.py --all`
  against the build tree: every chapter renders 0.00% different.
- **Part 3 is no longer gated.** Assimp was once optional, because it is large and slow
  to compile and the system-packages mode could not be relied on to have it; vcpkg builds
  it once and caches it, so `code/src/CMakeLists.txt` now adds the model-loading chapters
  unconditionally. `LVK_HAVE_ASSIMP` and `LVK_DEPS` are both gone. The scaffolder appends
  new `add_subdirectory` lines at the end of that file, which is now correct for every
  part.
- **Assimp exports stb_image's symbols.** Its static library defines all 43 `stbi_*`
  names, so a second global definition anywhere in this project makes every Part 3
  target fail to link with `multiple definition of stbi_load`. `stb_impl.cpp` therefore
  holds only `STB_IMAGE_WRITE_IMPLEMENTATION`; the two TUs that read images --
  vkcommon's `image.cpp` and chapter 1.11 -- compile their own copy with
  `STB_IMAGE_STATIC`. (4.4 used to be a third; it now loads its sky through
  `vkc::load_cubemap`.) stb is an `-isystem` include, so those TUs stay warning-free. Do
  not put a global `STB_IMAGE_IMPLEMENTATION` back.
- The repository path contains spaces. This was long recorded here as breaking vcpkg,
  and it does not, for this dependency set: the build tree and `vcpkg_installed` both
  live under the spaced path and install fine. The one autotools port that ever
  appeared was `libxcrypt`, reached through SDL3's `ibus` default feature, so
  `vcpkg.json` sets `"default-features": false` and asks only for `vulkan`, `wayland`
  and `x11`.
- **Slang comes from vcpkg's `shader-slang` port**, currently 2026.7.1. There used to be
  a second Slang -- a pinned official release fetched by CMake when the system build
  found none -- and the two versions were a standing hazard for any claim about Slang's
  output. There is now one.
- **VMA is pinned to 3.3.0** in `vcpkg.json`. 3.4.0 adds
  `VmaAllocationCreateInfo::minAlignment`, and the fully-designated initialisers in
  `buffer.cpp`, `image.cpp` and chapters 1.8 and 1.11 then warn about the missing field.
  Those initialisers are quoted in the articles, so the pin is cheaper than the edit.
