# LearnVulkan

A complete, beginner-oriented course in modern Vulkan, built the way
[LearnOpenGL](https://learnopengl.com) is built: every chapter continues the program
from the one before it, and the series runs all the way from opening a window to
physically based rendering with image-based lighting.

Targets **Vulkan 1.3** with dynamic rendering, `synchronization2`, descriptor indexing
and buffer device address - no render pass objects, no framebuffer objects, and
roughly a third less setup code than a 1.0-era tutorial.

See [`CREDITS.md`](CREDITS.md) for the debt this owes to LearnOpenGL, and the
licences of everything it depends on.

## Layout

```
site/       The articles (Astro + Starlight)
code/       The accompanying samples, one directory per chapter
  common/   vkcommon - shared code, but only ever code a previous chapter taught
tools/      Asset fetching, chapter scaffolding, screenshot regression
assets/     Downloaded by tools/bootstrap.py; not committed
```

## Building the code

### Linux

Dependencies come from the distribution. On Arch and derivatives:

```bash
sudo pacman -S cmake ninja clang vulkan-headers vulkan-validation-layers \
               volk sdl3 glm spdlog glslang
```

VMA and stb are header-only and are fetched automatically by CMake if they are not
already installed.

```bash
cd code
cmake --preset linux-debug
cmake --build out/build/linux-debug
./out/build/linux-debug/bin/0.smoke.clear_colour
```

### Windows

Needs Visual Studio 2022 with the C++ workload, the Vulkan SDK, and vcpkg with
`VCPKG_ROOT` set.

```powershell
cd code
cmake --preset x64-debug
cmake --build out/build/x64-debug
```

> **vcpkg and spaces:** several vcpkg ports build through autotools, which mishandles
> paths containing spaces. Keep the checkout somewhere without them. The Linux presets
> avoid vcpkg entirely for this reason; `linux-vcpkg-debug` exists if you want it
> anyway.

### Assets

Part 1 needs nothing downloaded: the test texture the chapters from 1.11 onward use is
drawn by this project and committed under `assets/textures/`.

```bash
python3 tools/bootstrap.py          # fetch what the later chapters need
python3 tools/bootstrap.py --list   # see what that is
python3 tools/make_textures.py      # redraw the committed test textures
```

## Building the site

```bash
cd site
npm install
npm run dev      # local preview
npm run build    # production build into site/dist
```

## Working on a chapter

```bash
# scaffold the article, the code directory and the build entry
python3 tools/new_chapter.py 2.1.colours \
    --title "Colours" --part lighting \
    --shaders lit.vert lit.frag

(cd code && cmake --preset linux-debug)   # pick up the new target
cmake --build code/out/build/linux-debug

# capture the reference screenshot once the sample looks right
python3 tools/capture.py --chapter 2.1 --update
```

Every sample accepts `--frames N`, `--screenshot PATH`, `--width`, `--height` and
`--no-validation`.

## Checks

Validation layers are on by default in debug builds, and **a sample that produces any
validation message is considered broken** - the samples are what people copy, so they
have to be correct, not merely working.

```bash
# render every chapter and diff against its reference image
python3 tools/capture.py --all
```

This is what catches a `vkcommon` refactor silently changing what an earlier chapter
renders.
