#!/usr/bin/env python3
"""Scaffold a new chapter: the article page, the code directory, and the build entry.

    python3 tools/new_chapter.py 1.7.hello_triangle --title "Hello Triangle" --part 1.getting-started --shaders triangle.slang

Creates:
    site/src/content/docs/<part>/<slug>.mdx
    code/src/<part-dir>/<slug>/{main.cpp,CMakeLists.txt,shaders}
and appends the add_subdirectory line to code/src/CMakeLists.txt.

Chapter ids are <part>.<number>.<slug>, which is what orders them in the sidebar and
names the binary. The number is the position within the part, not globally.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_SRC = ROOT / "code" / "src"
DOCS = ROOT / "site" / "src" / "content" / "docs"

MAIN_TEMPLATE = '''// {title}
//
// Chapter {chapter_id} of the LearnVulkan series. The article that walks through this
// file is at /{part}/{slug}.

#include <vkc/app.hpp>

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>

namespace {{

class {class_name} : public vkc::App {{
public:
    using vkc::App::App;

protected:
    void on_render(const vkc::FrameInfo& frame) override {{
        const VkRenderingAttachmentInfo colour{{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = frame.view,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {{.color = {{{{0.05F, 0.05F, 0.08F, 1.0F}}}}}},
        }};

        const VkRenderingInfo rendering{{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {{{{0, 0}}, frame.extent}},
            .layerCount = 1,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colour,
            .pDepthAttachment = nullptr,
            .pStencilAttachment = nullptr,
        }};

        vkCmdBeginRendering(frame.cmd, &rendering);
        // TODO({slug}): draw something.
        vkCmdEndRendering(frame.cmd);
    }}
}};

}}  // namespace

int main(int argc, char** argv) {{
    try {{
        vkc::App::Options options{{}};
        options.title = "LearnVulkan - {title}";
        {class_name} app(vkc::App::parse_args(argc, argv, options));
        return app.run();
    }} catch (const std::exception& error) {{
        spdlog::error("{{}}", error.what());
        return EXIT_FAILURE;
    }}
}}
'''

SLANG_TEMPLATE = """// Shaders for {chapter_id}.
//
// One file, every stage. Entry points are marked with [shader("...")] and this project
// names them vertexMain and fragmentMain, which is what main.cpp passes as pName.

struct VSInput {{
    [[vk::location(0)]] float3 position;
}};

struct VSOutput {{
    float4 position : SV_Position;
}};

[shader("vertex")]
VSOutput vertexMain(VSInput input) {{
    VSOutput output;
    output.position = float4(input.position, 1.0);
    return output;
}}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {{
    return float4(1.0, 0.0, 1.0, 1.0);
}}
"""

CMAKE_TEMPLATE = """add_chapter({chapter_id}
    SOURCES main.cpp{shader_block}
)
"""

MDX_TEMPLATE = '''---
title: {title}
description: TODO - one sentence on what this chapter teaches.
sidebar:
  order: {order}
---

TODO: open with the problem this chapter solves, before any code.

## What we are building

TODO: describe the end state, and show the reference image.

## The code

TODO: introduce the API in small pieces, each with the reasoning behind it.

```cpp title="code/src/{part_dir}/{slug}/main.cpp"
// TODO: excerpt
```

## What changed since last chapter

TODO: name exactly what moved into `vkcommon` and what is new here.

## Exercises

1. TODO
'''


def to_class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-]", slug)) + "App"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chapter_id", help="e.g. 1.8.hello_triangle")
    parser.add_argument("--title", required=True, help='e.g. "Hello Triangle"')
    parser.add_argument("--part", required=True, help="docs part directory, e.g. 1.getting-started")
    parser.add_argument("--shaders", nargs="*", default=[],
                        help="Slang shader files to create and compile, e.g. lit.slang")
    args = parser.parse_args()

    match = re.fullmatch(r"(\d+)\.(\d+)\.([a-z0-9_]+)", args.chapter_id)
    if not match:
        print(f"error: chapter id '{args.chapter_id}' must look like 1.8.hello_triangle", file=sys.stderr)
        return 1
    part_number, order, slug = match.group(1), int(match.group(2)), match.group(3)

    # Code directories carry the part number ("2.lighting"); docs directories do not,
    # because Starlight strips dots from URLs. Reuse the existing directory for this
    # part if there is one, so the first chapter of a part sets the name and the rest
    # follow it.
    part_dir = next((d.name for d in sorted(CODE_SRC.iterdir())
                     if d.is_dir() and d.name.startswith(f"{part_number}.")), None)
    if part_dir is None:
        part_dir = f"{part_number}.{args.part.replace('-', '_')}"
    code_dir = CODE_SRC / part_dir / slug

    if code_dir.exists():
        print(f"error: {code_dir} already exists", file=sys.stderr)
        return 1

    code_dir.mkdir(parents=True)
    (code_dir / "main.cpp").write_text(
        MAIN_TEMPLATE.format(title=args.title, chapter_id=args.chapter_id, slug=slug, part=args.part, class_name=to_class_name(slug))
    )

    shader_block = ""
    if args.shaders:
        shader_block = "\n    SHADERS " + " ".join(args.shaders)
        for shader in args.shaders:
            (code_dir / shader).write_text(SLANG_TEMPLATE.format(chapter_id=args.chapter_id))

    (code_dir / "CMakeLists.txt").write_text(CMAKE_TEMPLATE.format(chapter_id=args.chapter_id, shader_block=shader_block))

    src_lists = CODE_SRC / "CMakeLists.txt"
    line = f"add_subdirectory({part_dir}/{slug})\n"
    text = src_lists.read_text()
    if line not in text:
        src_lists.write_text(text.rstrip("\n") + "\n" + line)

    docs_dir = DOCS / args.part
    docs_dir.mkdir(parents=True, exist_ok=True)
    page = docs_dir / f"{slug.replace('_', '-')}.mdx"
    if not page.exists():
        page.write_text(
            MDX_TEMPLATE.format(
                title=args.title,
                order=order,
                slug=slug,
                part_dir=part_dir,
            )
        )

    print(f"created {code_dir.relative_to(ROOT)}")
    print(f"created {page.relative_to(ROOT)}")
    print(f"registered in {src_lists.relative_to(ROOT)}")
    print("\nRe-run CMake to pick up the new target:\n  cmake --preset linux-debug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
