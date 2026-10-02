#!/usr/bin/env python3
"""How much of `vkcommon` do the articles actually show, and in the right chapter?

`coverage.py` answers a different question: it diffs each chapter's sample against the
previous one and checks the article shows what is new. That says nothing about
`code/common/`, which a reader never writes as a sample but is still expected to
understand -- the vkcommon rule in CLAUDE.md says nothing enters it until the chapter
that teaches it has been written, and that a reader must never meet a helper they have
not built.

Two modes:

    python3 tools/attribution.py            # lenient: vs every Getting Started article
    python3 tools/attribution.py --strict   # vs the OWNING chapter's article only
    python3 tools/attribution.py --cmake    # per-chapter CMakeLists instead
    python3 tools/attribution.py -v         # list the lines nobody shows

Lenient mode matches against the union of all the articles, so a line counts as shown
if it appears anywhere, in any chapter, in any context. That over-credits short common
lines (`return handle_;`) and cannot tell "explained here" from "happens to appear".
Strict mode is the one that reflects the promise: the chapter that promotes a helper
has to be the chapter that shows it. Use strict as the gate; lenient only to see how
far off a file is.

Exit status is non-zero below --threshold, so this is usable as a check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON = ROOT / "code" / "common"
CODE = ROOT / "code" / "src"
DOCS = ROOT / "site" / "src" / "content" / "docs"
CHAPTERS_PATH = ROOT / "tools" / "chapters.json"

GETTING_STARTED = DOCS / "getting-started"

MIN_SIGNIFICANT = 4
FENCE = re.compile(r"^\s*(```|~~~)")

# Which chapter teaches each file -- the chapter whose article has to show it. This is
# first use, verified against the samples, not where the file happens to live:
#
#   * `Args`/`parse_args` live in capture.hpp and are used from 1.1; the `Capture`
#     class in the same header is not used until 1.5, so that header has two owners.
#   * window.hpp/cpp is taught in 1.1, which hand-writes it, and promoted at 1.2.
#   * pipeline.cpp has to be 1.7: chapter 1.8 already calls both `vkc::load_shader`
#     and `PipelineBuilder`.
#   * image.cpp splits -- create/upload/mips/samplers in 1.11, depth in 1.13.
#   * 4.7 adds multisampling: a sample count on ImageDesc and PipelineBuilder, and the
#     sampleRateShading feature in device.cpp. Its article shows those lines.
#   * 5.3 lets PipelineBuilder build a depth-only pipeline, with no fragment stage
#     and no colour attachment, for its shadow pass.
OWNERS: dict[str, list[str]] = {
    "base/include/vkc/capture.hpp": ["1.1", "1.5"],
    # parse_args belongs with the Args struct in 1.1; the Capture class is 1.5's.
    "base/src/capture.cpp": ["1.1", "1.5"],
    "base/include/vkc/check.hpp": ["1.2"],
    "base/src/check.cpp": ["1.2"],
    "base/include/vkc/paths.hpp": ["1.7"],
    "base/src/paths.cpp": ["1.7"],
    "base/src/vma_impl.cpp": ["1.3"],
    "base/src/stb_impl.cpp": ["1.11"],
    "scaffold/include/vkc/window.hpp": ["1.1"],
    "scaffold/src/window.cpp": ["1.1"],
    "scaffold/include/vkc/instance.hpp": ["1.2"],
    "scaffold/src/instance.cpp": ["1.2"],
    "scaffold/include/vkc/device.hpp": ["1.3"],
    "scaffold/src/device.cpp": ["1.3", "4.7"],
    "scaffold/include/vkc/swapchain.hpp": ["1.4"],
    "scaffold/src/swapchain.cpp": ["1.4"],
    "scaffold/include/vkc/context.hpp": ["1.5"],
    "scaffold/src/context.cpp": ["1.5"],
    "scaffold/include/vkc/frame.hpp": ["1.5", "1.6", "1.7"],
    "scaffold/src/frame.cpp": ["1.5", "1.6", "1.7"],
    "scaffold/include/vkc/pipeline.hpp": ["1.7", "4.7", "5.3"],
    "scaffold/src/pipeline.cpp": ["1.7", "4.7", "5.3"],
    "scaffold/include/vkc/app.hpp": ["1.8"],
    "scaffold/src/app.cpp": ["1.8"],
    "scaffold/include/vkc/buffer.hpp": ["1.8"],
    "scaffold/src/buffer.cpp": ["1.8"],
    "scaffold/include/vkc/image.hpp": ["1.11", "1.13", "4.7"],
    "scaffold/src/image.cpp": ["1.11", "1.13", "4.7"],
    "scaffold/include/vkc/camera.hpp": ["1.14"],
    "scaffold/src/camera.cpp": ["1.14"],
    # GpuTimer is written out in 4.6 and promoted in 4.7, which shows the lines the split
    # into a header and a source file changed.
    "scaffold/include/vkc/timer.hpp": ["4.6", "4.7"],
    "scaffold/src/timer.cpp": ["4.6", "4.7"],
}


def significant(line: str) -> bool:
    text = line.strip()
    if len(text) < MIN_SIGNIFICANT:
        return False
    if text.startswith(("//", "/*", "*", "#pragma once")):
        return False
    return True


def shown_lines(articles: list[pathlib.Path]) -> set[str]:
    """Every line that appears inside a fenced block in any of these articles."""
    shown: set[str] = set()
    for article in articles:
        inside = False
        for line in article.read_text().splitlines():
            if FENCE.match(line):
                inside = not inside
                continue
            if inside and significant(line):
                shown.add(line.strip())
    return shown


def article_for(chapter_prefix: str) -> pathlib.Path | None:
    """The article of the chapter whose id starts with e.g. "1.7."."""
    chapters = json.loads(CHAPTERS_PATH.read_text())["chapters"]
    wanted = chapter_prefix.rstrip(".") + "."
    for chapter in chapters:
        if chapter["id"].startswith(wanted):
            return DOCS / chapter["article"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="measure each file against its owning chapter only")
    parser.add_argument("--cmake", action="store_true",
                        help="measure per-chapter CMakeLists.txt instead of vkcommon")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list the lines no article shows")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="minimum acceptable percentage (default: 100)")
    args = parser.parse_args()

    if args.cmake:
        targets = [(str(p.relative_to(CODE)), p)
                   for p in sorted((CODE / "1.getting_started").glob("*/CMakeLists.txt"))]
    else:
        targets = [(str(p.relative_to(COMMON)), p)
                   for p in sorted(COMMON.rglob("*.hpp")) + sorted(COMMON.rglob("*.cpp"))]

    everything = shown_lines(sorted(GETTING_STARTED.glob("*.mdx")))

    if args.strict and not args.cmake:
        print("strict: each file measured against its owning chapter's article\n")
    else:
        print("lenient: measured against every Getting Started article\n")

    print(f"{'file':44} {'owner':10} {'lines':>6} {'shown':>6} {'%':>7}")
    total = covered_total = failures = 0
    for label, path in targets:
        lines = [ln for ln in path.read_text().splitlines() if significant(ln)]
        if not lines:
            continue

        owners = OWNERS.get(label, [])
        if args.strict and not args.cmake:
            if not owners:
                print(f"{label:44} {'UNOWNED':10} {len(lines):6} "
                      f"{'':6} {'':>7}  no owner in OWNERS")
                failures += 1
                continue
            articles = [a for a in (article_for(o) for o in owners) if a is not None]
            shown = shown_lines(articles)
        else:
            shown = everything

        missing = [ln for ln in lines if ln.strip() not in shown]
        covered = len(lines) - len(missing)
        total += len(lines)
        covered_total += covered
        percent = 100.0 * covered / len(lines)
        if percent < args.threshold:
            failures += 1
        owner_text = ",".join(owners) if owners else "-"
        print(f"{label:44} {owner_text:10} {len(lines):6} {covered:6} {percent:6.1f}%")
        if args.verbose and missing:
            for line in missing:
                print(f"    {line.strip()}")

    if total:
        print(f"\n{'TOTAL':44} {'':10} {total:6} {covered_total:6} "
              f"{100.0 * covered_total / total:6.1f}%")
    if failures:
        print(f"\n{failures} file(s) below {args.threshold:.0f}%.")
        return 1
    print(f"\nall files at or above {args.threshold:.0f}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
