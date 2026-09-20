#!/usr/bin/env python3
"""How much of a chapter's new code does its article actually show?

A chapter teaches the code that is new in it -- the diff between its sample and the
previous chapter's sample. A reader following along has to write every one of those
lines, so every one of them belongs in the article. Lines carried over unchanged from
an earlier chapter do not: the article points at where they came from instead.

    python3 tools/coverage.py --part 1          # every chapter in Part 1
    python3 tools/coverage.py --chapter 1.4     # one chapter
    python3 tools/coverage.py --chapter 1.4 -v  # ... and list what is missing

Exit status is non-zero if any chapter falls below the threshold, so this is usable as
a check and not only as a report.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "code" / "src"
DOCS = ROOT / "site" / "src" / "content" / "docs"

# Which sample each chapter's article documents, and which sample precedes it. The
# order is the reading order, which is not always the directory order.
CHAPTERS_PATH = ROOT / "tools" / "chapters.json"

# `.txt` is here for CMakeLists.txt. A reader who cannot build the chapter has not
# finished it, so the build file is code they have to write like any other.
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".slang", ".txt"}

# A line has to carry some substance before its absence means anything. Bare braces,
# `};` and one-word lines are noise: they match everywhere and teach nothing.
MIN_SIGNIFICANT = 4


def significant(line: str) -> bool:
    text = line.strip()
    if len(text) < MIN_SIGNIFICANT:
        return False
    if text.startswith(("//", "/*", "*", "#pragma once")):
        return False
    return True


def source_lines(sample: pathlib.Path) -> list[str]:
    if not sample.is_dir():
        return []
    out: list[str] = []
    for path in sorted(sample.iterdir()):
        if path.suffix in SOURCE_SUFFIXES:
            out.extend(path.read_text().splitlines())
    return out


def new_lines(sample: pathlib.Path, previous: pathlib.Path | None) -> list[str]:
    """The lines a reader of this chapter has to write that they did not write before."""
    current = source_lines(sample)
    earlier = source_lines(previous) if previous else []
    matcher = difflib.SequenceMatcher(None, earlier, current, autojunk=False)
    added: list[str] = []
    for tag, _, _, start, end in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(current[start:end])
    # A `replace` opcode spans a whole changed block, so adding one field to a struct
    # marks every line of it new. Those lines are not new to the reader: they typed
    # them in the previous chapter and the article points back at them. Keep only what
    # was not in front of the reader before, wherever it falls in the diff.
    carried = {line.strip() for line in earlier if significant(line)}
    return [line for line in added
            if significant(line) and line.strip() not in carried]


# Fenced blocks only. An indented block inside a list item is still fenced in MDX.
FENCE = re.compile(r"^\s*(```|~~~)")


def article_lines(article: pathlib.Path) -> set[str]:
    shown: set[str] = set()
    inside = False
    for line in article.read_text().splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if inside and significant(line):
            shown.add(line.strip())
    return shown


def load_chapters() -> list[dict]:
    return json.loads(CHAPTERS_PATH.read_text())["chapters"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", help="part number, e.g. 1")
    parser.add_argument("--chapter", help="chapter id prefix, e.g. 1.4")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list the new lines the article never shows")
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="minimum acceptable percentage (default: 90)")
    args = parser.parse_args()

    chapters = load_chapters()
    if args.part:
        chapters = [c for c in chapters if c["id"].startswith(f"{args.part}.")]
    if args.chapter:
        # Match on whole segments, so "1.1" finds 1.1 and not 1.10 through 1.14.
        wanted = args.chapter.rstrip(".") + "."
        chapters = [c for c in chapters if c["id"].startswith(wanted)]
    if not chapters:
        print("no chapters matched", file=sys.stderr)
        return 2

    print(f"{'chapter':24} {'new':>5} {'shown':>6} {'%':>7}")
    failures = 0
    for chapter in chapters:
        sample = CODE / chapter["sample"]
        previous = CODE / chapter["previous"] if chapter.get("previous") else None
        article = DOCS / chapter["article"]

        added = new_lines(sample, previous)
        shown = article_lines(article)
        missing = [line for line in added if line.strip() not in shown]
        covered = len(added) - len(missing)
        percent = 100.0 * covered / len(added) if added else 100.0

        mark = " " if percent >= args.threshold else " FAIL"
        print(f"{chapter['id']:24} {len(added):5} {covered:6} {percent:6.1f}%{mark}")
        if percent < args.threshold:
            failures += 1
        if args.verbose and missing:
            for line in missing:
                print(f"    {line.strip()}")

    if failures:
        print(f"\n{failures} chapter(s) below {args.threshold:.0f}%.")
        return 1
    print(f"\n{len(chapters)} chapter(s) ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
