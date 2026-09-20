#!/usr/bin/env python3
"""Is every idea a chapter rests on defined in that chapter?

`terms.py` asks whether the Vulkan *names* are explained. This asks the harder
question: whether the concepts and mathematics underneath them are. A chapter can name
`VkPipelineColorBlendAttachmentState` faithfully and still never say what a normal
matrix is, or why light falls off with the square of distance.

The check is narrow on purpose. A concept assigned to a chapter in `concepts.json` has
to appear in that chapter's `Words introduced in this chapter` table, with a definition
beside it. A looser test -- the phrase occurring anywhere in the prose -- matches
passing mentions and reports a chapter as fine when it only ever used the word.

    python3 tools/concepts.py --part 2
    python3 tools/concepts.py --chapter 2.2 -v

Exit status is non-zero when a chapter is missing a concept, so this is a check and not
only a report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "site" / "src" / "content" / "docs"
CHAPTERS_PATH = ROOT / "tools" / "chapters.json"
CONCEPTS_PATH = ROOT / "tools" / "concepts.json"

HEADING = "## Words introduced in this chapter"

# `| **term** | definition |`, which is the shape every table in the series uses.
ROW = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|(.*)\|\s*$")


def normalise(text: str) -> str:
    """Compare terms by their words, not their punctuation.

    The tables mark code with backticks and separate alternatives with slashes --
    "`loadOp` / `storeOp`", "yaw / pitch". None of that should decide whether a concept
    counts as defined.
    """
    text = text.replace("`", " ").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return " ".join(text.split())


def table_of(article: pathlib.Path) -> list[tuple[str, str]] | None:
    """The chapter's term table, or None when it has no table at all."""
    if not article.exists():
        return None
    rows: list[tuple[str, str]] = []
    inside = False
    for line in article.read_text().splitlines():
        if line.startswith(HEADING):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside:
            match = ROW.match(line)
            if match:
                rows.append((match.group(1), match.group(2).strip()))
    return rows if inside else None


def defines(concept: str, rows: list[tuple[str, str]]) -> bool:
    """Does one of the table's terms name this concept, and define it?

    An empty definition cell does not count. A row is a promise that the term is
    explained; a term with nothing beside it is the promise without the explanation.
    """
    wanted = normalise(concept)
    for term, definition in rows:
        if not definition:
            continue
        found = normalise(term)
        if wanted == found or wanted in found or found in wanted:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", help="part number, e.g. 2")
    parser.add_argument("--chapter", help="chapter id prefix, e.g. 2.2")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every undefined concept")
    args = parser.parse_args()

    chapters = json.loads(CHAPTERS_PATH.read_text())["chapters"]
    assigned = json.loads(CONCEPTS_PATH.read_text())["chapters"]

    if args.part:
        chapters = [c for c in chapters if c["id"].startswith(f"{args.part}.")]
    if args.chapter:
        wanted = args.chapter.rstrip(".") + "."
        chapters = [c for c in chapters if c["id"].startswith(wanted)]
    if not chapters:
        print("no chapters matched", file=sys.stderr)
        return 2

    print(f"{'chapter':26}{'concepts':>9}{'undefined':>11}")
    failures = 0
    detail: list[tuple[str, list[str]]] = []

    for chapter in chapters:
        concepts = assigned.get(chapter["id"], [])
        if not concepts:
            print(f"{chapter['id']:26}{'-':>9}{'-':>11}   no concepts assigned")
            continue

        rows = table_of(DOCS / chapter["article"])
        if rows is None:
            print(f"{chapter['id']:26}{len(concepts):9}{len(concepts):11}   "
                  f"FAIL no '{HEADING.lstrip('# ')}' section")
            failures += 1
            detail.append((chapter["id"], concepts))
            continue

        missing = [c for c in concepts if not defines(c, rows)]
        mark = " FAIL" if missing else ""
        print(f"{chapter['id']:26}{len(concepts):9}{len(missing):11}{mark}")
        if missing:
            failures += 1
            detail.append((chapter["id"], missing))

    if args.verbose:
        for cid, missing in detail:
            print(f"\n{cid}:")
            for concept in missing:
                print(f"    {concept}")

    if failures:
        print(f"\n{failures} chapter(s) rest on an idea they never define.")
        return 1
    print("\nevery assigned concept is defined in its own chapter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
