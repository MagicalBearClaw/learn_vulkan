#!/usr/bin/env python3
"""Is every Vulkan name explained in the chapter that first uses it?

Showing a line of code is not teaching it. A reader with no prior knowledge meeting
`VkPipelineLayout` for the first time needs it named and explained in the prose, not
just present in a listing they are told to copy.

This walks the chapters in reading order, collects the Vulkan names each sample uses,
works out where each one first appears, and checks that chapter's article mentions it
*outside* a code fence.

    python3 tools/terms.py --part 1
    python3 tools/terms.py --chapter 1.7 -v
    python3 tools/terms.py --part 1 --constants    # include the soft tier

Two tiers, because they need different standards:

  * Types (`VkFoo`) and functions (`vkFoo`) are the hard gate. Each one is a distinct
    object or operation with its own semantics; meeting one unexplained is a real gap.

  * Enum constants (`VK_FOO_BAR`) are reported only with --constants. Most are values
    whose family is explained once -- `VK_BLEND_FACTOR_ONE` needs no paragraph if
    blend factors were explained -- so gating on them individually produces noise
    rather than signal. `allowlist.json` records the families deemed self-evident.

Exit status is non-zero when the hard tier fails, so this is usable as a check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "code" / "src"
DOCS = ROOT / "site" / "src" / "content" / "docs"
CHAPTERS_PATH = ROOT / "tools" / "chapters.json"
ALLOWLIST_PATH = ROOT / "tools" / "allowlist.json"

SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".slang"}
FENCE = re.compile(r"^\s*(```|~~~)")

TYPE = re.compile(r"\bVk[A-Z][A-Za-z0-9]*")
FUNCTION = re.compile(r"\bvk[A-Z][A-Za-z0-9]*")
CONSTANT = re.compile(r"\bVK_[A-Z0-9_]+")

CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*")

# Suffix words that carry no concept of their own. `VkPipelineLayoutCreateInfo` is
# explained by prose about a "pipeline layout"; the CreateInfo/sType convention is
# taught once, in 1.2, and never needs restating per struct.
NOISE_WORDS = {"create", "info", "flags", "flag", "bits", "ext", "khr", "state",
               "properties", "2", "type"}

# Too generic to prove anything on their own, however long they are.
GENERIC_WORDS = {"requirements", "description", "descriptions", "attachment",
                 "reference", "structure", "version", "feature", "features",
                 "support", "surface", "buffers", "objects"}


def phrases(name: str) -> list[str]:
    """English forms an article might reasonably use for a Vulkan name.

    A good article writes "the pipeline layout", not `VkPipelineLayoutCreateInfo`, so
    matching the token alone reports teaching as a gap. These are the alternatives
    that count as having named the thing.
    """
    stem = name
    for prefix in ("vkCmd", "vkGet", "vkCreate", "vkDestroy", "vkEnumerate",
                   "vkAllocate", "vkFree", "vkBegin", "vkEnd", "vkReset", "vk", "Vk"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    words = [w.lower() for w in CAMEL.findall(stem)]
    core = [w for w in words if w not in NOISE_WORDS]

    out = {name.lower(), " ".join(words), " ".join(core)}
    if len(core) >= 2:
        out.add(" ".join(core[-2:]))
    # A distinctive final noun stands on its own: prose that says "the messenger" or
    # "the scissor rectangle" has named `VkDebugUtilsMessengerEXT` and
    # `vkCmdSetScissor`. Short or generic words would match anything, so they do not
    # count -- "size" must not let `VkDeviceSize` pass on the word "size" alone.
    if core and len(core[-1]) >= 7 and core[-1] not in GENERIC_WORDS:
        out.add(core[-1])
    # British spelling is used throughout the prose, while the Vulkan names are
    # American. Without both of these, "rasterisation" in an article fails to match
    # VkPipelineRasterizationStateCreateInfo and a well-explained concept is reported
    # as a gap.
    out |= {p.replace("color", "colour") for p in list(out)}
    out |= {p.replace("ization", "isation").replace("ize", "ise") for p in list(out)}
    return [p for p in out if p]


def is_named(name: str, prose_text: str) -> bool:
    lowered = prose_text.lower()
    return any(p in lowered for p in phrases(name))


def load_allowlist() -> dict:
    if ALLOWLIST_PATH.exists():
        return json.loads(ALLOWLIST_PATH.read_text())
    return {"families": [], "terms": []}


def prose(article: pathlib.Path) -> str:
    """Everything outside a fenced code block: the part that actually explains."""
    out: list[str] = []
    inside = False
    for line in article.read_text().splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            out.append(line)
    return "\n".join(out)


def sample_text(sample: pathlib.Path) -> str:
    if not sample.is_dir():
        return ""
    text = ""
    for path in sorted(sample.iterdir()):
        if path.suffix in SOURCE_SUFFIXES:
            text += path.read_text()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", help="part number, e.g. 1")
    parser.add_argument("--chapter", help="chapter id prefix, e.g. 1.7")
    parser.add_argument("--constants", action="store_true",
                        help="also report enum constants (soft tier)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every unexplained name")
    args = parser.parse_args()

    allowlist = load_allowlist()
    families = tuple(allowlist.get("families", []))
    allowed = set(allowlist.get("terms", []))

    chapters = json.loads(CHAPTERS_PATH.read_text())["chapters"]

    # First use is computed over the whole reading order, then filtered for display --
    # otherwise asking about one chapter would call every name in it "new".
    seen: set[str] = set()
    rows = []
    for chapter in chapters:
        text = sample_text(CODE / chapter["sample"])
        names = set(TYPE.findall(text)) | set(FUNCTION.findall(text))
        constants = set(CONSTANT.findall(text))
        # VK_ constants match the type pattern too; keep the tiers disjoint.
        names -= constants

        new_names = sorted(names - seen)
        new_constants = sorted(constants - seen)
        seen |= names | constants

        article = DOCS / chapter["article"]
        explained = prose(article) if article.exists() else ""

        missing = [n for n in new_names
                   if n not in allowed and not is_named(n, explained)]
        # Constants stay a literal match: they are the soft tier, and an underscored
        # SHOUTING_NAME has no natural English form the way a camel-cased type does.
        missing_constants = [
            c for c in new_constants
            if c not in explained and c not in allowed and not c.startswith(families)
        ]
        rows.append((chapter["id"], new_names, missing, new_constants,
                     missing_constants))

    if args.part:
        rows = [r for r in rows if r[0].startswith(f"{args.part}.")]
    if args.chapter:
        wanted = args.chapter.rstrip(".") + "."
        rows = [r for r in rows if r[0].startswith(wanted)]
    if not rows:
        print("no chapters matched", file=sys.stderr)
        return 2

    header = f"{'chapter':24}{'new':>6}{'unexplained':>13}"
    if args.constants:
        header += f"{'consts':>8}{'unexplained':>13}"
    print(header)

    failures = 0
    for cid, new_names, missing, new_constants, missing_constants in rows:
        line = f"{cid:24}{len(new_names):6}{len(missing):13}"
        if args.constants:
            line += f"{len(new_constants):8}{len(missing_constants):13}"
        print(line)
        if missing:
            failures += 1

    if args.verbose:
        for cid, _, missing, _, missing_constants in rows:
            if missing:
                print(f"\n{cid} types/functions:")
                for name in missing:
                    print(f"    {name}")
            if args.constants and missing_constants:
                print(f"\n{cid} constants:")
                for name in missing_constants:
                    print(f"    {name}")

    if failures:
        print(f"\n{failures} chapter(s) use a Vulkan type or function the article "
              f"never mentions in prose.")
        return 1
    print("\nevery type and function is named in the prose of the chapter that "
          "introduces it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
