#!/usr/bin/env python3
"""Render each chapter sample to a PNG and compare it against its reference image.

Every sample understands --frames and --screenshot, so a deterministic screenshot is
just a short headed run. Two uses:

    python3 tools/capture.py --all --update    # refresh the reference images
    python3 tools/capture.py --all             # check nothing changed

The check matters because chapters share vkcommon. When a subsystem is folded into
the shared library, every earlier chapter has to keep rendering exactly what its
article says it renders, and this is what proves it.

Needs a display: these are real windowed runs, not headless rendering.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "code" / "out" / "build"
REFERENCE_DIR = ROOT / "site" / "public" / "img" / "reference"

# Rendered at frame 90 so any start-up transient (the swapchain is commonly rebuilt
# once or twice as the compositor settles) is well past.
CAPTURE_FRAME = 90
# Percentage of pixels allowed to differ. Not zero: GPUs disagree in the last bit of
# a float, and a 0% threshold would fail on a driver update rather than on a bug.
TOLERANCE_PERCENT = 0.5


def read_png(path: Path) -> tuple[int, int, bytes]:
    """Decode an 8-bit RGBA PNG into raw pixels. Enough for what stb writes."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")

    width, height = struct.unpack(">II", data[16:24])
    bit_depth, colour_type = data[24], data[25]
    if bit_depth != 8 or colour_type != 6:
        raise ValueError(f"{path}: expected 8-bit RGBA, got depth {bit_depth} type {colour_type}")

    idat = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"IDAT":
            idat += data[offset + 8 : offset + 8 + length]
        offset += 12 + length

    raw = zlib.decompress(bytes(idat))
    return width, height, unfilter(raw, width, height)


def unfilter(raw: bytes, width: int, height: int) -> bytes:
    """Undo PNG per-scanline filtering for 4-byte pixels."""
    stride = width * 4
    out = bytearray(stride * height)
    previous = bytearray(stride)
    pos = 0

    for row in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride

        if filter_type == 1:  # Sub
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = line[i - 4] if i >= 4 else 0
                up = previous[i]
                up_left = previous[i - 4] if i >= 4 else 0
                estimate = left + up - up_left
                da, db, dc = abs(estimate - left), abs(estimate - up), abs(estimate - up_left)
                nearest = left if (da <= db and da <= dc) else (up if db <= dc else up_left)
                line[i] = (line[i] + nearest) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter {filter_type}")

        out[row * stride : (row + 1) * stride] = line
        previous = line

    return bytes(out)


def compare(actual: Path, expected: Path) -> tuple[bool, str]:
    aw, ah, apx = read_png(actual)
    ew, eh, epx = read_png(expected)

    if (aw, ah) != (ew, eh):
        return False, f"size changed: {ew}x{eh} -> {aw}x{ah}"

    differing = sum(1 for i in range(0, len(apx), 4) if apx[i : i + 3] != epx[i : i + 3])
    total = aw * ah
    percent = 100.0 * differing / total
    if percent > TOLERANCE_PERCENT:
        return False, f"{percent:.2f}% of pixels differ (tolerance {TOLERANCE_PERCENT}%)"
    return True, f"{percent:.2f}% of pixels differ"


def find_binaries(build_dir: Path, only: str | None) -> list[Path]:
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        raise SystemExit(f"error: {bin_dir} not found. Build first:\n  cmake --build {build_dir}")
    binaries = sorted(p for p in bin_dir.iterdir() if p.is_file() and p.stat().st_mode & 0o111)
    if only:
        binaries = [p for p in binaries if only in p.name]
    return binaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="linux-debug", help="CMake preset whose build tree to use")
    parser.add_argument("--all", action="store_true", help="capture every chapter")
    parser.add_argument("--chapter", help="capture only chapters whose name contains this")
    parser.add_argument("--update", action="store_true", help="overwrite the reference images instead of comparing")
    parser.add_argument("--out", type=Path, help="write captures here instead of a temporary directory")
    args = parser.parse_args()

    if not args.all and not args.chapter:
        parser.error("pass --all or --chapter NAME")

    build_dir = BIN_DIR / args.preset
    binaries = find_binaries(build_dir, args.chapter)
    if not binaries:
        print("nothing to capture", file=sys.stderr)
        return 1

    out_dir = args.out or (ROOT / "captures")
    out_dir.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    failures = []

    for binary in binaries:
        name = binary.name
        destination = (REFERENCE_DIR if args.update else out_dir) / f"{name}.png"

        print(f"{name}: rendering {CAPTURE_FRAME} frames")
        result = subprocess.run(
            [str(binary), "--frames", str(CAPTURE_FRAME), "--screenshot", str(destination)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Validation messages are fatal for a tutorial: the samples are what people
        # copy, and copying something the validator objects to teaches the wrong thing.
        validation_errors = [line for line in result.stderr.splitlines() + result.stdout.splitlines() if "[vulkan]" in line]
        if validation_errors:
            print(f"  VALIDATION ERRORS ({len(validation_errors)}):")
            for line in validation_errors[:5]:
                print(f"    {line}")
            failures.append(name)
            continue

        if result.returncode != 0 or not destination.exists():
            print(f"  FAILED (exit {result.returncode})")
            print(result.stdout[-2000:] or result.stderr[-2000:])
            failures.append(name)
            continue

        if args.update:
            print(f"  reference updated: {destination.relative_to(ROOT)}")
            continue

        reference = REFERENCE_DIR / f"{name}.png"
        if not reference.exists():
            print(f"  no reference image yet; run with --update to create one")
            continue

        ok, detail = compare(destination, reference)
        print(f"  {'ok' if ok else 'CHANGED'}: {detail}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} chapter(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    print(f"\n{len(binaries)} chapter(s) ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
