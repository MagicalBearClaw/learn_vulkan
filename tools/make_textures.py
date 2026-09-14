#!/usr/bin/env python3
"""Generate the test textures the early chapters use.

These are drawn here rather than downloaded so that a fresh clone can build and run the
texture chapters with no network access, and so that there is no licensing question
about them at all: they are this project's own work.

    python3 tools/make_textures.py

The grid texture is designed to make the things chapter 1.11 talks about visible:

  * a coarse checkerboard, so magnification filtering is obvious
  * one-pixel lines, so minification aliasing and mip levels are obvious
  * an asymmetric marker, so a flipped v coordinate is obvious at a glance
  * a coloured border, so the address modes are obvious when u or v leaves 0..1
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "textures"

SIZE = 512


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    """Write 8-bit RGBA with no filtering. Small and readable beats small."""

    raw = bytearray()
    stride = width * 4
    for row in range(height):
        raw.append(0)  # filter type: none
        raw += pixels[row * stride : (row + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def grid_texture() -> bytearray:
    pixels = bytearray(SIZE * SIZE * 4)

    # Two checkerboard colours, chosen to stay distinguishable when a mip level
    # averages them together.
    cool = (38, 92, 128)
    warm = (214, 138, 58)
    border = (222, 48, 122)
    line = (236, 240, 245)

    cell = 64

    for y in range(SIZE):
        for x in range(SIZE):
            checker = ((x // cell) + (y // cell)) % 2
            r, g, b = warm if checker else cool

            # A gentle diagonal gradient, so a stretched texture shows banding rather
            # than looking flat.
            shade = 0.82 + 0.18 * ((x + y) / (2.0 * SIZE))
            r, g, b = int(r * shade), int(g * shade), int(b * shade)

            # One-pixel lines every 16 texels across the bottom half. Under
            # minification these alias badly without mipmaps and cleanly with them,
            # which is the whole demonstration.
            if y >= SIZE // 2 and (x % 16 == 0 or y % 16 == 0):
                r, g, b = line

            # An upward-pointing triangle in the middle: the orientation marker. If the
            # v coordinate is flipped it points down, and you can see it immediately.
            cx, cy = SIZE // 2, SIZE // 2
            half_height = 120
            if cy - half_height <= y <= cy + half_height:
                t = (y - (cy - half_height)) / (2.0 * half_height)
                half_width = int(t * 110)
                if abs(x - cx) <= half_width:
                    r, g, b = line

            # An 8-texel border so the address modes are visible outside 0..1.
            if x < 8 or y < 8 or x >= SIZE - 8 or y >= SIZE - 8:
                r, g, b = border

            offset = (y * SIZE + x) * 4
            pixels[offset : offset + 4] = bytes((r, g, b, 255))

    return pixels


def main() -> int:
    target = OUT_DIR / "lvk_grid.png"
    write_png(target, SIZE, SIZE, grid_texture())
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
