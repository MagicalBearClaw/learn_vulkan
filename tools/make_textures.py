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

The crate pair is for chapter 2.4, lighting maps. It is two textures that describe one
surface: a diffuse map (what colour it is) and a specular map (how shiny each part of it
is). The design makes the specular map's job unmistakable -- a steel frame and rivets
that should gleam, wooden planks between them that should not.
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


def hash_noise(x: int, y: int, seed: int) -> float:
    """Deterministic value noise in 0..1 from integer coordinates.

    A hash rather than the random module, so the committed PNGs are byte-identical every
    time the script runs.
    """

    n = (x * 374761393 + y * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


FRAME = 56          # width of the steel frame, in texels
PLANKS = 5          # wooden planks between the frame's inner edges
RIVET_RADIUS = 9


def crate_region(x: int, y: int) -> str:
    """Which part of the crate a texel belongs to: frame, rivet, gap or wood."""

    rivet_centres = []
    for cx in (FRAME // 2, SIZE // 2, SIZE - FRAME // 2):
        for cy in (FRAME // 2, SIZE // 2, SIZE - FRAME // 2):
            if cx == SIZE // 2 and cy == SIZE // 2:
                continue
            rivet_centres.append((cx, cy))
    for cx, cy in rivet_centres:
        if (x - cx) ** 2 + (y - cy) ** 2 <= RIVET_RADIUS ** 2:
            return "rivet"

    if x < FRAME or y < FRAME or x >= SIZE - FRAME or y >= SIZE - FRAME:
        return "frame"

    inner = SIZE - 2 * FRAME
    plank_height = inner / PLANKS
    offset_in_plank = (y - FRAME) % plank_height
    if offset_in_plank < 3:
        return "gap"
    return "wood"


def crate_diffuse() -> bytearray:
    pixels = bytearray(SIZE * SIZE * 4)
    steel = (112, 120, 128)
    rivet = (150, 156, 162)
    gap = (38, 24, 14)
    wood_light = (168, 112, 62)
    wood_dark = (122, 76, 38)

    inner = SIZE - 2 * FRAME
    plank_height = inner / PLANKS

    for y in range(SIZE):
        for x in range(SIZE):
            region = crate_region(x, y)
            if region == "rivet":
                r, g, b = rivet
            elif region == "frame":
                # Brushed steel: fine streaks along the length of each frame bar.
                along = x if (y < FRAME or y >= SIZE - FRAME) else y
                streak = 0.9 + 0.1 * hash_noise(along // 3, 0 if along == x else 1, 7)
                r, g, b = (int(c * streak) for c in steel)
            elif region == "gap":
                r, g, b = gap
            else:
                # Wood grain: bands that run along the plank, wobbling a little, with a
                # different tint per plank so the planks read as separate boards.
                plank = int((y - FRAME) // plank_height)
                wobble = 6.0 * hash_noise(x // 24, plank, 11)
                grain = (y + wobble) % 9.0 / 9.0
                t = 0.55 + 0.45 * grain + 0.08 * (hash_noise(plank, 0, 3) - 0.5)
                t = max(0.0, min(1.0, t))
                r, g, b = (int(d + (l - d) * t) for l, d in zip(wood_light, wood_dark))
            offset = (y * SIZE + x) * 4
            pixels[offset : offset + 4] = bytes((r, g, b, 255))
    return pixels


def crate_specular() -> bytearray:
    """How much specular light each texel reflects, 0 (none) to 255 (all).

    This is data, not colour. The sample loads it as UNORM, and chapter 2.4 explains why
    loading it as SRGB would quietly change every value in it.
    """

    pixels = bytearray(SIZE * SIZE * 4)
    for y in range(SIZE):
        for x in range(SIZE):
            region = crate_region(x, y)
            if region == "rivet":
                v = 255
            elif region == "frame":
                # Shiny, but not uniformly: scuffs dull the steel in patches.
                scuff = hash_noise(x // 12, y // 12, 23)
                v = int(170 + 70 * scuff)
            elif region == "gap":
                v = 0
            else:
                # Varnished wood keeps a faint sheen.
                v = int(12 + 10 * hash_noise(x // 8, y // 8, 31))
            offset = (y * SIZE + x) * 4
            pixels[offset : offset + 4] = bytes((v, v, v, 255))
    return pixels


def main() -> int:
    for name, pixels in (
        ("lvk_grid.png", grid_texture()),
        ("lvk_crate_diffuse.png", crate_diffuse()),
        ("lvk_crate_specular.png", crate_specular()),
    ):
        target = OUT_DIR / name
        write_png(target, SIZE, SIZE, pixels)
        print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
