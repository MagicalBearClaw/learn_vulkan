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

The fern is for chapter 4.2, blending and culling. It is the first texture here whose
alpha channel means anything, and two details in it are deliberate:

  * the silhouette is intricate, with gaps between the leaflets, so that the difference
    between discarding a fragment and blending it is visible rather than theoretical
  * every fully transparent texel still carries a leaf-green rgb. A sampler filtering
    across the silhouette's edge averages the colours of texels on both sides, and if
    the outside were black the leaf would get a dark fringe that no amount of alpha
    testing removes.

The sky is for chapter 4.4, cubemaps: six faces of one environment. It is the only
texture here that is not drawn as a picture at all. There is one function, sky_colour(),
that answers "what is in this direction", and each of the six faces is that function
evaluated over the directions that face covers. Two things follow from doing it that
way, and both are the reason it is done that way:

  * the six faces agree along their twelve shared edges automatically, because a texel
    on either side of an edge asks about very nearly the same direction. Six pictures
    drawn independently would have to be made to match, and would not quite
  * the face convention is testable. If the mapping from face texel to direction were
    wrong, the horizon would step or mirror at a face edge instead of running straight
    through it, which is obvious at a glance rather than subtle
"""

from __future__ import annotations

import math
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


# The fern: a stem with LEAFLET_PAIRS pairs of leaflets swept upward off it.
LEAFLET_PAIRS = 11
LEAF_BLEED = (58, 96, 44)  # rgb of every transparent texel; see the module docstring


def stem_centre(h: float) -> float:
    """Horizontal centre of the stem at height h, in 0..1. A slight lean, more at the top."""

    return 0.5 + 0.055 * math.sin(h * 2.2) * h


def leaflets() -> list[tuple[float, float, float, float, float, float, int]]:
    """(ox, oy, ax, ay, length, width, index) for every leaflet, both sides."""

    out = []
    for i in range(LEAFLET_PAIRS):
        base_h = 0.08 + i * 0.079
        # Leaflets near the tip sweep up harder, which is what makes a fern read as a
        # fern rather than as a feather duster.
        angle = math.radians(44.0 - 24.0 * base_h)
        length = 0.36 * (1.0 - base_h) ** 0.8 + 0.045
        # Narrow enough that neighbouring leaflets do not touch. The gaps between them
        # are the point: they are what makes a cutout silhouette intricate rather than
        # a blob, and what makes the sorting artefacts in this chapter visible.
        width = 0.030 * (1.0 - base_h) + 0.008
        for side in (-1.0, 1.0):
            ax = side * math.cos(angle)
            ay = math.sin(angle)
            out.append((stem_centre(base_h), base_h, ax, ay, length, width, i))
    return out


def leaf_hit(u: float, h: float, leaf) -> float | None:
    """Distance along the leaflet if (u, h) is inside it, else None."""

    ox, oy, ax, ay, length, width, _ = leaf
    dx, dy = u - ox, h - oy
    t = dx * ax + dy * ay
    if t < 0.0 or t > length:
        return None
    n = -dx * ay + dy * ax
    # Half-width tapers to nothing at both ends, fattest a third of the way along.
    half = width * math.sin(math.pi * (t / length) ** 0.7)
    return t if abs(n) <= half else None


def foliage_texture() -> bytearray:
    pixels = bytearray(SIZE * SIZE * 4)

    leaves = leaflets()
    # A generous circular bound per leaflet, so most pixels reject most leaflets with
    # one comparison instead of the full frame transform.
    bounds = [
        (ox + 0.5 * length * ax, oy + 0.5 * length * ay, (0.5 * length + width) ** 2)
        for ox, oy, ax, ay, length, width, _ in leaves
    ]

    stem_dark = (74, 88, 40)
    stem_light = (108, 126, 58)
    leaf_dark = (30, 74, 30)
    leaf_light = (116, 168, 66)

    # Half a texel, expressed in the 0..1 coordinates the shapes are defined in. Two
    # subsamples per axis is enough to soften the silhouette without hiding the
    # aliasing that alpha testing reintroduces -- which chapter 4.2 discusses.
    step = 0.5 / SIZE

    for y in range(SIZE):
        h_centre = 1.0 - (y + 0.5) / SIZE
        for x in range(SIZE):
            u_centre = (x + 0.5) / SIZE

            nearby = [
                leaves[i]
                for i, (bx, by, r2) in enumerate(bounds)
                if (u_centre - bx) ** 2 + (h_centre - by) ** 2 <= r2
            ]
            stem_possible = 0.01 < h_centre < 0.97 and abs(
                u_centre - stem_centre(h_centre)
            ) < 0.03

            covered = 0
            r = g = b = 0
            if nearby or stem_possible:
                for sy in (-step, step):
                    for sx in (-step, step):
                        u, h = u_centre + sx, h_centre + sy

                        hit = None
                        for leaf in nearby:
                            t = leaf_hit(u, h, leaf)
                            if t is not None:
                                hit = (t, leaf)
                                break

                        if hit is not None:
                            t, leaf = hit
                            length, index = leaf[4], leaf[6]
                            # Greener at the base of each leaflet, paler at its tip, and
                            # paler again toward the top of the plant.
                            mix = 0.35 * (t / length) + 0.55 * (index / LEAFLET_PAIRS)
                            mix = min(1.0, mix + 0.1 * hash_noise(x // 4, y // 4, 17))
                            covered += 1
                            r += int(leaf_dark[0] + (leaf_light[0] - leaf_dark[0]) * mix)
                            g += int(leaf_dark[1] + (leaf_light[1] - leaf_dark[1]) * mix)
                            b += int(leaf_dark[2] + (leaf_light[2] - leaf_dark[2]) * mix)
                            continue

                        half = 0.016 * (1.0 - 0.8 * h) + 0.002
                        if 0.01 < h < 0.97 and abs(u - stem_centre(h)) <= half:
                            # A lit edge down one side of the stem, so it reads as round.
                            side = (u - stem_centre(h)) / half
                            mix = max(0.0, min(1.0, 0.5 + 0.5 * side))
                            covered += 1
                            r += int(stem_dark[0] + (stem_light[0] - stem_dark[0]) * mix)
                            g += int(stem_dark[1] + (stem_light[1] - stem_dark[1]) * mix)
                            b += int(stem_dark[2] + (stem_light[2] - stem_dark[2]) * mix)

            offset = (y * SIZE + x) * 4
            if covered == 0:
                pixels[offset : offset + 4] = bytes((*LEAF_BLEED, 0))
            else:
                alpha = (covered * 255) // 4
                pixels[offset : offset + 4] = bytes(
                    (r // covered, g // covered, b // covered, alpha)
                )

    return pixels


def ground_texture() -> bytearray:
    """A floor that keeps quiet.

    The grid texture above is the opposite of this on purpose: it is loud because
    chapter 1.11 needs every filtering artefact to be obvious. From Part 4 on, the floor
    is scenery rather than subject, and a scene about transparent surfaces cannot afford
    a ground plane that competes with them. Low contrast, no markers, nothing saturated.
    """

    pixels = bytearray(SIZE * SIZE * 4)

    base = (104, 99, 92)
    stone = 128  # one flagstone every 128 texels, so four across the texture
    joint_width = 4

    for y in range(SIZE):
        for x in range(SIZE):
            # Three scales of variation, none of them strong: a per-flagstone tint, a
            # coarse mottle, and a fine grain. Together they stop the tiling from
            # reading as one flat colour without drawing the eye to anything.
            tint = 0.92 + 0.16 * hash_noise(x // stone, y // stone, 41)
            mottle = 0.93 + 0.14 * hash_noise(x // 16, y // 16, 67)
            grain = 0.94 + 0.12 * hash_noise(x // 2, y // 2, 53)
            shade = tint * mottle * grain

            if (x % stone) < joint_width or (y % stone) < joint_width:
                shade *= 0.72

            offset = (y * SIZE + x) * 4
            pixels[offset : offset + 4] = bytes(
                (*(min(255, int(c * shade)) for c in base), 255)
            )

    return pixels


# ---------------------------------------------------------------------------
# Normal maps, for chapter 5.5
# ---------------------------------------------------------------------------

# A multiplier on every slope in the normal maps. At 2 the grooves and the sides of the
# rivets lean 40 to 60 degrees from straight out, and the crate frame's inner bevel, the
# steepest edge in either map, about 80: enough to read as relief under a lamp.
NORMAL_STRENGTH = 2.0


def ramp(distance: float, width: float) -> float:
    """0 at an edge, rising smoothly to 1 at `width` texels inside it.

    A height field with hard steps has slopes only on the one texel where the step is,
    which a normal map turns into a one-texel line that aliases under every filter. A
    ramp a few texels wide gives the slope room to be filtered.
    """

    return smoothstep(0.0, width, distance)


def crate_height(x: float, y: float) -> float:
    """The crate's surface as a height field, 0 to 1.35, shaped by the same regions as
    crate_diffuse(): the steel frame stands proud of the planks, the rivets are domes on
    the frame, and the gaps between planks are grooves.

    Coordinates are continuous texel positions, so the normal map can take differences
    at sub-texel spacing.
    """

    # Rivets first: a spherical cap on top of the frame.
    for cx in (FRAME / 2, SIZE / 2, SIZE - FRAME / 2):
        for cy in (FRAME / 2, SIZE / 2, SIZE - FRAME / 2):
            if cx == SIZE / 2 and cy == SIZE / 2:
                continue
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            if r2 <= RIVET_RADIUS**2:
                return 1.0 + 0.35 * math.sqrt(1.0 - r2 / RIVET_RADIUS**2)

    # Distance from the frame's inner edge, positive inside the planks' area.
    inside = min(x - FRAME, y - FRAME, SIZE - FRAME - x, SIZE - FRAME - y)
    if inside <= 0.0:
        # The frame: a flat bar with a bevel down its outer edge, where the crate's
        # faces meet.
        outer = min(x, y, SIZE - x, SIZE - y)
        return 0.75 + 0.25 * ramp(outer, 4.0)

    # The planks, falling away from the frame's inner edge down a three-texel bevel.
    plank_top = 0.30 + 0.70 * (1.0 - ramp(inside, 3.0))

    # Grooves between planks: down to nothing over three texels either side of each gap.
    inner = SIZE - 2 * FRAME
    plank_height = inner / PLANKS
    along = (y - FRAME) % plank_height
    to_gap = min(along, plank_height - along)
    groove = ramp(to_gap, 3.0)

    # No grain. The diffuse map's grain bands step every 24 texels, which reads as wood
    # in colour and as a row of seams in relief: the planks are planed flat instead.
    return plank_top * groove


def ground_height(x: float, y: float) -> float:
    """The floor as a height field: flagstones, with sunken joints between them and a
    slight unevenness across each stone. The joints line up with ground_texture()'s."""

    stone = 128
    joint = 4.0
    # Distance to the nearest joint centre line, in texels.
    to_x = abs(((x + stone / 2 - joint / 2) % stone) - stone / 2)
    to_y = abs(((y + stone / 2 - joint / 2) % stone) - stone / 2)
    edge = min(to_x, to_y) - joint / 2
    sunk = ramp(edge, 3.0)

    # A gentle bump per stone, centred on it, so each stone catches the light a little
    # differently from its neighbours.
    sx, sy = int(x // stone), int(y // stone)
    lean_x = hash_noise(sx, sy, 71) - 0.5
    lean_y = hash_noise(sx, sy, 73) - 0.5
    u = (x % stone) / stone - 0.5
    v = (y % stone) / stone - 0.5
    tilt = 0.04 * (lean_x * u + lean_y * v)

    return (0.6 + tilt) * sunk


def normal_map(height, scale: float, wrap: bool) -> bytearray:
    """Turn a height field into a tangent-space normal map.

    The surface z = scale * h(x, y) has the normal (-dh/dx, -dh/dy, 1) before
    normalising. Both derivatives are central differences one texel apart.

    The convention is glTF's: x is the direction of increasing u, y points *up* the
    image -- towards decreasing v, because the image's first row is v = 0 -- and z points
    out of the surface. So the y derivative is taken upwards, which is minus the row
    direction.
    """

    pixels = bytearray(SIZE * SIZE * 4)
    for row in range(SIZE):
        for col in range(SIZE):
            x = col + 0.5
            y = row + 0.5

            def h(px: float, py: float) -> float:
                if wrap:
                    px %= SIZE
                    py %= SIZE
                else:
                    px = min(max(px, 0.0), SIZE - 1e-3)
                    py = min(max(py, 0.0), SIZE - 1e-3)
                return scale * height(px, py)

            dh_dx = (h(x + 1.0, y) - h(x - 1.0, y)) / 2.0
            dh_dup = (h(x, y - 1.0) - h(x, y + 1.0)) / 2.0

            nx, ny, nz = normalise(-dh_dx, -dh_dup, 1.0)
            encode = lambda c: int(round((c * 0.5 + 0.5) * 255.0))
            offset = (row * SIZE + col) * 4
            pixels[offset : offset + 4] = bytes((encode(nx), encode(ny), encode(nz), 255))
    return pixels


def crate_normal() -> bytearray:
    # The heights are 0..1.35 and the features a few texels wide, so a scale in texels
    # of the same order turns them into slopes of a sensible steepness.
    return normal_map(crate_height, 6.0 * NORMAL_STRENGTH, wrap=False)


def ground_normal() -> bytearray:
    # The floor tiles, so its differences wrap round the edges of the image.
    return normal_map(ground_height, 4.0 * NORMAL_STRENGTH, wrap=True)


# ---------------------------------------------------------------------------
# The sky cubemap, for chapter 4.4
# ---------------------------------------------------------------------------

SKY_SIZE = 512

# The six faces in Vulkan's layer order. A cubemap is a six-layer image and the layer
# index *is* the face, so this order is not a convention of ours to choose: layer 0 is
# +X, layer 1 is -X, and so on. The sample uploads the files in exactly this sequence.
CUBE_FACES = ("px", "nx", "py", "ny", "pz", "nz")

# Toward the sun. The scene's directional light travels the other way, so the sky and
# the lighting agree about where the sun is -- which stops mattering the moment you
# stop looking and starts mattering again the moment a mirrored surface shows the lit
# side of an object and the sun itself in the same picture.
SUN = (0.45, 1.0, 0.38)

ZENITH = (48, 92, 172)
HORIZON_SKY = (188, 210, 232)
CLOUD_LIT = (250, 250, 250)
CLOUD_BASE = (176, 182, 196)
MOUNTAIN_HIGH = (96, 104, 124)
MOUNTAIN_LOW = (62, 66, 82)
GROUND = (58, 54, 48)
SUN_DISC = (255, 252, 240)


def normalise(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    return x / length, y / length, z / length


def face_direction(face: int, u: float, v: float) -> tuple[float, float, float]:
    """The direction one texel of one cube face looks along.

    u and v run -1..1 across the face, v downward, which is the order the texels are
    stored in. The six cases are the Vulkan specification's cube map face selection
    table read backwards: the spec says which face a direction lands on and where,
    and this says which direction a given place on a given face came from.
    """

    if face == 0:
        return normalise(1.0, -v, -u)  # +X
    if face == 1:
        return normalise(-1.0, -v, u)  # -X
    if face == 2:
        return normalise(u, 1.0, v)  # +Y
    if face == 3:
        return normalise(u, -1.0, -v)  # -Y
    if face == 4:
        return normalise(u, -v, 1.0)  # +Z
    return normalise(-u, -v, -1.0)  # -Z


def hash_noise3(x: int, y: int, z: int, seed: int) -> float:
    n = (
        x * 374761393 + y * 668265263 + z * 1274126177 + seed * 2147483647
    ) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def value_noise3(x: float, y: float, z: float, seed: int) -> float:
    """Smoothly interpolated noise on a 3D lattice."""

    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    # Smoothstep each axis, otherwise the lattice shows through as diamonds.
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    sz = fz * fz * (3.0 - 2.0 * fz)

    total = 0.0
    for dz in (0, 1):
        wz = sz if dz else 1.0 - sz
        for dy in (0, 1):
            wy = sy if dy else 1.0 - sy
            for dx in (0, 1):
                wx = sx if dx else 1.0 - sx
                total += wx * wy * wz * hash_noise3(ix + dx, iy + dy, iz + dz, seed)
    return total


def fbm3(x: float, y: float, z: float, seed: int, octaves: int) -> float:
    total = 0.0
    amplitude = 1.0
    norm = 0.0
    for octave in range(octaves):
        total += amplitude * value_noise3(x, y, z, seed + octave)
        norm += amplitude
        x, y, z = x * 2.03, y * 2.03, z * 2.03
        amplitude *= 0.5
    return total / norm


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = (value - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t: float):
    return tuple(ca + (cb - ca) * t for ca, cb in zip(a, b))


def ridge_elevation(dx: float, dz: float) -> float:
    """Height of the mountain horizon in this compass direction, as a sine of elevation.

    A function of the azimuth alone, deliberately. The ridge has to be one continuous
    silhouette all the way round, and the only way to guarantee that across six
    separately generated faces is for it to depend on nothing else.
    """

    length = math.hypot(dx, dz)
    if length < 1e-6:
        return 0.0
    ax, az = dx / length, dz / length
    coarse = fbm3(ax * 2.6, 0.0, az * 2.6, 83, 3)
    fine = fbm3(ax * 9.0, 0.0, az * 9.0, 91, 2)
    return 0.045 + 0.115 * coarse + 0.030 * fine


def sky_colour(dx: float, dy: float, dz: float) -> tuple[int, int, int]:
    """What the environment looks like in one direction. The whole cubemap is this."""

    sun = normalise(*SUN)
    cos_sun = dx * sun[0] + dy * sun[1] + dz * sun[2]

    ridge = ridge_elevation(dx, dz)

    # The sky itself: a gradient from a pale horizon to a deep zenith. The exponent
    # compresses the pale band toward the horizon, which is what real aerial
    # perspective does and what stops the gradient looking like a linear ramp.
    sky = mix(HORIZON_SKY, ZENITH, max(dy, 0.0) ** 0.55)

    # The sun: a small hard disc inside a broad halo. Two powers of the cosine, one
    # tight and one wide, because a single one gives either a dot with no glow or a
    # smear with no sun.
    if cos_sun > 0.0:
        halo = 0.85 * cos_sun ** 900 + 0.30 * cos_sun ** 48
        sky = mix(sky, SUN_DISC, min(1.0, halo))
        if cos_sun > math.cos(math.radians(1.4)):
            sky = SUN_DISC

    # Clouds, on a plane above the viewer. The noise is sampled where the view
    # direction crosses that plane, so they crowd together and flatten toward the
    # horizon the way clouds actually do, instead of tiling like wallpaper.
    if dy > 0.0:
        height = max(dy, 0.06)
        density = fbm3(dx / height * 1.1, 0.0, dz / height * 1.1, 61, 3)
        coverage = smoothstep(0.50, 0.76, density) * smoothstep(0.02, 0.20, dy)
        if coverage > 0.0:
            cloud = mix(CLOUD_BASE, CLOUD_LIT, smoothstep(0.52, 0.85, density))
            sky = mix(sky, cloud, coverage)

    # The mountains, and the ground below them. Both are blended in over a narrow band
    # rather than switched on at a threshold: a hard comparison against the ridge line
    # would give a staircase silhouette that no amount of mip filtering removes.
    mountain_mix = smoothstep(ridge + 0.004, ridge - 0.004, dy)
    if mountain_mix > 0.0:
        depth = smoothstep(ridge, ridge - 0.26, dy)
        rock = mix(MOUNTAIN_HIGH, MOUNTAIN_LOW, depth)
        # Haze: the ridge line sits behind a lot of air, so it washes toward the colour
        # of the sky beside it and only the nearer, lower slopes keep their own colour.
        rock = mix(HORIZON_SKY, rock, 0.45 + 0.55 * depth)

        # The ground, on a plane below the viewer, sampled exactly the way the clouds
        # are sampled on a plane above it. Faint on purpose: the scene has a floor of
        # its own, and this is only what a mirrored surface sees underneath itself.
        drop = max(-dy, 0.04)
        patch = fbm3(dx / drop * 0.9, 0.0, dz / drop * 0.9, 29, 3)
        ground = mix(GROUND, mix(GROUND, HORIZON_SKY, 0.20), patch)
        rock = mix(rock, ground, smoothstep(-0.22, -0.52, dy))
        sky = mix(sky, rock, mountain_mix)

    return tuple(max(0, min(255, int(channel))) for channel in sky)


def sky_face(face: int) -> bytearray:
    pixels = bytearray(SKY_SIZE * SKY_SIZE * 4)
    for y in range(SKY_SIZE):
        v = 2.0 * (y + 0.5) / SKY_SIZE - 1.0
        for x in range(SKY_SIZE):
            u = 2.0 * (x + 0.5) / SKY_SIZE - 1.0
            r, g, b = sky_colour(*face_direction(face, u, v))
            offset = (y * SKY_SIZE + x) * 4
            pixels[offset : offset + 4] = bytes((r, g, b, 255))
    return pixels


def main() -> int:
    for index, face in enumerate(CUBE_FACES):
        target = OUT_DIR / f"lvk_sky_{face}.png"
        write_png(target, SKY_SIZE, SKY_SIZE, sky_face(index))
        print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")

    for name, pixels in (
        ("lvk_grid.png", grid_texture()),
        ("lvk_crate_diffuse.png", crate_diffuse()),
        ("lvk_crate_specular.png", crate_specular()),
        ("lvk_foliage.png", foliage_texture()),
        ("lvk_ground.png", ground_texture()),
        ("lvk_crate_normal.png", crate_normal()),
        ("lvk_ground_normal.png", ground_normal()),
    ):
        target = OUT_DIR / name
        write_png(target, SIZE, SIZE, pixels)
        print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
