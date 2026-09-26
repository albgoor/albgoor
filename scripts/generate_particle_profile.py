"""Build-time generator for the animated particle hero in assets/profile-terminal-*.svg.

Reads the private source photo (YASEEN.jpeg, git-ignored), derives a particle
portrait from it, builds four procedural target states, matches particles
between states and writes an SMIL-only particle block into both hero SVGs
between the PARTICLES:BEGIN / PARTICLES:END markers.

Usage (from the repository root):
    .venv/Scripts/python scripts/generate_particle_profile.py [--debug]

Nothing produced here is needed at runtime: the SVGs are self-contained and
never embed the photograph or any raster derived from it.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "YASEEN.jpeg"
DEBUG_DIR = ROOT / "tmp"
SVG_DARK = ROOT / "assets" / "profile-terminal-dark.svg"
SVG_LIGHT = ROOT / "assets" / "profile-terminal-light.svg"

SEED = 42
N = 1300

# Source-pixel geometry (YASEEN.jpeg is 1170x1496). Verified against tmp/source-mask-debug.png.
CIGARETTE_POLY = [(581, 840), (621, 833), (818, 1066), (815, 1110), (781, 1117), (584, 873)]
HAND_POLY = [
    (580, 345), (615, 362), (642, 390), (645, 422), (680, 438), (770, 442), (812, 450),
    (838, 472), (825, 500), (838, 550), (880, 565), (1000, 565), (1170, 565),
    (1170, 88), (1022, 80), (818, 124), (687, 190), (621, 241), (606, 299),
]
HOOD_ELLIPSE = ((584, 613), (372, 548))  # centre, semi-axes: rebuilt hood contour behind the hand
CIGARETTE_SHIFT = 80  # px; texture donor offset, larger than the full dilated cigarette width
FACE_AXIS_X = 593  # midpoint between the pupils
FACE_ELLIPSE = (585, 700, 290, 440)  # cx, cy, ax, ay: soft face region for the light-theme density
CROP = (58, 58, 1110, 1210)  # x0, y0, x1, y1
FEATURES = [  # (cx, cy, sx, sy, gain): identity-bearing regions, in priority order
    (464, 552, 70, 30, 1.6),   # left eye
    (723, 548, 70, 30, 1.6),   # right eye
    (453, 416, 110, 34, 1.2),  # left brow
    (740, 492, 95, 30, 1.2),   # right brow
    (635, 690, 60, 90, 1.0),   # nose
    (643, 820, 110, 40, 0.8),  # lips
    (628, 945, 190, 120, 0.7),  # beard
]

# SVG layout: VISUAL.MAP viewport inside the 1200x520 hero.
VIEW = (48, 124, 432, 244)
CENTER = (264.0, 247.0)
PORTRAIT_BOX = (216.0, 236.0)
SAMPLE_GAMMA = 1.45
MIN_SPACING = 2.0

DUR = 18
KEY_SECONDS = [0, 3.8, 5.0, 7.0, 8.2, 10.2, 11.4, 13.4, 14.6, 16.3, 18]
HOLD, EASE = "0 0 1 1", ".45 0 .25 1"
SPLINES = [HOLD, EASE, HOLD, EASE, HOLD, EASE, HOLD, EASE, HOLD, EASE]
STATE_NAMES = ["PORTRAIT", "CONTAINER", "YARD", "ARCHITECTURE", "SPATIAL 3D"]


# --------------------------------------------------------------------------- portrait


def _poly(points, sx, sy, ox, oy):
    return np.array([((x - ox) * sx, (y - oy) * sy) for x, y in points], np.int32)


def _remove_background(bgr: np.ndarray) -> np.ndarray:
    from rembg import new_session, remove

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for model in ("u2net_human_seg", "u2net"):
        try:
            return np.asarray(remove(rgb, session=new_session(model), only_mask=True))
        except Exception as exc:  # model download / runtime failure
            print(f"rembg model {model} failed: {exc}")
    raise RuntimeError("rembg could not produce a subject mask")


@dataclass
class PortraitMaps:
    weights: dict[str, np.ndarray]  # per theme
    detail: np.ndarray
    scale: float  # working pixels per SVG unit


def build_portrait_maps(debug: bool) -> PortraitMaps:
    bgr = cv2.imread(str(SOURCE))
    if bgr is None:
        raise FileNotFoundError(f"missing private source photo: {SOURCE}")
    h, w = bgr.shape[:2]
    alpha = _remove_background(bgr)

    cig = np.zeros((h, w), np.uint8)
    cv2.fillPoly(cig, [np.array(CIGARETTE_POLY, np.int32)], 255)
    cig = cv2.dilate(cig, np.ones((31, 31), np.uint8))
    hand = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hand, [np.array(HAND_POLY, np.int32)], 255)
    hood = np.zeros((h, w), np.uint8)
    cv2.ellipse(hood, HOOD_ELLIPSE[0], HOOD_ELLIPSE[1], 0, 0, 360, 255, -1)

    right_upper = np.zeros((h, w), np.uint8)
    right_upper[:900, FACE_AXIS_X:] = 255  # everything beside the hood above the shoulder line
    hand_outside = cv2.bitwise_and(cv2.bitwise_or(hand, right_upper), cv2.bitwise_not(hood))
    hand_inside = cv2.bitwise_and(hand, hood)
    subject = ((alpha > 127).astype(np.uint8) * 255) & cv2.bitwise_not(hand_outside)
    subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # Replace the cigarette with real neighbouring beard/lip texture before any edge work:
    # average two copies shifted perpendicular to the cigarette axis (one from each side).
    (ax, ay), (bx, by) = CIGARETTE_POLY[0], CIGARETTE_POLY[3]
    axis = np.array([bx - ax, by - ay], np.float32)
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis) * CIGARETTE_SHIFT
    shifted = [cv2.warpAffine(bgr, np.float32([[1, 0, -sx], [0, 1, -sy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
               for sx, sy in (normal, -normal)]
    fill = 0.5 * shifted[0].astype(np.float32) + 0.5 * shifted[1].astype(np.float32)
    soft = cv2.GaussianBlur(cig, (0, 0), 5).astype(np.float32)[..., None] / 255
    clean = (bgr * (1 - soft) + fill * soft).astype(np.uint8)
    # Fingers over the forehead/hair: fill with the mirrored left side of the head (feathered).
    mirror_x = np.clip(2 * FACE_AXIS_X - np.arange(w), 0, w - 1)
    mirrored = clean[:, mirror_x]
    feather = cv2.GaussianBlur(cv2.dilate(hand_inside, np.ones((15, 15), np.uint8)), (0, 0), 9)[..., None] / 255.0
    clean = (clean * (1 - feather) + mirrored * feather).astype(np.uint8)

    if debug:
        _write_source_debug(bgr, subject, cig, hand, hand_inside)

    x0, y0, x1, y1 = CROP
    scale = 3.0
    ww, wh = int(PORTRAIT_BOX[0] * scale), int(PORTRAIT_BOX[1] * scale)
    def fit(a):
        return cv2.resize(a[y0:y1, x0:x1], (ww, wh), interpolation=cv2.INTER_AREA)

    img = fit(clean)
    mask = fit(subject).astype(np.float32) / 255
    cigw = fit(cig) > 0
    downw = fit(hand_inside).astype(np.float32) / 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.4)
    tone = np.clip((blur.astype(np.float32) - 38) / 190, 0, 1) ** 1.15

    edges = cv2.Canny(blur, 28, 80).astype(np.float32) / 255
    edges = cv2.GaussianBlur(edges, (0, 0), 1.3)
    edges /= edges.max() + 1e-6
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = np.clip(grad / (np.percentile(grad, 99) + 1e-6), 0, 1)

    # Silhouette band; replicate-pad so the crop border is not treated as an outline.
    pad = 40
    padded = cv2.copyMakeBorder((mask > .5).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
    band = np.exp(-dist / 5.0) * (mask > .5)

    yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
    feature = np.zeros((wh, ww), np.float32)
    sx, sy = ww / (x1 - x0), wh / (y1 - y0)
    for cx, cy, fx, fy, gain in FEATURES:
        u, v = (cx - x0) * sx, (cy - y0) * sy
        feature += gain * np.exp(-(((xx - u) / (fx * sx)) ** 2 + ((yy - v) / (fy * sy)) ** 2))

    # No hard edges may survive around the removed cigarette or the synthetic hand fill.
    seam = cv2.GaussianBlur(np.maximum(cigw.astype(np.float32), downw), (0, 0), 4)
    seam = np.clip(seam * 2, 0, 1)
    edges *= 1 - seam
    grad *= 1 - seam

    detail = np.clip(0.6 * edges + 0.4 * grad, 0, 1)
    fade = np.clip((wh - yy) / (0.22 * wh), 0, 1) ** 1.5  # dissolve the shoulders into the grid
    envelope = mask * (1 - 0.3 * downw) * fade

    # Dark theme: light particles on a dark panel, so particles follow luminance.
    weight_dark = ((0.46 * tone + 0.36 * edges + 0.18 * grad) * (1 + 1.8 * feature) + 0.6 * band) * envelope

    # Light theme: dark particles on a light panel, so particles follow darkness inside the face
    # (brows, eyes, beard) while the hood stays an outline instead of a solid mass.
    cx, cy, ax, ay = FACE_ELLIPSE
    face = np.exp(-((((xx - (cx - x0) * sx) / (ax * sx)) ** 2 + ((yy - (cy - y0) * sy) / (ay * sy)) ** 2) ** 2))
    darkness = np.clip((200 - blur.astype(np.float32)) / 170, 0, 1) ** 1.3
    weight_light = ((0.46 * darkness * face + 0.36 * edges + 0.18 * grad) * (1 + 1.8 * feature)
                    + 0.6 * band + 0.03 * darkness) * envelope

    if debug:
        for name, wgt in (("dark", weight_dark), ("light", weight_light)):
            cv2.imwrite(str(DEBUG_DIR / f"portrait-weight-{name}.png"), (255 * wgt / wgt.max()).astype(np.uint8))
        cv2.imwrite(str(DEBUG_DIR / "portrait-clean-gray.png"), (gray * (mask > .5)).astype(np.uint8))
    return PortraitMaps(weights={"dark": weight_dark, "light": weight_light}, detail=detail, scale=scale)


def _write_source_debug(bgr, subject, cig, hand, hand_inside):
    ov = bgr.copy()
    tint = np.zeros_like(ov)
    tint[subject > 0] = (90, 70, 0)
    ov = cv2.addWeighted(ov, 1, tint, .55, 0)
    ov[hand_inside > 0] = (ov[hand_inside > 0] * .5 + np.array([0, 140, 255]) * .5).astype(np.uint8)
    cv2.polylines(ov, [np.array(HAND_POLY, np.int32)], True, (0, 165, 255), 3)
    cv2.ellipse(ov, HOOD_ELLIPSE[0], HOOD_ELLIPSE[1], 0, 0, 360, (255, 255, 0), 2)
    contours, _ = cv2.findContours(cig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, contours, -1, (0, 0, 255), 3)
    cv2.rectangle(ov, CROP[:2], CROP[2:], (0, 255, 0), 3)
    legend = [("subject mask (blue tint)", (255, 200, 0)), ("cigarette exclusion", (0, 0, 255)),
              ("hand: cut outside hood / down-weight inside", (0, 165, 255)),
              ("rebuilt hood contour", (255, 255, 0)), ("portrait crop", (0, 255, 0))]
    for i, (label, color) in enumerate(legend):
        cv2.putText(ov, label, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5)
        cv2.putText(ov, label, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2)
    cv2.imwrite(str(DEBUG_DIR / "source-mask-debug.png"), cv2.resize(ov, None, fx=.6, fy=.6))


def sample_portrait(maps: PortraitMaps, theme: str, rng: np.random.Generator):
    """Weighted candidates + variable-radius rejection (Poisson-disc-like)."""
    base = maps.weights[theme]
    wgt = base ** SAMPLE_GAMMA
    h, w = wgt.shape
    p = wgt.ravel() / wgt.sum()
    idx = rng.choice(p.size, size=N * 40, replace=True, p=p)
    ys, xs = np.divmod(idx, w)
    pts = np.stack([xs, ys], 1).astype(np.float32) + rng.random((idx.size, 2), dtype=np.float32)
    local = np.clip(base[ys, xs] / np.percentile(base[base > 0], 97), 0, 1)

    # Minimum spacing (SVG units) shrinks on detailed pixels so features stay crisp.
    radius = (MIN_SPACING * (0.75 + 0.9 * (1 - local))) * maps.scale
    cell = radius.max()
    grid: dict[tuple[int, int], list[int]] = {}
    acc: list[int] = []
    for i, (x, y) in enumerate(pts):
        gx, gy = int(x // cell), int(y // cell)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    qx, qy = pts[j]
                    if (qx - x) ** 2 + (qy - y) ** 2 < max(radius[i], radius[j]) ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gx, gy), []).append(i)
            acc.append(i)
            if len(acc) == N:
                break
    if len(acc) < N:
        raise RuntimeError(f"portrait sampling reached only {len(acc)} of {N} points; lower MIN_SPACING")
    sel = np.array(acc)
    xy = pts[sel] / maps.scale
    xy[:, 0] += CENTER[0] - PORTRAIT_BOX[0] / 2
    xy[:, 1] += CENTER[1] - PORTRAIT_BOX[1] / 2
    detail = maps.detail[ys[sel], xs[sel]]
    return xy, np.clip(0.55 * local[sel] + 0.45 * detail, 0, 1)


# --------------------------------------------------------------------------- procedural states


class ShapeBuilder:
    """Collects weighted line segments / point clusters and resamples them to exactly N points."""

    def __init__(self):
        self.segs: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.dots: list[tuple[np.ndarray, float, int]] = []

    def line(self, a, b, density=1.0):
        self.segs.append((np.asarray(a, float), np.asarray(b, float), density))

    def polyline(self, pts, density=1.0, closed=False):
        pts = list(pts) + ([pts[0]] if closed else [])
        for a, b in zip(pts, pts[1:]):
            self.line(a, b, density)

    def cluster(self, c, radius, count):
        self.dots.append((np.asarray(c, float), radius, count))

    def build(self, rng: np.random.Generator, jitter=0.35) -> np.ndarray:
        out = []
        for c, radius, count in self.dots:
            ang = rng.random(count) * 2 * np.pi
            rad = radius * np.sqrt(rng.random(count))
            out.append(c + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1))
        fixed = sum(len(o) for o in out)
        remaining = N - fixed
        lengths = np.array([np.linalg.norm(b - a) * d for a, b, d in self.segs])
        counts = np.floor(lengths / lengths.sum() * remaining).astype(int)
        order = np.argsort(-(lengths / lengths.sum() * remaining - counts))
        counts[order[: remaining - counts.sum()]] += 1
        for (a, b, _), k in zip(self.segs, counts):
            if k <= 0:
                continue
            t = (np.arange(k) + rng.random(k) * 0.6 + 0.2) / k
            out.append(a + (b - a) * t[:, None])
        pts = np.concatenate(out)
        return pts + rng.normal(0, jitter, pts.shape)


def fit_box(pts: np.ndarray, w: float, h: float, dy: float = 0.0) -> np.ndarray:
    lo, hi = pts.min(0), pts.max(0)
    s = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    return (pts - (lo + hi) / 2) * s + np.array([CENTER[0], CENTER[1] + dy])


def dimetric(x, y, z):
    return (0.96 * x - 0.64 * y, 0.28 * x + 0.38 * y - z)


def container_state(rng):
    b = ShapeBuilder()
    L, W, H = 2.5, 1.0, 1.05
    P = lambda x, y, z: tuple(92 * np.array(dimetric(x, y, z)))
    corners = {(i, j, k): P(i * L, j * W, k * H) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for a in corners:
        for bkey in corners:
            if sum(x != y for x, y in zip(a, bkey)) == 1 and a < bkey:
                hidden = a == (0, 0, 0) or bkey == (0, 0, 0)
                b.line(corners[a], corners[bkey], 0.9 if hidden else 3.2)
    for i in range(1, 16):  # corrugated long side (y = W face)
        x = i * L / 16
        b.line(P(x, W, 0.05), P(x, W, H - 0.05), 1.05)
    for z in (0.08, H - 0.08):  # top / bottom rails
        b.line(P(0, W, z), P(L, W, z), 1.4)
        b.line(P(L, 0, z), P(L, W, z), 1.4)
    for y in (0.24, 0.47, 0.5, 0.73):  # door end: two leaves, locking bars
        b.line(P(L, y * W, 0.08), P(L, y * W, H - 0.08), 1.6 if y in (0.47, 0.5) else 1.2)
    for y in (0.24, 0.73):
        b.line(P(L, y * W - .05, .45), P(L, y * W + .05, .45), 2.0)
    for i in range(1, 7):  # roof ribs
        x = i * L / 7
        b.line(P(x, 0, H), P(x, W, H), 0.45)
    for key, pt in corners.items():  # corner castings
        if key != (0, 0, 0):
            b.cluster(pt, 3.2, 10)
    return fit_box(b.build(rng), 300, 200)


def yard_state(rng):
    b = ShapeBuilder()
    cx, cy = CENTER
    x0, y0, x1, y1 = cx - 175, cy - 96, cx + 175, cy + 96
    b.polyline([(x0, y0 + 70), (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0 + 126)], 1.3)
    # gate with booths on the west side
    for gy in (y0 + 78, y0 + 118):
        b.cluster((x0, gy), 3.0, 12)
    b.line((x0 - 16, y0 + 98), (x0 + 60, y0 + 98), 1.2)
    b.polyline([(x0 + 8, y0 + 86), (x0 + 22, y0 + 86), (x0 + 22, y0 + 110), (x0 + 8, y0 + 110)], 1.0, True)
    # main lane + cross lanes (dashed)
    for x in np.arange(x0 + 60, x1 - 10, 12):
        b.line((x, cy - 4), (x + 6, cy - 4), 1.0)
    for lx in (cx - 40, cx + 70):
        for y in np.arange(y0 + 10, y1 - 10, 12):
            b.line((lx, y), (lx, y + 6), 0.9)
    # container stacks (north-west block)
    for r in range(3):
        for c in range(5):
            sx, sy = x0 + 64 + c * 22, y0 + 14 + r * 22
            b.polyline([(sx, sy), (sx + 18, sy), (sx + 18, sy + 14), (sx, sy + 14)], 0.75, True)
    # parking rows (south-west block)
    for r in range(2):
        yb = cy + 20 + r * 38
        b.line((x0 + 60, yb + 30 if r == 0 else yb), (cx - 50, yb + 30 if r == 0 else yb), 0.9)
        for x in np.arange(x0 + 60, cx - 48, 11):
            b.line((x, yb + 4), (x, yb + 26), 0.7)
    # warehouse zone (east)
    wx0, wy0, wx1, wy1 = cx + 88, y0 + 16, x1 - 16, cy - 18
    b.polyline([(wx0, wy0), (wx1, wy0), (wx1, wy1), (wx0, wy1)], 2.0, True)
    for x in np.linspace(wx0, wx1, 6)[1:-1]:
        b.line((x, wy0), (x, wy1), 0.6)
    for dx in np.linspace(wx0 + 10, wx1 - 10, 4):
        b.line((dx, wy1), (dx, wy1 + 8), 1.5)
    # storage blocks (south-east)
    for c in range(3):
        sx = cx - 22 + c * 30
        b.polyline([(sx, cy + 18), (sx + 22, cy + 18), (sx + 22, y1 - 16), (sx, y1 - 16)], 0.7, True)
    # route nodes linking zones
    nodes = [(x0 + 40, y0 + 98), (cx - 40, cy - 4), (cx + 70, cy - 4), (cx + 128, cy + 36), (cx + 150, y1 - 24)]
    for a, c in zip(nodes, nodes[1:]):
        b.polyline([a, (c[0], a[1]), c], 0.55)
    for n in nodes:
        b.cluster(n, 3.4, 14)
    return b.build(rng, jitter=0.3)


def architecture_state(rng):
    b = ShapeBuilder()
    cx, cy = CENTER

    def box(x, y, w, h, d=1.6):
        r = 5
        pts = []
        for ang0, (ox, oy) in zip((180, 270, 0, 90), ((x + r, y + r), (x + w - r, y + r), (x + w - r, y + h - r), (x + r, y + h - r))):
            for a in np.radians(np.linspace(ang0, ang0 + 90, 4)):
                pts.append((ox + r * np.cos(a), oy + r * np.sin(a)))
        b.polyline(pts, d, closed=True)
        return (x, y, w, h)

    def cylinder(x, y, w, h, d=1.6):
        t = np.linspace(0, 2 * np.pi, 22)
        top = [(x + w / 2 + w / 2 * np.cos(a), y + 5 * np.sin(a)) for a in t]
        b.polyline(top, d)
        bot = [(x + w / 2 + w / 2 * np.cos(a), y + h + 5 * np.sin(a)) for a in t[: len(t) // 2 + 1]]
        b.polyline(bot, d)
        b.line((x, y), (x, y + h), d)
        b.line((x + w, y), (x + w, y + h), d)
        return (x, y - 5, w, h + 10)

    def link(a, c, d=0.8):
        ax, ay = a[0] + a[2], a[1] + a[3] / 2
        bx, by = c[0], c[1] + c[3] / 2
        mx = (ax + bx) / 2
        b.polyline([(ax, ay), (mx, ay), (mx, by), (bx, by)], d)
        b.cluster((bx - 2, by), 2.2, 5)

    fe1 = box(cx - 185, cy - 78, 62, 36)
    fe2 = box(cx - 185, cy + 36, 62, 36)
    api = box(cx - 95, cy - 24, 58, 48, 2.0)
    be1 = box(cx - 8, cy - 92, 62, 34)
    be2 = box(cx - 8, cy - 17, 62, 34, 1.9)
    be3 = box(cx - 8, cy + 58, 62, 34)
    db = cylinder(cx + 84, cy - 80, 44, 42)
    rd = cylinder(cx + 84, cy + 34, 44, 34)
    ops = box(cx + 150, cy - 26, 40, 52, 1.6)
    for a in (fe1, fe2):
        link(a, api)
    for c in (be1, be2, be3):
        link(api, c)
    link(be1, db)
    link(be2, db)
    link(be3, rd)
    link(be2, rd)
    b.line((db[0] + db[2], cy - 59), (ops[0], cy - 8), 0.6)
    b.line((rd[0] + rd[2], cy + 51), (ops[0], cy + 8), 0.6)
    # inner glyph lines in nodes (no text)
    for x, y, w, h in (fe1, fe2, be1, be2, be3, ops):
        b.line((x + 9, y + h / 2 - 4), (x + w * .62, y + h / 2 - 4), 0.5)
        b.line((x + 9, y + h / 2 + 5), (x + w * .45, y + h / 2 + 5), 0.5)
    x, y, w, h = api
    b.cluster((x + w / 2, y + h / 2), 7, 28)
    return b.build(rng, jitter=0.3)


def spatial_state(rng):
    b = ShapeBuilder()
    cx, cy = CENTER
    ang, tilt = np.radians(35), np.radians(24)

    def proj(x, y, z, s=78):
        xr = x * np.cos(ang) - z * np.sin(ang)
        zr = x * np.sin(ang) + z * np.cos(ang)
        yr = y * np.cos(tilt) - zr * np.sin(tilt)
        zz = y * np.sin(tilt) + zr * np.cos(tilt)
        k = 1.0 - 0.04 * zz  # near-orthographic, just enough depth cue
        return (cx + xr * s * k, cy + 6 - yr * s * k)

    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for i, a in enumerate(v):
        for c in v[i + 1:]:
            if sum(p != q for p, q in zip(a, c)) == 1:
                b.line(proj(*a), proj(*c), 3.0)
    for yl in (-1, -1 / 3, 1 / 3):  # layered inner grid planes
        for t in np.linspace(-1, 1, 5):
            dens = 0.55 if yl == -1 else 0.32
            b.line(proj(t, yl, -1), proj(t, yl, 1), dens)
            b.line(proj(-1, yl, t), proj(1, yl, t), dens)
    floor = [proj(x, -1.35, z) for x, z in ((-1.8, -1.8), (1.8, -1.8), (1.8, 1.8), (-1.8, 1.8))]
    b.polyline(floor, 0.7, closed=True)  # ground plate
    for t in (-0.6, 0.6):
        b.line(proj(t * 3, -1.35, -1.8), proj(t * 3, -1.35, 1.8), 0.35)
        b.line(proj(-1.8, -1.35, t * 3), proj(1.8, -1.35, t * 3), 0.35)
    b.cluster(proj(0, 0.25, 0), 6, 36)  # spatial core
    for a in v:
        b.cluster(proj(*a), 2.8, 8)
    return fit_box(b.build(rng, jitter=0.25), 330, 206)


# --------------------------------------------------------------------------- matching


def match_states(states: list[np.ndarray], passes: int = 4) -> list[np.ndarray]:
    """Deterministic assignment so every particle index flows smoothly around the loop.

    States[0] (portrait) keeps its order. Each other state is reassigned against both
    loop neighbours (previous and next state) with an optimal assignment on squared
    distance, which keeps trajectories short and avoids long crossing paths.
    """
    k = len(states)
    out = [states[0]] + [s.copy() for s in states[1:]]
    for i in range(1, k):  # forward chain
        cost = ((out[i - 1][:, None, :] - out[i][None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        out[i] = out[i][col]
    for _ in range(passes):  # refine against both neighbours
        for i in range(1, k):
            prev, nxt = out[i - 1], out[(i + 1) % k]
            cost = ((prev[:, None, :] - out[i][None, :, :]) ** 2).sum(-1)
            cost += ((nxt[:, None, :] - out[i][None, :, :]) ** 2).sum(-1)
            _, col = linear_sum_assignment(cost)
            out[i] = out[i][col]
    return out


# --------------------------------------------------------------------------- SVG output


def fmt(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


@dataclass
class Theme:
    tiers: list[tuple[str, float]]  # (hex, share), ordered from most to least detailed particles
    sprinkle: list[tuple[str, float]]  # highlight colours assigned at random, independent of detail
    opacity_range: tuple[float, float]


DARK = Theme([("#D7E6F5", .55), ("#9AB7D1", .30), ("#6F7FA3", .15)],
             [("#78DCE8", .07), ("#22D3EE", .05), ("#8B5CF6", .02)], (.5, 1.0))
LIGHT = Theme([("#1F2937", .55), ("#334155", .30), ("#64748B", .15)],
              [("#0E7490", .07), ("#0891B2", .05), ("#7C3AED", .02)], (.62, 1.0))


def particle_block(paths: list[np.ndarray], detail: np.ndarray, theme: Theme, rng: np.random.Generator) -> str:
    key_times = ";".join(fmt(t / DUR) if t in (0, DUR) else f"{t / DUR:.4f}".rstrip("0") for t in KEY_SECONDS)
    splines = ";".join(SPLINES)
    order = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0]

    colors = [c for c, _ in theme.tiers] + [c for c, _ in theme.sprinkle]
    shares = np.array([s for _, s in theme.tiers])
    cuts = np.cumsum(shares / shares.sum()) * N
    by_detail = np.argsort(-(detail + rng.normal(0, .12, N)))
    color_idx = np.empty(N, int)
    color_idx[by_detail] = np.searchsorted(cuts, np.arange(N), side="right")
    pool = rng.permutation(N)
    start = 0
    for k, (_, share) in enumerate(theme.sprinkle):
        count = int(N * share)
        color_idx[pool[start:start + count]] = len(theme.tiers) + k
        start += count

    radius = 0.55 + 1.0 * np.clip(detail + rng.normal(0, .12, N), 0, 1)
    lo, hi = theme.opacity_range
    opacity = lo + (hi - lo) * np.clip(0.35 + 0.65 * detail + rng.normal(0, .15, N), 0, 1)

    groups: dict[int, list[str]] = {}
    for i in range(N):
        vals = ";".join(f"{fmt(paths[s][i, 0])} {fmt(paths[s][i, 1])}" for s in order)
        groups.setdefault(color_idx[i], []).append(
            f'<circle r="{radius[i]:.2f}" opacity="{opacity[i]:.2f}"><animateTransform attributeName="transform" '
            f'dur="{DUR}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" '
            f'keySplines="{splines}" values="{vals}"/></circle>'
        )
    lines = []
    for ci, hex_color in enumerate(colors):
        if ci in groups:
            lines.append(f'<g fill="{hex_color}">' + "".join(groups[ci]) + "</g>")
    return "\n".join(lines)


def inject(svg_path: Path, block: str) -> None:
    text = svg_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- PARTICLES:BEGIN -->)(.*?)(<!-- PARTICLES:END -->)", re.S)
    if not pattern.search(text):
        raise RuntimeError(f"{svg_path.name} has no PARTICLES markers")
    text = pattern.sub(lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), text)
    svg_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {svg_path.relative_to(ROOT)} ({svg_path.stat().st_size / 1024:.0f} KB)")


def render_debug(states, detail, theme: str):
    bg, fg = ((34, 20, 11), (245, 230, 215)) if theme == "dark" else ((250, 248, 246), (55, 41, 31))
    names = ["portrait", "container", "yard", "architecture", "spatial"]
    x, y, w, h = VIEW
    s = 3
    tiles = []
    for pts, name in zip(states, names):
        img = np.full((h * s, w * s, 3), bg, np.uint8)
        for (px, py), d in zip(pts, detail):
            cv2.circle(img, (int((px - x) * s), int((py - y) * s)), max(1, int((0.55 + d) * s)), fg, -1, cv2.LINE_AA)
        cv2.putText(img, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (178, 145, 8), 2)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:] + [np.full_like(tiles[0], bg)])])
    cv2.imwrite(str(DEBUG_DIR / f"particle-states-{theme}.png"), grid)
    s = 4
    ox, oy = CENTER[0] - PORTRAIT_BOX[0] / 2 - 6, CENTER[1] - PORTRAIT_BOX[1] / 2 - 6
    img = np.full((int((PORTRAIT_BOX[1] + 12) * s), int((PORTRAIT_BOX[0] + 12) * s), 3), bg, np.uint8)
    for (px, py), d in zip(states[0], detail):
        cv2.circle(img, (int((px - ox) * s), int((py - oy) * s)), max(1, int((0.55 + d) * s)), fg, -1, cv2.LINE_AA)
    cv2.imwrite(str(DEBUG_DIR / f"portrait-particles-{theme}.png"), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="write inspection images to tmp/")
    parser.add_argument("--no-write", action="store_true", help="skip rewriting the SVG assets")
    args = parser.parse_args()
    if args.debug:
        DEBUG_DIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    maps = build_portrait_maps(args.debug)
    shapes = [container_state(rng), yard_state(rng), architecture_state(rng), spatial_state(rng)]

    for offset, (name, theme, svg) in enumerate((("dark", DARK, SVG_DARK), ("light", LIGHT, SVG_LIGHT)), 1):
        portrait, detail = sample_portrait(maps, name, np.random.default_rng(SEED + 10 * offset))
        states = [portrait] + shapes
        for label, st in zip(STATE_NAMES, states):
            assert st.shape == (N, 2), (label, st.shape)
        paths = match_states(states)
        if args.debug:
            render_debug(paths, detail, name)
            print(f"[{name}]")
            for i in range(5):
                d = np.linalg.norm(paths[(i + 1) % 5] - paths[i], axis=1)
                print(f"  {STATE_NAMES[i]:>12} -> {STATE_NAMES[(i + 1) % 5]:<12} mean {d.mean():6.1f}  p95 {np.percentile(d, 95):6.1f}")
        if not args.no_write:
            inject(svg, particle_block(paths, detail, theme, np.random.default_rng(SEED + offset)))


if __name__ == "__main__":
    main()
