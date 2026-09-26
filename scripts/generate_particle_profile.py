"""Build-time generator for the animated stippled hero in assets/profile-terminal-*.svg.

Reads three private source images from the repository root (all git-ignored):
YASEEN.jpeg, VVV.jpg and ALX.jpg. Each one is turned into a dense weighted-Voronoi
stipple rendering. The particle block written between the PARTICLES:BEGIN /
PARTICLES:END markers of both hero SVGs contains:

* one static detail layer per image (compact round-capped path dots) that
  crossfades in and out with SMIL group opacity, and
* a moving subset of circles, matched between the images, that morphs
  YASEEN -> VVV -> ALX -> YASEEN in a seamless loop.

Usage (from the repository root):
    .venv/Scripts/python scripts/generate_particle_profile.py            # write both hero SVGs
    .venv/Scripts/python scripts/generate_particle_profile.py --preview  # tmp/ previews only

Nothing produced here is needed at runtime: the SVGs are self-contained, use no
JavaScript and never embed the source images or any raster derived from them.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = ROOT / "tmp"
SVG_DARK = ROOT / "assets" / "profile-terminal-dark.svg"
SVG_LIGHT = ROOT / "assets" / "profile-terminal-light.svg"

SEED = 42
N_MOVING = 1600

# SVG layout: the visual viewport inside the 1200x520 hero.
VIEW = (48, 92, 432, 272)
CENTER = (264.0, 228.0)
FIT_HEIGHT = 240.0  # SVG units each image is scaled to (~88% of the viewport height)
RASTER_PX = 4.0  # working raster pixels per SVG unit
LLOYD_ITERATIONS = 40

DUR = 15
KEY_SECONDS = [0, 3.4, 5.0, 8.4, 10.0, 13.4, 15.0]
HOLD, EASE = "0 0 1 1", ".45 0 .25 1"
SPLINES = [HOLD, EASE, HOLD, EASE, HOLD, EASE]
FADE_SHARE = 0.5  # share of each morph used by the detail-layer crossfade

PALETTES = {
    "dark": [("#B8C4D8", .30), ("#A7B6CF", .25), ("#8EA0BF", .20), ("#8792D0", .11), ("#9B8ED8", .10), ("#22D3EE", .04)],
    "light": [("#1E293B", .30), ("#334155", .25), ("#475569", .20), ("#3B3F7A", .11), ("#4C3F8F", .10), ("#0E7490", .04)],
}
# Per theme: base radius, radius span, base opacity, opacity gamma.
STYLE = {"dark": (0.21, 0.17, 0.50, 0.6), "light": (0.19, 0.16, 0.38, 0.8)}


# --------------------------------------------------------------------------- masks


def _remove_background(bgr: np.ndarray) -> np.ndarray:
    from rembg import new_session, remove

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for model in ("u2net_human_seg", "u2net"):
        try:
            return np.asarray(remove(rgb, session=new_session(model), only_mask=True))
        except Exception as exc:  # model download / runtime failure
            print(f"rembg model {model} failed: {exc}")
    raise RuntimeError("rembg could not produce a subject mask")


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    flood = np.pad(binary, 1).astype(np.uint8) * 255
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    return (flood[1:-1, 1:-1] != 128)


def _drop_small(binary: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def mask_rembg(bgr: np.ndarray) -> np.ndarray:
    return _remove_background(bgr)


def mask_bright_on_black(bgr: np.ndarray) -> np.ndarray:
    """Bust and lettering on a black ground: threshold, close, fill, keep letters and commas."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    solid = gray > 30
    solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    solid = _drop_small(_fill_holes(solid), 60)
    return cv2.GaussianBlur(solid.astype(np.float32) * 255, (0, 0), 1.0).astype(np.uint8)


def mask_ink_on_paper(bgr: np.ndarray) -> np.ndarray:
    """Engraving on paper: close the ink into one silhouette so paper texture outside is dropped."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    ink = gray < 165
    solid = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    solid = _drop_small(_fill_holes(solid), 4000)
    solid = cv2.dilate(solid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    return cv2.GaussianBlur(solid.astype(np.float32) * 255, (0, 0), 2.0).astype(np.uint8)


# --------------------------------------------------------------------------- sources


# tone(lum, raw, mask): lum is the (optionally CLAHE-equalised) luminance, raw the plain luminance.
ToneFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Source:
    key: str
    path: Path
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels
    mask: Callable[[np.ndarray], np.ndarray]
    tone: dict[str, ToneFn]  # theme -> tone in 0..1, "denser" where larger
    n_detail: int = 11000
    clahe: float = 2.6  # CLAHE clip limit; 0 disables local equalisation
    tone_blur: float = 0.0  # extra blur (raster px) applied to luminance before the tone curve
    weights: dict[str, float] = field(default_factory=lambda: {"tone": 0.65, "lc": 0.20, "sil": 0.10, "edge": 0.05})
    flat_damping: float = 0.0  # tone reduction in large low-texture areas
    bottom_fade: bool = False  # soft fade where the photo frame cuts the body
    radius_scale: float = 1.0


VVV_TEXT_BAND = 0.83  # share of the VVV crop height where the lettering starts


def _vvv_letters(lum: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lettering band mask and solid letter tone from raw luminance.

    CLAHE lifts the black letter counters to grey and the hole-filled mask closes them,
    so the lettering follows its own unequalised brightness.
    """
    band = (np.arange(lum.shape[0]) >= VVV_TEXT_BAND * lum.shape[0])[:, None]
    return band, np.clip((raw - 0.25) / 0.3, 0, 1)


def vvv_dark_tone(lum: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    band, letters = _vvv_letters(lum, raw)
    return np.where(band, letters, lum ** 1.15)


def vvv_light_tone(lum: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dark ink on a light ground: shaded stone for the bust, solid ink for the pale lettering."""
    band, letters = _vvv_letters(lum, raw)
    stone = np.clip((0.85 - lum) / 0.6, 0, 1)
    return np.where(band, letters, 0.08 + 0.92 * stone)


SOURCES = [
    Source(
        key="yaseen",
        path=ROOT / "YASEEN.jpeg",
        crop=(0, 40, 1170, 1330),
        mask=mask_rembg,
        # Light dots on a dark ground emit light, so brighter = denser; dark dots on a light ground use ink density.
        tone={"dark": lambda lum, raw, m: 0.08 + 0.92 * lum ** 1.6, "light": lambda lum, raw, m: 1 - lum},
        flat_damping=0.4,
        bottom_fade=True,
    ),
    Source(
        key="vvv",
        path=ROOT / "VVV.jpg",
        crop=(100, 130, 640, 915),
        mask=mask_bright_on_black,
        tone={"dark": vvv_dark_tone, "light": vvv_light_tone},
        n_detail=13500,
        weights={"tone": 0.62, "lc": 0.18, "sil": 0.15, "edge": 0.05},
    ),
    Source(
        key="alx",
        path=ROOT / "ALX.jpg",
        crop=(0, 100, 736, 905),
        mask=mask_ink_on_paper,
        # Line art: hatching is averaged into tone (it cannot resolve at hero size), then stretched so
        # shadowed masses stay dense and lit paper stays open; the edge term keeps the contours.
        tone={"dark": lambda lum, raw, m: np.clip((1 - lum - 0.08) / 0.6, 0, 1) ** 1.3,
              "light": lambda lum, raw, m: np.clip((1 - lum - 0.08) / 0.6, 0, 1) ** 1.3},
        clahe=0,
        tone_blur=3.0,
        weights={"tone": 0.72, "lc": 0.10, "sil": 0.06, "edge": 0.12},
        radius_scale=0.9,
    ),
]


@dataclass
class Field:
    source: Source
    density: dict[str, np.ndarray]  # per theme
    crop_bgr: np.ndarray
    mask: np.ndarray
    origin: tuple[float, float]  # SVG position of the raster's top-left


def build_field(src: Source) -> Field:
    bgr = cv2.imread(str(src.path))
    if bgr is None:
        raise FileNotFoundError(f"missing private source image: {src.path}")
    alpha = src.mask(bgr)

    x0, y0, x1, y1 = src.crop
    wh = int(round(FIT_HEIGHT * RASTER_PX))
    ww = int(round((x1 - x0) * FIT_HEIGHT / (y1 - y0) * RASTER_PX))
    crop = cv2.resize(bgr[y0:y1, x0:x1], (ww, wh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(alpha[y0:y1, x0:x1], (ww, wh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    raw = cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), 0.8)
    if src.clahe:
        gray = cv2.createCLAHE(clipLimit=src.clahe, tileGridSize=(8, 8)).apply(gray)
    lum = cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), 0.8)
    inside = mask > .5

    lc = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 6))
    lc = np.clip(lc / (np.percentile(lc[inside], 99) + 1e-6), 0, 1)
    quiet = np.ones_like(lum)
    if src.flat_damping:
        var = cv2.GaussianBlur(lum ** 2, (0, 0), 10) - cv2.GaussianBlur(lum, (0, 0), 10) ** 2
        texture = np.sqrt(np.maximum(var, 0))
        texture = np.clip(texture / (np.percentile(texture[inside], 90) + 1e-6), 0, 1)
        quiet = 1 - src.flat_damping * (1 - texture)
    pad = 20
    solid = cv2.copyMakeBorder(inside.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
    sil = np.exp(-dist / (1.5 * RASTER_PX)) * inside
    edge = cv2.GaussianBlur(cv2.Canny((lum * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255, (0, 0), 1.5)
    edge = np.clip(edge / (edge.max() + 1e-6), 0, 1)

    fade = np.ones_like(lum)
    if src.bottom_fade:
        yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
        fade = np.clip((wh - yy) / (0.12 * wh), 0, 1)
        side = np.minimum(xx, ww - 1 - xx) / (0.07 * ww)
        fade *= np.where(yy > 0.62 * wh, np.clip(side, 0, 1), 1.0)

    w = src.weights
    density = {}
    tone_lum = cv2.GaussianBlur(lum, (0, 0), src.tone_blur) if src.tone_blur else lum
    for theme, tone_fn in src.tone.items():
        tone = np.clip(tone_fn(tone_lum, raw, mask), 0, 1)
        d = (w["tone"] * tone * quiet + w["lc"] * lc + w["sil"] * sil + w["edge"] * edge) * mask * fade
        density[theme] = np.clip(d, 0, None).astype(np.float32)

    origin = (CENTER[0] - ww / RASTER_PX / 2, CENTER[1] - wh / RASTER_PX / 2)
    return Field(source=src, density=density, crop_bgr=crop, mask=mask, origin=origin)


# --------------------------------------------------------------------------- stippling


def stipple(density: np.ndarray, n: int, rng: np.random.Generator, iterations: int = LLOYD_ITERATIONS) -> np.ndarray:
    """Weighted Voronoi stippling: weighted random init, then density-weighted Lloyd relaxation."""
    ys, xs = np.nonzero(density > 1e-4)
    wts = density[ys, xs].astype(np.float64)
    pix = np.stack([xs, ys], 1).astype(np.float64) + 0.5
    pts = pix[rng.choice(len(wts), size=n, replace=False, p=wts / wts.sum())]
    pts += rng.random(pts.shape) - 0.5
    for _ in range(iterations):
        _, owner = cKDTree(pts).query(pix, workers=-1)
        mass = np.bincount(owner, weights=wts, minlength=n)
        cx = np.bincount(owner, weights=wts * pix[:, 0], minlength=n)
        cy = np.bincount(owner, weights=wts * pix[:, 1], minlength=n)
        live = mass > 0
        pts[live, 0] = cx[live] / mass[live]
        pts[live, 1] = cy[live] / mass[live]
    return pts


def style_stipple(pts_px: np.ndarray, density: np.ndarray, theme: str, rng: np.random.Generator, radius_scale: float = 1.0):
    h, w = density.shape
    ix = np.clip(pts_px[:, 0].astype(int), 0, w - 1)
    iy = np.clip(pts_px[:, 1].astype(int), 0, h - 1)
    local = cv2.GaussianBlur(density, (0, 0), 3)[iy, ix]
    dn = np.clip(local / (np.percentile(local, 98) + 1e-6), 0, 1)
    n = len(pts_px)
    base_r, span_r, base_o, gamma_o = STYLE[theme]
    radius = (base_r + span_r * dn + rng.normal(0, 0.025, n)) * radius_scale
    bump = rng.random(n) < 0.03
    radius[bump] += 0.14
    radius = np.clip(radius, 0.16, 0.55)
    opacity = np.clip(base_o + (0.95 - base_o) * dn ** gamma_o + rng.normal(0, 0.05, n), 0.35, 0.95)
    palette = PALETTES[theme]
    shares = np.array([s for _, s in palette])
    color = rng.choice(len(palette), size=n, p=shares / shares.sum())
    return radius, opacity, color


@dataclass
class Rendering:
    detail_xy: np.ndarray
    radius: np.ndarray
    opacity: np.ndarray
    color: np.ndarray
    moving_xy: np.ndarray


def render_source(fld: Field, theme: str) -> Rendering:
    dens = fld.density[theme]
    origin = np.array(fld.origin)
    pts = stipple(dens, fld.source.n_detail, np.random.default_rng(SEED))
    radius, opacity, color = style_stipple(pts, dens, theme, np.random.default_rng(SEED + 7), fld.source.radius_scale)
    moving = stipple(dens, N_MOVING, np.random.default_rng(SEED + 1), iterations=25)
    return Rendering(pts / RASTER_PX + origin, radius, opacity, color, moving / RASTER_PX + origin)


# --------------------------------------------------------------------------- matching


def match_states(states: list[np.ndarray], passes: int = 4) -> list[np.ndarray]:
    """Deterministic assignment so every particle index flows smoothly around the loop.

    States[0] keeps its order. Each other state is reassigned against both loop
    neighbours with an optimal assignment on squared distance, which keeps
    trajectories short and avoids long crossing paths.
    """
    k = len(states)
    out = [states[0]] + [s.copy() for s in states[1:]]
    for i in range(1, k):
        cost = ((out[i - 1][:, None, :] - out[i][None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        out[i] = out[i][col]
    for _ in range(passes):
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


def _tenths(t: int) -> str:
    sign = "-" if t < 0 else ""
    whole, frac = divmod(abs(t), 10)
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole if whole else ''}.{frac}"


def dots_path(xy: np.ndarray) -> str:
    """Zero-length round-capped segments: ~10 bytes per dot, positions exact to 0.1 unit."""
    t = np.round(xy * 10).astype(int)
    order = np.lexsort((np.where((t[:, 1] // 40) % 2, -t[:, 0], t[:, 0]), t[:, 1] // 40))
    t = t[order]
    parts = [f"M{_tenths(t[0, 0])} {_tenths(t[0, 1])}h0"]
    for (dx, dy) in np.diff(t, axis=0):
        sx, sy = _tenths(int(dx)), _tenths(int(dy))
        parts.append(f"m{sx}{sy if sy.startswith('-') else ' ' + sy}h0")
    return "".join(parts)


def detail_paths(r: Rendering, theme: str) -> str:
    palette = PALETTES[theme]
    r_bins = np.array([0.22, 0.30, 0.40, 0.55])
    o_bins = np.array([0.5, 0.65, 0.8, 0.95])
    ri = np.searchsorted(r_bins, r.radius)
    oi = np.searchsorted(o_bins, r.opacity)
    out = []
    for c, (hex_color, _) in enumerate(palette):
        paths = []
        for a in range(len(r_bins)):
            for b in range(len(o_bins)):
                sel = (r.color == c) & (ri == a) & (oi == b)
                if not sel.any():
                    continue
                width = 2 * r.radius[sel].mean()
                paths.append(f'<path stroke-width="{width:.2f}" stroke-opacity="{r.opacity[sel].mean():.2f}" d="{dots_path(r.detail_xy[sel])}"/>')
        if paths:
            out.append(f'<g stroke="{hex_color}">' + "".join(paths) + "</g>")
    return "".join(out)


def _key_times(seconds: list[float]) -> str:
    return ";".join(fmt(t / DUR) if t in (0, DUR) else f"{t / DUR:.4f}".rstrip("0") for t in seconds)


def layer_opacity(index: int, count: int) -> tuple[str, str, str]:
    """(initial opacity, values, keyTimes) so layer `index` shows during its hold and crossfades with neighbours."""
    hold_start = [KEY_SECONDS[2 * i] for i in range(count)]
    hold_end = [KEY_SECONDS[2 * i + 1] for i in range(count)]
    morph = KEY_SECONDS[2] - KEY_SECONDS[1]
    fade = FADE_SHARE * morph
    if index == 0:
        values, times = [1, 1, 0, 0, 1], [0, hold_end[0], hold_end[0] + fade, DUR - fade, DUR]
    else:
        s, e = hold_start[index], hold_end[index]
        values, times = [0, 0, 1, 1, 0, 0], [0, s - fade, s, e, e + fade, DUR]
    return str(values[0]), ";".join(map(str, values)), _key_times(times)


def moving_circles(paths: list[np.ndarray], theme: str, rng: np.random.Generator) -> str:
    palette = PALETTES[theme]
    shares = np.array([s for _, s in palette])
    color = rng.choice(len(palette), size=N_MOVING, p=shares / shares.sum())
    radius = np.clip(0.36 + rng.normal(0, 0.05, N_MOVING), 0.26, 0.5)
    opacity = np.clip(0.8 + rng.normal(0, 0.08, N_MOVING), 0.55, 0.95)
    order = [0, 0, 1, 1, 2, 2, 0]
    key_times = _key_times(KEY_SECONDS)
    splines = ";".join(SPLINES)
    groups: dict[int, list[str]] = {}
    for i in range(N_MOVING):
        vals = ";".join(f"{fmt(paths[s][i, 0])} {fmt(paths[s][i, 1])}" for s in order)
        groups.setdefault(int(color[i]), []).append(
            f'<circle r="{radius[i]:.2f}" opacity="{opacity[i]:.2f}"><animateTransform attributeName="transform" '
            f'dur="{DUR}s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" '
            f'keySplines="{splines}" values="{vals}"/></circle>'
        )
    return "\n".join(f'<g fill="{palette[c][0]}">' + "".join(items) + "</g>" for c, items in sorted(groups.items()))


def particle_block(renders: list[Rendering], theme: str) -> str:
    layers = []
    for i, r in enumerate(renders):
        initial, values, times = layer_opacity(i, len(renders))
        layers.append(
            f'<g opacity="{initial}"><animate attributeName="opacity" values="{values}" keyTimes="{times}" '
            f'dur="{DUR}s" repeatCount="indefinite"/>{detail_paths(r, theme)}</g>'
        )
    paths = match_states([r.moving_xy for r in renders])
    moving = moving_circles(paths, theme, np.random.default_rng(SEED + 11))
    return '<g fill="none" stroke-linecap="round">\n' + "\n".join(layers) + "\n</g>\n" + moving


def inject(svg_path: Path, block: str) -> None:
    text = svg_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- PARTICLES:BEGIN -->)(.*?)(<!-- PARTICLES:END -->)", re.S)
    if not pattern.search(text):
        raise RuntimeError(f"{svg_path.name} has no PARTICLES markers")
    text = pattern.sub(lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), text)
    svg_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {svg_path.relative_to(ROOT)} ({svg_path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- previews


def _panel_svg(theme: str, body: str) -> str:
    bg, grid, stroke, tag = (("#0b1422", "#1a2940", "#263a55", "#51627f") if theme == "dark"
                             else ("#fbfcfd", "#e3e9f0", "#d0d7de", "#8c959f"))
    x, y, w, h = VIEW
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}">'
        f'<defs><pattern id="g" width="24" height="24" patternUnits="userSpaceOnUse">'
        f'<path d="M24 0H0V24" fill="none" stroke="{grid}"/></pattern></defs>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{bg}" stroke="{stroke}"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="url(#g)" opacity=".8"/>'
        f'<g fill="{tag}" font-family="JetBrains Mono,Consolas,monospace" font-size="10" font-weight="700" letter-spacing="1.5">'
        f'<text x="76" y="113">WMS · ERP</text><text x="452" y="113" text-anchor="end">YARD · GATE</text>'
        f'<text x="76" y="349">API · DB</text><text x="452" y="349" text-anchor="end">OPS · XR</text></g>'
        f"{body}</svg>"
    )


def write_previews(fields: list[Field], renders: dict[str, list[Rendering]]) -> None:
    for fld in fields:
        key = fld.source.key
        flat = np.full_like(fld.crop_bgr, (28, 22, 18))
        cv2.imwrite(str(DEBUG_DIR / f"src-{key}-crop.png"), np.where((fld.mask > .5)[..., None], fld.crop_bgr, flat))
        for theme, dens in fld.density.items():
            cv2.imwrite(str(DEBUG_DIR / f"src-{key}-density-{theme}.png"), (255 * dens / dens.max()).astype(np.uint8))
    for theme, rs in renders.items():
        palette = PALETTES[theme]
        for fld, r in zip(fields, rs):
            dots = f'<g fill="none" stroke-linecap="round">{detail_paths(r, theme)}</g>'
            moving = "".join(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r=".36" fill="{palette[0][0]}" opacity=".8"/>' for x, y in r.moving_xy)
            (DEBUG_DIR / f"panel-{fld.source.key}-{theme}.svg").write_text(_panel_svg(theme, dots + moving), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="write per-image previews to tmp/ and skip the hero SVGs")
    args = parser.parse_args()

    fields = [build_field(src) for src in SOURCES]
    renders = {theme: [render_source(f, theme) for f in fields] for theme in ("dark", "light")}
    for theme, rs in renders.items():
        for fld, r in zip(fields, rs):
            y = r.detail_xy[:, 1]
            print(f"[{theme}] {fld.source.key}: {len(r.detail_xy)} detail + {len(r.moving_xy)} moving, "
                  f"height {np.ptp(y):.0f} units, radius mean {r.radius.mean():.2f}")

    if args.preview:
        DEBUG_DIR.mkdir(exist_ok=True)
        write_previews(fields, renders)
        return
    inject(SVG_DARK, particle_block(renders["dark"], "dark"))
    inject(SVG_LIGHT, particle_block(renders["light"], "light"))


if __name__ == "__main__":
    main()
