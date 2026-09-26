"""Generate the themed README cards in assets/ (standard library only).

* edge-radar-{dark,light}.svg  - "MY EDGE" radar charts from the EDGE config below
* stats-{dark,light}.svg       - "LET THE NUMBERS TALK" metrics from the GitHub API
* languages-{dark,light}.svg   - "Most Used Languages" bar and list from the GitHub API

Metrics that cannot be fetched are omitted, never estimated. Contributions come from the
GraphQL contributionsCollection and are only shown when that query returns a valid count
(it needs a token, and the default Actions GITHUB_TOKEN may not be allowed to read it).

Usage (from the repository root):
    python scripts/generate_readme_assets.py            # radar + live GitHub cards
    python scripts/generate_readme_assets.py --offline  # radar only

Environment: GITHUB_USER (default albgoor), STATS_TOKEN or GITHUB_TOKEN (optional).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
USER = os.environ.get("GITHUB_USER", "albgoor")
TOKEN = os.environ.get("STATS_TOKEN") or os.environ.get("GITHUB_TOKEN")
API = "https://api.github.com"
FONT = "'JetBrains Mono','SFMono-Regular',Consolas,monospace"

# Self-assessed, 0-10. Edit these values to adjust the MY EDGE charts.
EDGE = [
    ("FULL-STACK CAPABILITY", [("Frontend", 8.5), ("Backend", 8.0), ("Databases", 7.5), ("DevOps", 6.5), ("3D / XR", 7.0)]),
    ("DOMAIN / TECHNICAL FOCUS", [("ERP", 8.5), ("Logistics", 8.5), ("System Design", 7.5), ("Problem Solving", 8.5), ("Product Thinking", 7.5)]),
]

THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#0b1422", "border": "#22324a", "grid": "#1b2a3e", "text": "#e6edf3",
        "muted": "#8b9bb4", "faint": "#51627f", "accent": "#22d3ee", "accent2": "#8792d0", "good": "#34d399",
    },
    "light": {
        "bg": "#f6f8fa", "panel": "#ffffff", "border": "#d0d7de", "grid": "#e3e9f0", "text": "#1f2328",
        "muted": "#57606a", "faint": "#8c959f", "accent": "#0e7490", "accent2": "#4c3f8f", "good": "#15803d",
    },
}

LANG_COLORS = {
    "TypeScript": "#3178c6", "JavaScript": "#f1e05a", "Python": "#3572A5", "HTML": "#e34c26", "CSS": "#663399",
    "SCSS": "#c6538c", "Java": "#b07219", "C#": "#178600", "C++": "#f34b7d", "C": "#555555", "Go": "#00ADD8",
    "Rust": "#dea584", "PHP": "#4F5D95", "Ruby": "#701516", "Shell": "#89e051", "PowerShell": "#012456",
    "Dockerfile": "#384d54", "Vue": "#41b883", "Svelte": "#ff3e00", "Kotlin": "#A97BFF", "Swift": "#F05138",
    "Dart": "#00B4AB", "GLSL": "#5686a5", "PLpgSQL": "#336790", "TSQL": "#e38c00", "MDX": "#fcb32c",
}
FALLBACK_COLORS = ["#22d3ee", "#8792d0", "#34d399", "#f6c453", "#ff6b6b", "#9b8ed8", "#8ea0bf"]
STAMP = re.compile(r"<!--updated-->.*?<!--/updated-->")


# --------------------------------------------------------------------------- GitHub data


def _request(url: str, body: bytes | None = None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": f"{USER}-readme-assets"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_contributions() -> int | None:
    if not TOKEN:
        print("contributions: no token, metric omitted")
        return None
    query = {"query": "query($login:String!){user(login:$login){contributionsCollection{contributionCalendar{totalContributions}}}}",
             "variables": {"login": USER}}
    try:
        data = _request(f"{API}/graphql", json.dumps(query).encode("utf-8"))
        total = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["totalContributions"]
    except (urllib.error.URLError, KeyError, TypeError, ValueError) as exc:
        print(f"contributions: unavailable ({exc}), metric omitted")
        return None
    if data.get("errors") or not isinstance(total, int) or total < 0:
        print("contributions: invalid response, metric omitted")
        return None
    return total


def fetch_github() -> tuple[list[tuple[str, int]], dict[str, int]]:
    """Return (metrics, language bytes). Raises on REST failure so stale cards are kept."""
    user = _request(f"{API}/users/{USER}")
    repos, page = [], 1
    while True:
        batch = _request(f"{API}/users/{USER}/repos?type=owner&per_page=100&page={page}")
        repos += batch
        if len(batch) < 100:
            break
        page += 1
    own = [r for r in repos if not r.get("fork")]
    metrics = [
        ("PUBLIC REPOS", int(user["public_repos"])),
        ("STARS EARNED", sum(int(r["stargazers_count"]) for r in own)),
        ("FOLLOWERS", int(user["followers"])),
        ("FOLLOWING", int(user["following"])),
    ]
    contributions = fetch_contributions()
    if contributions is not None:
        metrics.append(("CONTRIBUTIONS · 1Y", contributions))

    languages: dict[str, int] = {}
    for repo in own:
        for name, size in _request(repo["languages_url"]).items():
            languages[name] = languages.get(name, 0) + int(size)
    return metrics, languages


# --------------------------------------------------------------------------- SVG helpers


def _svg(width: int, height: int, title: str, body: str, t: dict) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<title>{escape(title)}</title>'
        f'<style>text{{font-family:{FONT}}}</style>'
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="16" fill="{t["bg"]}" stroke="{t["border"]}" stroke-width="2"/>'
        f"{body}</svg>\n"
    )


def _header(label: str, t: dict, width: int, right: str = "") -> str:
    out = (f'<circle cx="34" cy="34" r="4.5" fill="{t["good"]}"/>'
           f'<text x="48" y="39" fill="{t["text"]}" font-size="14" font-weight="800" letter-spacing="3">{escape(label)}</text>')
    if right:
        out += f'<text x="{width - 32}" y="39" text-anchor="end" fill="{t["faint"]}" font-size="12" letter-spacing="1">{right}</text>'
    return out + f'<path d="M24 58H{width - 24}" stroke="{t["grid"]}"/>'


def _write(name: str, content: str) -> None:
    """Write unless only the update stamp changed, so scheduled runs don't commit noise."""
    path = ASSETS / name
    if path.exists() and STAMP.sub("", path.read_text(encoding="utf-8")) == STAMP.sub("", content):
        print(f"unchanged {name}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {name}")


# --------------------------------------------------------------------------- cards


def radar_svg(theme: str) -> str:
    t = THEMES[theme]
    width, height = 1200, 470
    body = _header("MY EDGE", t, width, "SELF-ASSESSED · 0–10")
    for i, (title, axes) in enumerate(EDGE):
        cx, cy, r = 300 + i * 600, 285, 132
        n = len(axes)
        angles = [-math.pi / 2 + k * 2 * math.pi / n for k in range(n)]
        pt = lambda a, rad: (rad * math.cos(a), rad * math.sin(a))
        g = [f'<text x="{cx}" y="90" text-anchor="middle" fill="{t["accent"]}" font-size="13" font-weight="800" letter-spacing="2.5">{escape(title)}</text>']
        for ring in (0.2, 0.4, 0.6, 0.8, 1.0):
            pts = " ".join(f"{cx + x:.1f},{cy + y:.1f}" for x, y in (pt(a, r * ring) for a in angles))
            g.append(f'<polygon points="{pts}" fill="none" stroke="{t["grid"]}" stroke-width="{1.5 if ring == 1 else 1}"/>')
        for a in angles:
            x, y = pt(a, r)
            g.append(f'<path d="M{cx} {cy}L{cx + x:.1f} {cy + y:.1f}" stroke="{t["grid"]}"/>')
        shape = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(a, r * v / 10) for a, (_, v) in zip(angles, axes)))
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{t["accent"]}"/>'
                       for x, y in (pt(a, r * v / 10) for a, (_, v) in zip(angles, axes)))
        g.append(
            f'<g transform="translate({cx} {cy})"><g>'
            f'<animateTransform attributeName="transform" type="scale" values="0;1" dur="1.4s" begin="{0.2 + i * 0.25}s" '
            f'calcMode="spline" keyTimes="0;1" keySplines=".2 .8 .2 1" fill="freeze"/>'
            f'<polygon points="{shape}" fill="url(#fill{i})" stroke="{t["accent"]}" stroke-width="2.5" stroke-linejoin="round"/>{dots}</g></g>'
        )
        for a, (label, v) in zip(angles, axes):
            x, y = pt(a, r + 30)
            anchor = "middle" if abs(x) < 20 else ("start" if x > 0 else "end")
            dy = -6 if y < -r * 0.9 else (14 if y > r * 0.5 else 4)
            g.append(f'<text x="{cx + x:.1f}" y="{cy + y + dy:.1f}" text-anchor="{anchor}" fill="{t["text"]}" font-size="14" font-weight="700">'
                     f'{escape(label)} <tspan fill="{t["accent"]}">{v:g}</tspan></text>')
        body += (f'<defs><linearGradient id="fill{i}" x1="0" y1="0" x2="1" y2="1">'
                 f'<stop offset="0" stop-color="{t["accent"]}" stop-opacity=".32"/>'
                 f'<stop offset="1" stop-color="{t["accent2"]}" stop-opacity=".22"/></linearGradient></defs>' + "".join(g))
    body += f'<path d="M600 84V440" stroke="{t["grid"]}" stroke-dasharray="2 7"/>'
    return _svg(width, height, "My edge: self-assessed full-stack capability and domain focus radar charts", body, t)


def stats_svg(theme: str, metrics: list[tuple[str, int]], stamp: str) -> str:
    t = THEMES[theme]
    width, height = 1200, 210
    body = _header(f"GITHUB · @{USER}", t, width, f"<!--updated-->UPDATED {stamp}<!--/updated-->")
    gap, x0 = 16, 24
    tile = (width - 2 * x0 - gap * (len(metrics) - 1)) / len(metrics)
    for i, (label, value) in enumerate(metrics):
        x = x0 + i * (tile + gap)
        body += (f'<rect x="{x:.1f}" y="78" width="{tile:.1f}" height="108" rx="12" fill="{t["panel"]}" stroke="{t["border"]}"/>'
                 f'<path d="M{x + 18:.1f} 78h36" stroke="{t["accent"]}" stroke-width="3"/>'
                 f'<text x="{x + 20:.1f}" y="140" fill="{t["text"]}" font-size="40" font-weight="800">{value:,}</text>'
                 f'<text x="{x + 20:.1f}" y="168" fill="{t["muted"]}" font-size="12" font-weight="700" letter-spacing="1.5">{escape(label)}</text>')
    summary = ", ".join(f"{label.lower()} {value}" for label, value in metrics)
    return _svg(width, height, f"GitHub numbers for {USER}: {summary}", body, t)


def languages_svg(theme: str, languages: dict[str, int], stamp: str) -> str:
    t = THEMES[theme]
    total = sum(languages.values())
    ranked = sorted(languages.items(), key=lambda kv: -kv[1])
    top, rest = ranked[:8], sum(v for _, v in ranked[8:])
    if rest:
        top.append(("Other", rest))
    rows = (len(top) + 2) // 3
    width, height = 1200, 150 + rows * 34
    body = _header("MOST USED LANGUAGES", t, width, f"<!--updated-->BY CODE SIZE · UPDATED {stamp}<!--/updated-->")
    x, bar_w = 24.0, width - 48
    body += f'<clipPath id="bar"><rect x="24" y="80" width="{bar_w}" height="14" rx="7"/></clipPath><g clip-path="url(#bar)">'
    colors = []
    for i, (name, size) in enumerate(top):
        color = LANG_COLORS.get(name, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]) if name != "Other" else t["faint"]
        colors.append(color)
        w = bar_w * size / total
        body += f'<rect x="{x:.2f}" y="80" width="{w + 0.5:.2f}" height="14" fill="{color}"/>'
        x += w
    body += "</g>"
    for i, ((name, size), color) in enumerate(zip(top, colors)):
        col, row = i % 3, i // 3
        lx, ly = 32 + col * 384, 132 + row * 34
        body += (f'<circle cx="{lx + 6}" cy="{ly - 5}" r="6" fill="{color}"/>'
                 f'<text x="{lx + 22}" y="{ly}" fill="{t["text"]}" font-size="15" font-weight="700">{escape(name)}</text>'
                 f'<text x="{lx + 350}" y="{ly}" text-anchor="end" fill="{t["muted"]}" font-size="15">{100 * size / total:.1f}%</text>')
    summary = ", ".join(f"{name} {100 * size / total:.1f}%" for name, size in top)
    return _svg(width, height, f"Most used languages across {USER}'s public repositories: {summary}", body, t)


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="only render the radar charts")
    args = parser.parse_args()

    for theme in THEMES:
        _write(f"edge-radar-{theme}.svg", radar_svg(theme))
    if args.offline:
        return 0
    try:
        metrics, languages = fetch_github()
    except (urllib.error.URLError, KeyError, TypeError, ValueError) as exc:
        print(f"GitHub API unavailable ({exc}); existing stats cards left untouched", file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for theme in THEMES:
        _write(f"stats-{theme}.svg", stats_svg(theme, metrics, stamp))
        if languages:
            _write(f"languages-{theme}.svg", languages_svg(theme, languages, stamp))
    if not languages:
        print("no language data returned; languages cards left untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
