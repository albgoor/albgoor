"""Build the README "BUILT WITH" icon grid as one self-contained SVG per theme.

Icons come from skillicons.dev (MIT) at build time; Cursor is not offered there, so
its tile is drawn here in the same style from the official mark (simple-icons, CC0).
Shipping one local image keeps both rows scaling together on narrow screens and
removes the runtime dependency on skillicons.dev.

Usage: python scripts/generate_built_with.py
"""
from __future__ import annotations

import pathlib
import re
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

ROWS = [
    ["ts", "js", "react", "nextjs", "nodejs", "nestjs", "rust"],
    ["py", "postgres", "tailwind", "git", "github", "docker", "vscode"],
]
EXTRA_ROW2 = "cursor"

TILE, STEP = 256, 300
COLUMNS = 8
ICON_PX = 48

CURSOR_PATH = (
    "M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0"
    "l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0"
    "M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335"
    "a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23"
)
THEMES = {
    "dark": {"tile": "#242938", "cursor": "#EDECEC"},
    "light": {"tile": "#F4F2ED", "cursor": "#14120B"},
}
TITLE = ("TypeScript, JavaScript, React, Next.js, Node.js, NestJS, Rust, Python, PostgreSQL, "
         "Tailwind CSS, Git, GitHub, Docker, VS Code and Cursor")


def fetch_row(icons: list[str], theme: str) -> str:
    url = f"https://skillicons.dev/icons?i={','.join(icons)}&perline={len(icons)}&theme={theme}"
    req = urllib.request.Request(url, headers={"User-Agent": "albgoor-readme-assets"})
    svg = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    if "undefined" in svg or svg.count("<g transform=") != len(icons):
        raise SystemExit(f"skillicons returned an incomplete row for {icons} ({theme})")
    if re.search(r"<script|<image|href=\"http", svg):
        raise SystemExit(f"unexpected content in skillicons row {icons} ({theme})")
    start, end = svg.index(">", svg.index("<svg")) + 1, svg.rindex("</svg>")
    return svg[start:end].strip()


def cursor_tile(theme: str) -> str:
    c = THEMES[theme]
    return (f'<rect width="{TILE}" height="{TILE}" rx="60" fill="{c["tile"]}"/>'
            f'<path transform="translate(53 53) scale(6.25)" fill="{c["cursor"]}" d="{CURSOR_PATH}"/>')


def build(theme: str) -> str:
    width = COLUMNS * STEP - (STEP - TILE)
    height = len(ROWS) * STEP - (STEP - TILE)
    row1_offset = (width - (len(ROWS[0]) * STEP - (STEP - TILE))) // 2
    scale = ICON_PX / TILE
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width * scale:g}" height="{height * scale:g}" viewBox="0 0 {width} {height}" '
        f'fill="none" role="img" aria-label="{TITLE}">\n'
        f"<title>{TITLE}</title>\n"
        f'<g transform="translate({row1_offset} 0)">{fetch_row(ROWS[0], theme)}</g>\n'
        f'<g transform="translate(0 {STEP})">{fetch_row(ROWS[1], theme)}</g>\n'
        f'<g transform="translate({len(ROWS[1]) * STEP} {STEP})">{cursor_tile(theme)}</g>\n'
        "</svg>\n"
    )


def main() -> None:
    for theme in THEMES:
        out = ASSETS / f"built-with-{theme}.svg"
        out.write_text(build(theme), encoding="utf-8", newline="\n")
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
