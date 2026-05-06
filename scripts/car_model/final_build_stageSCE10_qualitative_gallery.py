#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path


def _first_images(render_dir: Path, limit: int) -> list[Path]:
    if not render_dir.is_dir():
        return []
    return sorted([p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight HTML qualitative gallery from render directories.")
    parser.add_argument("--entry", action="append", nargs=2, metavar=("LABEL", "RENDER_DIR"))
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    lines = ["<!doctype html><meta charset='utf-8'><title>SCE10 Gallery</title>", "<h1>SCE10 Qualitative Gallery</h1>"]
    for label, render_dir in args.entry or []:
        lines.append(f"<h2>{html.escape(label)}</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>")
        for image in _first_images(Path(render_dir), int(args.limit)):
            lines.append(f"<figure><img src='{html.escape(str(image.resolve()))}' style='max-width:240px'><figcaption>{html.escape(image.name)}</figcaption></figure>")
        lines.append("</div>")
    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output_html": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

