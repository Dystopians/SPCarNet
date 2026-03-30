#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _latest_test_method_dir(model_path: Path) -> Path:
    test_dir = model_path / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test dir: {test_dir}")
    best = None
    best_it = -1
    for d in test_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        # ours_15000
        try:
            it = int(name.split("_")[-1])
        except Exception:
            it = -1
        if it >= best_it:
            best_it = it
            best = d
    if best is None:
        raise RuntimeError(f"No method directory under: {test_dir}")
    return best


def _load_image(path: Path, target_hw: Tuple[int, int] = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if target_hw is not None:
        im = im.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return im


def _draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, 24), fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return out


def _stack_h(images: List[Image.Image]) -> Image.Image:
    h = max(i.height for i in images)
    w = sum(i.width for i in images)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _sorted_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side qualitative panels across runs.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec name=/abs/or/relative/model_path; can repeat",
    )
    parser.add_argument("--max_views", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    panel_dir = out_dir / "qualitative_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    run_items: List[Tuple[str, Path, Path, Path]] = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Bad --run spec: {spec}")
        run_name, model_raw = spec.split("=", 1)
        model_path = Path(model_raw).resolve()
        method_dir = _latest_test_method_dir(model_path)
        renders_dir = method_dir / "renders"
        gt_dir = method_dir / "gt"
        if (not renders_dir.exists()) or (not gt_dir.exists()):
            raise FileNotFoundError(f"Missing renders/gt in {method_dir}")
        run_items.append((run_name, model_path, renders_dir, gt_dir))

    # Use first run's frame names as reference.
    ref_names = [p.name for p in _sorted_pngs(run_items[0][2])]
    if len(ref_names) == 0:
        raise RuntimeError("No render pngs found.")
    max_views = min(int(args.max_views), len(ref_names))
    chosen = ref_names[:max_views]

    panels = []
    for fname in chosen:
        gt_ref = _load_image(run_items[0][3] / fname)
        hw = (gt_ref.height, gt_ref.width)
        row = [_draw_label(gt_ref, "GT")]
        for run_name, _, renders_dir, _ in run_items:
            im = _load_image(renders_dir / fname, target_hw=hw)
            row.append(_draw_label(im, run_name))
        panel = _stack_h(row)
        out_path = panel_dir / fname
        panel.save(out_path)
        panels.append(str(out_path))

    index_md = out_dir / "qualitative_summary.md"
    lines = ["# Qualitative Comparison", ""]
    lines.append("Columns: `GT` + each run in given order.")
    lines.append("")
    for p in panels:
        rel = os.path.relpath(p, out_dir)
        lines.append(f"- `{rel}`")
    index_md.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "runs": [{"name": n, "model_path": str(m)} for n, m, _, _ in run_items],
        "max_views": int(max_views),
        "panels": panels,
    }
    (out_dir / "qualitative_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[Qual] Panels dir: {panel_dir}")
    print(f"[Qual] Markdown: {index_md}")
    print(f"[Qual] JSON: {out_dir / 'qualitative_summary.json'}")


if __name__ == "__main__":
    main()
