#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


SCENES = {
    "bonsai": {
        "clean": "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
        "method": "outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/bonsai/adaptive_global_policy_v5_seed0/recovery_model",
    },
    "courtyard": {
        "clean": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
        "method": "outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/courtyard/adaptive_global_policy_v5_seed0/recovery_model",
    },
    "room": {
        "clean": "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000",
        "method": "outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/room/adaptive_global_policy_v5_seed0/recovery_model",
    },
    "counter": {
        "clean": "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000",
        "method": "outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/counter/adaptive_global_policy_v5_seed0/recovery_model",
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(per_view: dict[str, Any], iteration: str, metric: str, image: str) -> float:
    value = per_view.get(iteration, {}).get(metric, {}).get(image, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _image_names(render_dir: Path) -> set[str]:
    if not render_dir.is_dir():
        return set()
    return {path.name for path in render_dir.glob("*.png")}


def _pick_views(rows: list[dict[str, Any]], per_scene: int) -> list[dict[str, Any]]:
    finite = [row for row in rows if math.isfinite(float(row["d_psnr"]))]
    if not finite:
        return rows[:per_scene]
    ordered = sorted(finite, key=lambda row: float(row["d_psnr"]))
    if per_scene <= 1:
        positions = [len(ordered) // 2]
    else:
        positions = [round(i * (len(ordered) - 1) / (per_scene - 1)) for i in range(per_scene)]
    picks = [ordered[pos] for pos in positions]
    out = []
    seen = set()
    for row in picks:
        image = str(row["image"])
        if image in seen:
            continue
        seen.add(image)
        out.append(row)
        if len(out) >= per_scene:
            break
    if len(out) < per_scene:
        for row in ordered:
            image = str(row["image"])
            if image in seen:
                continue
            seen.add(image)
            out.append(row)
            if len(out) >= per_scene:
                break
    return out


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base)


def _scene_rows(scene: str, clean_root: Path, method_root: Path) -> list[dict[str, Any]]:
    clean_iter = "ours_22000"
    method_iter = "ours_26000"
    clean_render = clean_root / "test" / clean_iter / "renders"
    clean_gt = clean_root / "test" / clean_iter / "gt"
    method_render = method_root / "test" / method_iter / "renders"
    method_gt = method_root / "test" / method_iter / "gt"
    common = sorted(_image_names(clean_render) & _image_names(method_render) & (_image_names(clean_gt) | _image_names(method_gt)))
    clean_per = _read_json(clean_root / "per_view.json") if (clean_root / "per_view.json").is_file() else {}
    method_per = _read_json(method_root / "per_view.json") if (method_root / "per_view.json").is_file() else {}
    rows = []
    for image in common:
        clean_psnr = _metric(clean_per, clean_iter, "PSNR", image)
        method_psnr = _metric(method_per, method_iter, "PSNR", image)
        rows.append(
            {
                "scene": scene,
                "image": image,
                "clean_psnr": clean_psnr,
                "method_psnr": method_psnr,
                "d_psnr": method_psnr - clean_psnr if math.isfinite(clean_psnr) and math.isfinite(method_psnr) else math.nan,
                "gt": str((method_gt / image) if (method_gt / image).is_file() else (clean_gt / image)),
                "clean_render": str(clean_render / image),
                "method_render": str(method_render / image),
            }
        )
    return rows


def _write_md(path: Path, selected: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# F82 Qualitative Gallery Manifest",
        "",
        "This manifest aligns the strongest clean-long baseline render with the F82 fixed-policy method render at identical held-out test views.",
        "Views are selected mechanically per scene from per-view PSNR deltas: worst, median, and best method-vs-clean views, plus evenly spaced extras when requested.",
        "",
        "## Scene Coverage",
        "",
        "| scene | common views | selected views | min dPSNR | median dPSNR | max dPSNR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene in sorted({row["scene"] for row in all_rows}):
        rows = [row for row in all_rows if row["scene"] == scene]
        picked = [row for row in selected if row["scene"] == scene]
        values = sorted(float(row["d_psnr"]) for row in rows if math.isfinite(float(row["d_psnr"])))
        if values:
            lines.append(f"| {scene} | {len(rows)} | {len(picked)} | {values[0]:+.4f} | {values[len(values)//2]:+.4f} | {values[-1]:+.4f} |")
        else:
            lines.append(f"| {scene} | {len(rows)} | {len(picked)} | nan | nan | nan |")
    lines.extend(
        [
            "",
            "## Selected Views",
            "",
            "| scene | view | clean PSNR | F82 PSNR | dPSNR |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['scene']} | {row['image']} | {float(row['clean_psnr']):.4f} | "
            f"{float(row['method_psnr']):.4f} | {float(row['d_psnr']):+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, selected: list[dict[str, Any]]) -> None:
    def img(src: str) -> str:
        return f'<img src="{_rel(Path(src), path.parent)}" loading="lazy">'

    rows = []
    for row in selected:
        rows.append(
            "<section>"
            f"<h2>{row['scene']} / {row['image']} / dPSNR {float(row['d_psnr']):+.4f}</h2>"
            "<div class='grid'>"
            f"<figure>{img(row['gt'])}<figcaption>GT</figcaption></figure>"
            f"<figure>{img(row['clean_render'])}<figcaption>clean 22k</figcaption></figure>"
            f"<figure>{img(row['method_render'])}<figcaption>F82 fixed policy</figcaption></figure>"
            "</div>"
            "</section>"
        )
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>F82 qualitative gallery</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; background: #f6f6f4; color: #1b1b1b; }
h1 { font-size: 24px; margin-bottom: 8px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }
figure { margin: 0; background: white; border: 1px solid #ddd; padding: 8px; }
img { width: 100%; height: auto; display: block; }
figcaption { font-size: 13px; margin-top: 6px; color: #555; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>F82 fixed-policy qualitative gallery</h1>
<p>Each row compares GT, strongest clean-long baseline, and F82 on the same held-out view.</p>
""" + "\n".join(rows) + "\n</body>\n</html>\n"
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fair clean-long vs F82 qualitative gallery.")
    parser.add_argument("--scenes", default=",".join(SCENES.keys()))
    parser.add_argument("--per-scene", type=int, default=5)
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF82_qualitative_gallery")
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for scene in [item.strip() for item in args.scenes.split(",") if item.strip()]:
        spec = SCENES[scene]
        rows = _scene_rows(scene, ROOT / spec["clean"], ROOT / spec["method"])
        all_rows.extend(rows)
        selected.extend(_pick_views(rows, args.per_scene))
    (out / "all_views.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    (out / "selected_views.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    _write_md(out / "gallery_manifest.md", selected, all_rows)
    _write_html(out / "gallery.html", selected)
    print(f"Wrote qualitative gallery to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
