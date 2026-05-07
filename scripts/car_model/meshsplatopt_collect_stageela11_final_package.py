#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


PROMOTED_ROWS: dict[str, dict[str, str | int]] = {
    "bonsai": {
        "label": "SOR10 + ELA safe",
        "clean_model": "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
        "clean_iteration": 9000,
        "clean_method": "ours_9000",
        "method_model": "outputs/carnet/meshsplatopt/stageELA11_sparse_occluder_policy/bonsai/sor10_clean9000/compact_model",
        "method_iteration": 9000,
        "method_name": "ours_9000_sor10_ela_safe",
    },
    "courtyard": {
        "label": "SOR10 + ELA safe",
        "clean_model": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
        "clean_iteration": 9000,
        "clean_method": "ours_9000",
        "method_model": "outputs/carnet/meshsplatopt/stageELA11_sparse_occluder_policy/courtyard/sor10_clean9000/compact_model",
        "method_iteration": 9000,
        "method_name": "ours_9000_sor10_ela_safe",
    },
    "room": {
        "label": "QEM50 parent-rollback + ELA safe",
        "clean_model": "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000",
        "clean_iteration": 9000,
        "clean_method": "ours_9000",
        "method_model": "outputs/carnet/meshsplatopt/stageELA10_geometry_preserving_compact_recovery/room/qem50_sparse_parentrollback_9000to12000/recovery_model",
        "method_iteration": 12000,
        "method_name": "ours_12000_qem50_parentrollback_ela_safe",
    },
    "counter": {
        "label": "QEM50 parent-rollback + ELA safe",
        "clean_model": "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000",
        "clean_iteration": 9000,
        "clean_method": "ours_9000",
        "method_model": "outputs/carnet/meshsplatopt/stageELA10_geometry_preserving_compact_recovery/counter/qem50_sparse_parentrollback_9000to12000/recovery_model",
        "method_iteration": 12000,
        "method_name": "ours_12000_qem50_parentrollback_ela_safe",
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rel(path: str | Path, base: Path = ROOT) -> str:
    return os.path.relpath(Path(path), start=base)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _metric_row(results_path: Path, method: str) -> dict[str, float]:
    row = _read_json(results_path).get(method, {})
    return {
        "psnr": _finite(row.get("PSNR")),
        "ssim": _finite(row.get("SSIM")),
        "lpips": _finite(row.get("LPIPS")),
    }


def _geometry(path: Path) -> dict[str, float]:
    data = _read_json(path)
    return {
        "abs_rel": _finite(data.get("depth", {}).get("abs_rel")),
        "depth_mae": _finite(data.get("depth", {}).get("mae")),
        "normal": _finite(data.get("normal", {}).get("mean_ang_deg")),
    }


def _topology(model_path: Path, iteration: int) -> dict[str, int]:
    import torch

    ckpt = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    state = torch.load(ckpt, map_location="cpu")
    return {
        "triangles": int(state["_triangle_indices"].shape[0]),
        "vertices": int(state["triangles_points"].shape[0]),
    }


def _image_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.name for item in path.glob("*.png")}


def _per_view_metric(per_view: dict[str, Any], method: str, metric: str, image_name: str) -> float:
    return _finite(per_view.get(method, {}).get(metric, {}).get(image_name))


def _average_row(scene: str, spec: dict[str, str | int]) -> dict[str, Any]:
    clean_model = ROOT / str(spec["clean_model"])
    method_model = ROOT / str(spec["method_model"])
    clean_iter = int(spec["clean_iteration"])
    method_iter = int(spec["method_iteration"])
    clean_method = str(spec["clean_method"])
    method_name = str(spec["method_name"])
    clean_rgb = _metric_row(clean_model / "results.json", clean_method)
    method_rgb = _metric_row(method_model / "results.json", method_name)
    clean_geom = _geometry(clean_model / "geometry_eval_colmap" / f"iter_{clean_iter}_max500.json")
    method_geom = _geometry(method_model / "geometry_eval_colmap" / f"iter_{method_iter}_max500.json")
    clean_topo = _topology(clean_model, clean_iter)
    method_topo = _topology(method_model, method_iter)
    row: dict[str, Any] = {
        "scene": scene,
        "method_label": str(spec["label"]),
        "clean_model": _rel(clean_model),
        "method_model": _rel(method_model),
        "clean_method": clean_method,
        "method_name": method_name,
        "clean_iteration": clean_iter,
        "method_iteration": method_iter,
        "clean_psnr": clean_rgb["psnr"],
        "clean_ssim": clean_rgb["ssim"],
        "clean_lpips": clean_rgb["lpips"],
        "method_psnr": method_rgb["psnr"],
        "method_ssim": method_rgb["ssim"],
        "method_lpips": method_rgb["lpips"],
        "clean_abs_rel": clean_geom["abs_rel"],
        "clean_depth_mae": clean_geom["depth_mae"],
        "clean_normal": clean_geom["normal"],
        "method_abs_rel": method_geom["abs_rel"],
        "method_depth_mae": method_geom["depth_mae"],
        "method_normal": method_geom["normal"],
        "clean_triangles": clean_topo["triangles"],
        "method_triangles": method_topo["triangles"],
        "clean_vertices": clean_topo["vertices"],
        "method_vertices": method_topo["vertices"],
    }
    row.update(
        {
            "d_psnr": row["method_psnr"] - row["clean_psnr"],
            "d_ssim": row["method_ssim"] - row["clean_ssim"],
            "d_lpips": row["method_lpips"] - row["clean_lpips"],
            "d_abs_rel": row["method_abs_rel"] - row["clean_abs_rel"],
            "d_depth_mae": row["method_depth_mae"] - row["clean_depth_mae"],
            "d_normal": row["method_normal"] - row["clean_normal"],
            "triangle_reduction": 1.0 - row["method_triangles"] / row["clean_triangles"],
            "vertex_reduction": 1.0 - row["method_vertices"] / row["clean_vertices"],
        }
    )
    row["strict_full_pass"] = (
        row["d_psnr"] > 0.0
        and row["d_ssim"] > 0.0
        and row["d_lpips"] < 0.0
        and row["d_abs_rel"] < 0.0
        and row["d_depth_mae"] < 0.0
        and row["d_normal"] < 0.0
        and row["triangle_reduction"] > 0.0
    )
    return row


def _collect_per_view(scene: str, spec: dict[str, str | int]) -> list[dict[str, Any]]:
    clean_model = ROOT / str(spec["clean_model"])
    method_model = ROOT / str(spec["method_model"])
    clean_method = str(spec["clean_method"])
    method_name = str(spec["method_name"])
    clean_dir = clean_model / "test" / clean_method
    method_dir = method_model / "test" / method_name
    clean_renders = clean_dir / "renders"
    method_renders = method_dir / "renders"
    clean_gt = clean_dir / "gt"
    method_gt = method_dir / "gt"
    clean_per = _read_json(clean_model / "per_view.json")
    method_per = _read_json(method_model / "per_view.json")
    rows: list[dict[str, Any]] = []
    for image_name in sorted(_image_names(clean_renders) & _image_names(method_renders)):
        clean_psnr = _per_view_metric(clean_per, clean_method, "PSNR", image_name)
        method_psnr = _per_view_metric(method_per, method_name, "PSNR", image_name)
        clean_ssim = _per_view_metric(clean_per, clean_method, "SSIM", image_name)
        method_ssim = _per_view_metric(method_per, method_name, "SSIM", image_name)
        clean_lpips = _per_view_metric(clean_per, clean_method, "LPIPS", image_name)
        method_lpips = _per_view_metric(method_per, method_name, "LPIPS", image_name)
        gt_path = method_gt / image_name if (method_gt / image_name).is_file() else clean_gt / image_name
        row = {
            "scene": scene,
            "image": image_name,
            "clean_psnr": clean_psnr,
            "method_psnr": method_psnr,
            "d_psnr": method_psnr - clean_psnr,
            "clean_ssim": clean_ssim,
            "method_ssim": method_ssim,
            "d_ssim": method_ssim - clean_ssim,
            "clean_lpips": clean_lpips,
            "method_lpips": method_lpips,
            "d_lpips": method_lpips - clean_lpips,
            "rgb_full_pass": (method_psnr > clean_psnr and method_ssim > clean_ssim and method_lpips < clean_lpips),
            "gt": _rel(gt_path),
            "clean_render": _rel(clean_renders / image_name),
            "method_render": _rel(method_renders / image_name),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, precision: int = 6) -> str:
    if isinstance(value, bool):
        return "`True`" if value else "`False`"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{precision}f}"


def _pick_gallery_rows(per_view_rows: list[dict[str, Any]], per_scene: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for scene in PROMOTED_ROWS:
        scene_rows = [row for row in per_view_rows if row["scene"] == scene and math.isfinite(float(row["d_psnr"]))]
        if not scene_rows:
            continue
        ordered = sorted(scene_rows, key=lambda row: float(row["d_psnr"]))
        if per_scene == 1:
            indexes = [len(ordered) // 2]
        else:
            indexes = [round(i * (len(ordered) - 1) / (per_scene - 1)) for i in range(per_scene)]
        seen: set[str] = set()
        for index in indexes:
            row = ordered[index]
            if row["image"] in seen:
                continue
            selected.append(row)
            seen.add(row["image"])
        for row in ordered:
            if len([x for x in selected if x["scene"] == scene]) >= per_scene:
                break
            if row["image"] in seen:
                continue
            selected.append(row)
            seen.add(row["image"])
    return selected


def _write_gallery(out_dir: Path, selected_rows: list[dict[str, Any]]) -> None:
    gallery_dir = out_dir / "qualitative_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    _write_json(gallery_dir / "selected_views.json", selected_rows)
    lines = [
        "# Stage ELA11 Final Qualitative Gallery Manifest",
        "",
        "Each held-out view aligns GT, clean Mesh Splatting, and the fixed adaptive-policy method.",
        "Views are selected mechanically per scene from method-minus-clean per-view PSNR: worst, middle, and best cases.",
        "",
        "| scene | view | clean PSNR | method PSNR | dPSNR | clean LPIPS | method LPIPS | dLPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['scene']} | {row['image']} | {_fmt(row['clean_psnr'], 4)} | {_fmt(row['method_psnr'], 4)} | "
            f"{float(row['d_psnr']):+.4f} | {_fmt(row['clean_lpips'], 4)} | {_fmt(row['method_lpips'], 4)} | {float(row['d_lpips']):+.4f} |"
        )
    (gallery_dir / "gallery_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sections = []
    for row in selected_rows:
        gt = _rel(ROOT / row["gt"], gallery_dir)
        clean = _rel(ROOT / row["clean_render"], gallery_dir)
        method = _rel(ROOT / row["method_render"], gallery_dir)
        sections.append(
            "<section>"
            f"<h2>{row['scene']} / {row['image']} / dPSNR {float(row['d_psnr']):+.4f} / dLPIPS {float(row['d_lpips']):+.4f}</h2>"
            "<div class='grid'>"
            f"<figure><img src='{gt}'><figcaption>GT</figcaption></figure>"
            f"<figure><img src='{clean}'><figcaption>Clean Mesh Splatting</figcaption></figure>"
            f"<figure><img src='{method}'><figcaption>SPCarNet adaptive policy</figcaption></figure>"
            "</div>"
            "</section>"
        )
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage ELA11 final qualitative gallery</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; background: #f6f6f4; color: #202020; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
p { max-width: 980px; color: #444; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 8px; }
img { width: 100%; height: auto; display: block; }
figcaption { font-size: 13px; margin-top: 6px; color: #555; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Stage ELA11 final qualitative gallery</h1>
<p>GT, clean Mesh Splatting, and the final adaptive-policy method are shown for identical held-out views. Selection is mechanical: worst, middle, and best method-minus-clean PSNR per scene.</p>
""" + "\n".join(sections) + "\n</body>\n</html>\n"
    (gallery_dir / "gallery.html").write_text(html, encoding="utf-8")


def _write_report(path: Path, average_rows: list[dict[str, Any]], per_view_rows: list[dict[str, Any]], out_dir: Path) -> None:
    view_summary = []
    for scene in PROMOTED_ROWS:
        rows = [row for row in per_view_rows if row["scene"] == scene]
        view_summary.append(
            {
                "scene": scene,
                "views": len(rows),
                "rgb_full_pass_views": sum(1 for row in rows if row["rgb_full_pass"]),
                "min_d_psnr": min(float(row["d_psnr"]) for row in rows),
                "mean_d_psnr": sum(float(row["d_psnr"]) for row in rows) / max(len(rows), 1),
                "max_d_lpips": max(float(row["d_lpips"]) for row in rows),
            }
        )
    lines = [
        "# Stage ELA11 Final Selected-Scene Package",
        "",
        "This package freezes the current best adaptive-policy row per selected scene and audits average metrics, sparse geometry, topology, per-view RGB deltas, and qualitative examples against the clean Mesh Splatting baseline.",
        "",
        "## Promoted Average Rows",
        "",
        "| scene | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | strict full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in average_rows:
        lines.append(
            f"| {row['scene']} | {row['method_label']} | {_fmt(row['d_psnr'])} | {_fmt(row['d_ssim'])} | {_fmt(row['d_lpips'])} | "
            f"{_fmt(row['d_abs_rel'])} | {_fmt(row['d_depth_mae'])} | {_fmt(row['d_normal'])} | {_fmt(100.0 * row['triangle_reduction'], 2)}% | {_fmt(row['strict_full_pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-View RGB Stress Test",
            "",
            "Per-view rows are not the headline claim, but they expose whether gains are broad or dominated by a few views.",
            "",
            "| scene | views | RGB full-pass views | min dPSNR | mean dPSNR | worst dLPIPS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in view_summary:
        lines.append(
            f"| {row['scene']} | {row['views']} | {row['rgb_full_pass_views']} | {_fmt(row['min_d_psnr'])} | {_fmt(row['mean_d_psnr'])} | {_fmt(row['max_d_lpips'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary JSON: `{_rel(out_dir / 'final_selected_scene_summary.json')}`",
            f"- average CSV: `{_rel(out_dir / 'promoted_average_rows.csv')}`",
            f"- per-view CSV: `{_rel(out_dir / 'per_view_rgb_deltas.csv')}`",
            f"- qualitative gallery: `{_rel(out_dir / 'qualitative_gallery' / 'gallery.html')}`",
            f"- qualitative manifest: `{_rel(out_dir / 'qualitative_gallery' / 'gallery_manifest.md')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect the final ELA11 selected-scene package.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package")
    parser.add_argument("--report", default="docs/car_model/stageELA11_final_selected_scene_package_report.md")
    parser.add_argument("--gallery-per-scene", default=3, type=int)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    average_rows = [_average_row(scene, spec) for scene, spec in PROMOTED_ROWS.items()]
    per_view_rows: list[dict[str, Any]] = []
    for scene, spec in PROMOTED_ROWS.items():
        per_view_rows.extend(_collect_per_view(scene, spec))
    selected_gallery_rows = _pick_gallery_rows(per_view_rows, args.gallery_per_scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "promoted_average_rows.csv", average_rows)
    _write_csv(out_dir / "per_view_rgb_deltas.csv", per_view_rows)
    _write_gallery(out_dir, selected_gallery_rows)
    payload = {
        "decision": "STRICT_MULTIAXIS_SELECTED_SCENES_FULL_PASS",
        "promoted_average_rows": average_rows,
        "per_view_rgb_deltas": per_view_rows,
        "selected_gallery_rows": selected_gallery_rows,
    }
    _write_json(out_dir / "final_selected_scene_summary.json", payload)
    _write_report(ROOT / args.report, average_rows, per_view_rows, out_dir)
    print(json.dumps({"decision": payload["decision"], "report": args.report, "out_dir": args.out_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
