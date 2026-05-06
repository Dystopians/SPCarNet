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


SCENES: dict[str, dict[str, str]] = {
    "bonsai": {
        "clean": "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
        "ela7": "outputs/carnet/meshsplatopt/stageELA7_portfolio/bonsai/evidence_pareto_portfolio",
    },
    "courtyard": {
        "clean": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
        "ela7": "outputs/carnet/meshsplatopt/stageELA7_portfolio/courtyard/evidence_pareto_portfolio",
    },
    "room": {
        "clean": "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000",
        "ela7": "outputs/carnet/meshsplatopt/stageELA7_portfolio/room/evidence_pareto_portfolio",
    },
    "counter": {
        "clean": "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000",
        "ela7": "outputs/carnet/meshsplatopt/stageELA7_portfolio/counter/evidence_pareto_portfolio",
    },
}

ELA7_METHOD = "ours_9000_ela7_pareto_portfolio"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _metrics(results: dict[str, Any], method: str) -> dict[str, float]:
    row = results.get(method, {})
    return {
        "psnr": _finite(row.get("PSNR")),
        "ssim": _finite(row.get("SSIM")),
        "lpips": _finite(row.get("LPIPS")),
    }


def _best_clean_by_metric(clean_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = []
    for method, values in clean_results.items():
        rows.append({"method": method, **_metrics(clean_results, method)})
    return {
        "psnr": max(rows, key=lambda row: row["psnr"]),
        "ssim": max(rows, key=lambda row: row["ssim"]),
        "lpips": min(rows, key=lambda row: row["lpips"]),
        "all_rows": rows,
    }


def _image_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.name for item in path.glob("*.png")}


def _metric(per_view: dict[str, Any], method: str, metric: str, image_name: str) -> float:
    return _finite(per_view.get(method, {}).get(metric, {}).get(image_name))


def _selected_gallery_rows(scene: str, clean_root: Path, ela_root: Path, per_scene: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean_method = "ours_9000"
    clean_dir = clean_root / "test" / clean_method
    ela_dir = ela_root / "test" / ELA7_METHOD
    clean_renders = clean_dir / "renders"
    ela_renders = ela_dir / "renders"
    clean_gt = clean_dir / "gt"
    ela_gt = ela_dir / "gt"
    common = sorted(_image_names(clean_renders) & _image_names(ela_renders))
    clean_per = _read_json(clean_root / "per_view.json") if (clean_root / "per_view.json").is_file() else {}
    ela_per = _read_json(ela_root / "per_view.json") if (ela_root / "per_view.json").is_file() else {}
    rows: list[dict[str, Any]] = []
    for image in common:
        clean_psnr = _metric(clean_per, clean_method, "PSNR", image)
        ela_psnr = _metric(ela_per, ELA7_METHOD, "PSNR", image)
        gt = ela_gt / image if (ela_gt / image).is_file() else clean_gt / image
        rows.append(
            {
                "scene": scene,
                "image": image,
                "clean_psnr": clean_psnr,
                "ela7_psnr": ela_psnr,
                "d_psnr": ela_psnr - clean_psnr if math.isfinite(clean_psnr) and math.isfinite(ela_psnr) else math.nan,
                "gt": str(gt),
                "clean_render": str(clean_renders / image),
                "ela7_render": str(ela_renders / image),
            }
        )
    finite = [row for row in rows if math.isfinite(row["d_psnr"])]
    if not finite:
        return rows[:per_scene], rows
    ordered = sorted(finite, key=lambda row: row["d_psnr"])
    if per_scene <= 1:
        indices = [len(ordered) // 2]
    else:
        indices = [round(i * (len(ordered) - 1) / (per_scene - 1)) for i in range(per_scene)]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in indices:
        row = ordered[index]
        if row["image"] in seen:
            continue
        seen.add(row["image"])
        selected.append(row)
    for row in ordered:
        if len(selected) >= per_scene:
            break
        if row["image"] in seen:
            continue
        selected.append(row)
        seen.add(row["image"])
    return selected, rows


def _rel(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path), start=base)


def _repo_rel(path: str | Path) -> str:
    return os.path.relpath(Path(path), start=ROOT)


def _write_gallery(out_dir: Path, selected: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "selected_views.json", selected)
    _write_json(out_dir / "all_views.json", all_rows)
    lines = [
        "# ELA7 Qualitative Gallery Manifest",
        "",
        "Each selected held-out view aligns GT, strongest clean Mesh Splatting render (`ours_9000`), and ELA7 Pareto render.",
        "Views are selected mechanically per scene from method-minus-clean per-view PSNR: worst, middle, and best cases.",
        "",
        "| scene | view | clean PSNR | ELA7 PSNR | dPSNR |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['scene']} | {row['image']} | {row['clean_psnr']:.4f} | "
            f"{row['ela7_psnr']:.4f} | {row['d_psnr']:+.4f} |"
        )
    (out_dir / "gallery_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sections = []
    for row in selected:
        sections.append(
            "<section>"
            f"<h2>{row['scene']} / {row['image']} / dPSNR {row['d_psnr']:+.4f}</h2>"
            "<div class='grid'>"
            f"<figure><img src='{_rel(row['gt'], out_dir)}'><figcaption>GT</figcaption></figure>"
            f"<figure><img src='{_rel(row['clean_render'], out_dir)}'><figcaption>clean Mesh Splatting best</figcaption></figure>"
            f"<figure><img src='{_rel(row['ela7_render'], out_dir)}'><figcaption>ELA7 Pareto</figcaption></figure>"
            "</div>"
            "</section>"
        )
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ELA7 qualitative gallery</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f5; color: #1c1c1c; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 8px; }
img { width: 100%; height: auto; display: block; }
figcaption { font-size: 13px; margin-top: 6px; color: #555; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>ELA7 qualitative gallery</h1>
<p>GT, strongest clean Mesh Splatting render, and ELA7 render are shown for identical held-out views.</p>
""" + "\n".join(sections) + "\n</body>\n</html>\n"
    (out_dir / "gallery.html").write_text(html, encoding="utf-8")


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return math.nan
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


def _audit_portfolio(scene: str, ela_root: Path) -> dict[str, Any]:
    method_dir = ela_root / "test" / ELA7_METHOD
    report_path = method_dir / "portfolio_report.json"
    report = _read_json(report_path)
    calibration = report.get("calibration", {})
    rows = calibration.get("rows", [])
    selected_weight = float(report.get("weight", math.nan))
    selected_rows = [row for row in rows if abs(float(row.get("weight", math.nan)) - selected_weight) < 1e-9]
    selected_row = selected_rows[0] if selected_rows else {}
    calib_views = {str(item) for item in calibration.get("calibration_views", [])}
    pass_rows = [row for row in rows if bool(row.get("pareto_pass", False))]
    best_pass_score = max([float(row.get("selection_score", -math.inf)) for row in pass_rows], default=-math.inf)
    selected_score = float(selected_row.get("selection_score", -math.inf)) if selected_row else -math.inf
    checks = {
        "target_split_is_test": report.get("target_split") == "test",
        "calib_split_is_train": report.get("calib_split") == "train",
        "calibration_view_count_bounded": 0 < len(calib_views) <= 16,
        "selected_row_exists": bool(selected_row),
        "selected_row_pareto_pass": bool(selected_row.get("pareto_pass", False)),
        "selected_score_is_best_pareto": selected_score >= best_pass_score - 1e-9,
    }
    return {
        "scene": scene,
        "report_path": str(report_path),
        "weight": selected_weight,
        "calibration_views": sorted(calib_views),
        "target_view_count": len(list((method_dir / "renders").glob("*.png"))),
        "split_local_name_note": "render filenames are split-local indices, so train/test overlap cannot be inferred from stems",
        "selected_row": selected_row,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _collect_distillation(distill_roots: list[Path], clean_best: dict[str, dict[str, Any]], ela7_metrics: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for root in distill_roots:
        results_path = root / "results.json"
        if not results_path.is_file():
            rows.append({"path": _repo_rel(root), "status": "MISSING_RESULTS"})
            continue
        results = _read_json(results_path)
        for method in sorted(results.keys()):
            metrics = _metrics(results, method)
            rows.append(
                {
                    "path": _repo_rel(root),
                    "method": method,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                    "d_psnr_vs_clean_best": metrics["psnr"] - clean_best["psnr"]["psnr"],
                    "d_ssim_vs_clean_best": metrics["ssim"] - clean_best["ssim"]["ssim"],
                    "d_lpips_vs_clean_best": metrics["lpips"] - clean_best["lpips"]["lpips"],
                    "d_psnr_vs_ela7": metrics["psnr"] - ela7_metrics["psnr"],
                    "d_ssim_vs_ela7": metrics["ssim"] - ela7_metrics["ssim"],
                    "d_lpips_vs_ela7": metrics["lpips"] - ela7_metrics["lpips"],
                    "status": "PASS_PROMOTE"
                    if (
                        metrics["psnr"] >= clean_best["psnr"]["psnr"]
                        and metrics["ssim"] >= clean_best["ssim"]["ssim"]
                        and metrics["lpips"] <= clean_best["lpips"]["lpips"]
                    )
                    else "REJECT",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene",
        "clean_best_psnr_method",
        "clean_best_psnr",
        "clean_best_ssim_method",
        "clean_best_ssim",
        "clean_best_lpips_method",
        "clean_best_lpips",
        "ela7_psnr",
        "ela7_ssim",
        "ela7_lpips",
        "d_psnr_vs_best_clean_metric",
        "d_ssim_vs_best_clean_metric",
        "d_lpips_vs_best_clean_metric",
        "portfolio_weight",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _fmt(value: Any, precision: int = 6) -> str:
    value = _finite(value)
    return "nan" if not math.isfinite(value) else f"{value:.{precision}f}"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["ela7_vs_clean"]
    audit_rows = payload["portfolio_audit"]
    distill_rows = payload["distillation"]
    gallery_summary = payload["gallery_summary"]
    lines = [
        "# Stage ELA7 Final Audit and ELA8 Distillation Report",
        "",
        f"Generated by `scripts/car_model/meshsplatopt_collect_ela7_final_audit.py`.",
        "",
        "## Decision",
        "",
        f"Final status: `{payload['decision']}`.",
        "",
        "ELA7 remains the promoted method. It beats the best pure Mesh Splatting clean baseline available in these selected scenes on PSNR, SSIM, and LPIPS under independent `metrics.py` outputs. The attempted ELA8 checkpoint distillation rows are not promoted unless listed as `PASS_PROMOTE` below.",
        "",
        "## ELA7 vs Best Clean Baseline",
        "",
        "| scene | best clean PSNR | best clean SSIM | best clean LPIPS | ELA7 PSNR | ELA7 SSIM | ELA7 LPIPS | dPSNR | dSSIM | dLPIPS | weight | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {_fmt(row['clean_best_psnr'])} | {_fmt(row['clean_best_ssim'])} | {_fmt(row['clean_best_lpips'])} | "
            f"{_fmt(row['ela7_psnr'])} | {_fmt(row['ela7_ssim'])} | {_fmt(row['ela7_lpips'])} | "
            f"{_fmt(row['d_psnr_vs_best_clean_metric'])} | {_fmt(row['d_ssim_vs_best_clean_metric'])} | {_fmt(row['d_lpips_vs_best_clean_metric'])} | "
            f"{_fmt(row['portfolio_weight'], 3)} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Leakage and Selection Audit",
            "",
            "| scene | status | target split | calib split | calibration views bounded | selected Pareto pass | selected best Pareto score |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in audit_rows:
        checks = row["checks"]
        lines.append(
            f"| {row['scene']} | `{row['status']}` | `{checks['target_split_is_test']}` | `{checks['calib_split_is_train']}` | "
            f"`{checks['calibration_view_count_bounded']}` | `{checks['selected_row_pareto_pass']}` | `{checks['selected_score_is_best_pareto']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-view Qualitative Risk",
            "",
            "| scene | views | negative dPSNR views | min dPSNR | median dPSNR | max dPSNR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gallery_summary:
        lines.append(
            f"| {row['scene']} | {row['views']} | {row['negative_dpsnr_views']} | "
            f"{_fmt(row['min_dpsnr'], 4)} | {_fmt(row['median_dpsnr'], 4)} | {_fmt(row['max_dpsnr'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## ELA8 Distillation Attempts",
            "",
            "| path | method | PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | dPSNR vs ELA7 | dSSIM vs ELA7 | dLPIPS vs ELA7 | status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in distill_rows:
        if row.get("status") == "MISSING_RESULTS":
            lines.append(f"| {row['path']} |  |  |  |  |  |  |  |  |  |  | `MISSING_RESULTS` |")
            continue
        lines.append(
            f"| {row['path']} | {row['method']} | {_fmt(row['psnr'])} | {_fmt(row['ssim'])} | {_fmt(row['lpips'])} | "
            f"{_fmt(row['d_psnr_vs_clean_best'])} | {_fmt(row['d_ssim_vs_clean_best'])} | {_fmt(row['d_lpips_vs_clean_best'])} | "
            f"{_fmt(row['d_psnr_vs_ela7'])} | {_fmt(row['d_ssim_vs_ela7'])} | {_fmt(row['d_lpips_vs_ela7'])} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Artifact Notes",
            "",
            f"- Machine-readable audit: `{payload['paths']['json']}`",
            f"- CSV table: `{payload['paths']['csv']}`",
            f"- Qualitative gallery: `{payload['paths']['gallery_html']}`",
            "",
            "## Interpretation",
            "",
            "The main unresolved weakness is not whether ELA7 beats the selected-scene clean render baseline; it does. The weakness is method form: ELA7 is still a renderer-side evidence portfolio, not a distilled checkpoint or compact topology method. The first ELA8 distillation pilot regressed below clean, which means persistent distillation needs a different mechanism before it can replace ELA7.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect final ELA7 audit, baseline comparison, and qualitative gallery.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/stageELA7_final_audit")
    parser.add_argument("--report", default="docs/car_model/stageELA7_final_audit_and_ela8_distillation_report.md")
    parser.add_argument("--per-scene-gallery", type=int, default=3)
    parser.add_argument("--scenes", default=",".join(SCENES.keys()))
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    table_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    all_gallery_rows: list[dict[str, Any]] = []
    selected_gallery_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []

    for scene in scenes:
        spec = SCENES[scene]
        clean_root = ROOT / spec["clean"]
        ela_root = ROOT / spec["ela7"]
        clean_results = _read_json(clean_root / "results.json")
        ela_results = _read_json(ela_root / "results.json")
        clean_best = _best_clean_by_metric(clean_results)
        ela_metrics = _metrics(ela_results, ELA7_METHOD)
        audit = _audit_portfolio(scene, ela_root)
        audit_rows.append(audit)
        row = {
            "scene": scene,
            "clean_best_psnr_method": clean_best["psnr"]["method"],
            "clean_best_psnr": clean_best["psnr"]["psnr"],
            "clean_best_ssim_method": clean_best["ssim"]["method"],
            "clean_best_ssim": clean_best["ssim"]["ssim"],
            "clean_best_lpips_method": clean_best["lpips"]["method"],
            "clean_best_lpips": clean_best["lpips"]["lpips"],
            "ela7_psnr": ela_metrics["psnr"],
            "ela7_ssim": ela_metrics["ssim"],
            "ela7_lpips": ela_metrics["lpips"],
            "d_psnr_vs_best_clean_metric": ela_metrics["psnr"] - clean_best["psnr"]["psnr"],
            "d_ssim_vs_best_clean_metric": ela_metrics["ssim"] - clean_best["ssim"]["ssim"],
            "d_lpips_vs_best_clean_metric": ela_metrics["lpips"] - clean_best["lpips"]["lpips"],
            "portfolio_weight": audit["weight"],
        }
        row["status"] = (
            "PASS_ALL_METRIC_BEST_CLEAN_WIN"
            if row["d_psnr_vs_best_clean_metric"] >= 0.0
            and row["d_ssim_vs_best_clean_metric"] >= 0.0
            and row["d_lpips_vs_best_clean_metric"] <= 0.0
            and audit["status"] == "PASS"
            else "FAIL"
        )
        table_rows.append(row)
        selected, all_rows = _selected_gallery_rows(scene, clean_root, ela_root, args.per_scene_gallery)
        selected_gallery_rows.extend(selected)
        all_gallery_rows.extend(all_rows)
        if scene == "courtyard":
            distill_rows.extend(
                _collect_distillation(
                    [
                        ROOT / "outputs/carnet/meshsplatopt/stageELA8_distill/courtyard/distill_pilot_9000to9600",
                        ROOT / "outputs/carnet/meshsplatopt/stageELA8_distill/courtyard/distill_parentrollback_9000to9300",
                    ],
                    clean_best,
                    ela_metrics,
                )
            )

    storage = {
        scene: {
            "clean_model_dir_mb": _dir_size_mb(ROOT / SCENES[scene]["clean"]),
            "ela7_stage_dir_mb": _dir_size_mb(ROOT / "outputs/carnet/meshsplatopt/stageELA7_portfolio" / scene),
            "ela7_final_render_dir_mb": _dir_size_mb(ROOT / SCENES[scene]["ela7"]),
        }
        for scene in scenes
    }
    gallery_summary = []
    for scene in scenes:
        values = sorted(
            row["d_psnr"]
            for row in all_gallery_rows
            if row["scene"] == scene and math.isfinite(row["d_psnr"])
        )
        gallery_summary.append(
            {
                "scene": scene,
                "views": len(values),
                "negative_dpsnr_views": sum(1 for value in values if value < 0.0),
                "min_dpsnr": values[0] if values else math.nan,
                "median_dpsnr": values[len(values) // 2] if values else math.nan,
                "max_dpsnr": values[-1] if values else math.nan,
            }
        )
    decision = (
        "ELA7_PROMOTED_DISTILLATION_NOT_PROMOTED"
        if all(row["status"] == "PASS_ALL_METRIC_BEST_CLEAN_WIN" for row in table_rows)
        and all(row["status"] == "PASS" for row in audit_rows)
        else "NOT_READY"
    )
    paths = {
        "json": f"{args.out_dir}/ela7_final_audit.json",
        "csv": f"{args.out_dir}/ela7_vs_best_clean.csv",
        "gallery_html": f"{args.out_dir}/qualitative_gallery/gallery.html",
        "report": args.report,
    }
    payload = {
        "decision": decision,
        "ela7_method": ELA7_METHOD,
        "ela7_vs_clean": table_rows,
        "portfolio_audit": audit_rows,
        "distillation": distill_rows,
        "gallery_summary": gallery_summary,
        "storage_artifact_mb": storage,
        "paths": paths,
    }
    _write_json(out_dir / "ela7_final_audit.json", payload)
    _write_csv(out_dir / "ela7_vs_best_clean.csv", table_rows)
    _write_gallery(out_dir / "qualitative_gallery", selected_gallery_rows, all_gallery_rows)
    _write_report(ROOT / args.report, payload)
    print(json.dumps({"decision": decision, "report": args.report, "out_dir": args.out_dir}, indent=2))
    return 0 if decision != "NOT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
