"""Collect parking engineering baseline vs recovery metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_counts(path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def _metrics_row(label: str, model_path: Path, state_path: Path, geometry_path: Path, baseline_type: str) -> dict[str, Any]:
    render_metrics = _load_json(model_path / "results.json").get("ours_200", {})
    geometry = _load_json(geometry_path)
    triangles, vertices = _state_counts(state_path)
    return {
        "label": label,
        "baseline_type": baseline_type,
        "model_path": str(model_path),
        "triangles": triangles,
        "vertices": vertices,
        "render_ssim": float(render_metrics.get("SSIM", float("nan"))),
        "render_psnr": float(render_metrics.get("PSNR", float("nan"))),
        "render_lpips": float(render_metrics.get("LPIPS", float("nan"))),
        "geometry_views": int(geometry.get("num_views_evaluated", 0)),
        "geometry_depth_count": int(geometry.get("depth", {}).get("count", 0)),
        "geometry_depth_absrel": float(geometry.get("depth", {}).get("abs_rel", float("nan"))),
        "geometry_depth_mae": float(geometry.get("depth", {}).get("mae", float("nan"))),
        "geometry_normal_mean_ang": float(geometry.get("normal", {}).get("mean_ang_deg", float("nan"))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_model = Path(args.engineering_baseline_model)
    recovery_model = Path(args.recovery_model)
    rows = [
        _metrics_row(
            "engineering_baseline_200iter",
            baseline_model,
            baseline_model / "point_cloud/iteration_200/point_cloud_state_dict.pt",
            baseline_model / "geometry_eval_colmap/iter_200.json",
            "engineering_baseline_current_repo_no_meshprior_application",
        ),
        _metrics_row(
            "recovery_cleanup_200iter",
            recovery_model,
            recovery_model / "point_cloud/iteration_200/point_cloud_state_dict.pt",
            recovery_model / "geometry_eval_colmap/iter_200.json",
            "meshprior_recovery_variant_checkpoint_copy",
        ),
    ]
    base = rows[0]
    for row in rows:
        row["delta_render_psnr_vs_engineering"] = row["render_psnr"] - base["render_psnr"]
        row["delta_render_ssim_vs_engineering"] = row["render_ssim"] - base["render_ssim"]
        row["delta_render_lpips_vs_engineering"] = row["render_lpips"] - base["render_lpips"]
        row["delta_depth_absrel_vs_engineering"] = row["geometry_depth_absrel"] - base["geometry_depth_absrel"]
        row["delta_normal_mean_ang_vs_engineering"] = row["geometry_normal_mean_ang"] - base["geometry_normal_mean_ang"]

    report = {
        "rows": rows,
        "paper_baseline_status": "MISSING",
        "paper_baseline_required": {
            "method": "original_or_clean_Mesh_Splatting",
            "dataset": "parking_phone_tiny_anonymized/dataset_view",
            "budget": "same iterations and evaluation protocol as MeshPrior variants",
            "reason": "engineering baseline is current modified repository and short-run only",
        },
        "decision": "SOFT_PASS_STABILITY_ONLY",
        "notes": [
            "The recovery cleanup variant is render/geometry stable against the current engineering baseline.",
            "No paper-level improvement claim is supported until the original/clean Mesh Splatting baseline exists.",
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parking_comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "parking_comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "parking_comparison_summary.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Comparison Summary\n\n")
        f.write(f"- decision: `{report['decision']}`\n")
        f.write(f"- paper baseline status: `{report['paper_baseline_status']}`\n")
        f.write("| label | triangles | PSNR | SSIM | LPIPS | AbsRel | normal mean angle |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                "| {label} | {triangles} | {psnr:.7f} | {ssim:.7f} | {lpips:.7f} | {absrel:.7f} | {ang:.7f} |\n".format(
                    label=row["label"],
                    triangles=row["triangles"],
                    psnr=row["render_psnr"],
                    ssim=row["render_ssim"],
                    lpips=row["render_lpips"],
                    absrel=row["geometry_depth_absrel"],
                    ang=row["geometry_normal_mean_ang"],
                )
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect parking engineering baseline and recovery metrics.")
    parser.add_argument("--engineering_baseline_model", default="outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model")
    parser.add_argument("--recovery_model", default="outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/comparison_summary")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"decision": report["decision"], "paper_baseline_status": report["paper_baseline_status"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
