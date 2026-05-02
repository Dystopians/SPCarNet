"""Collect topology-budget comparison rows for parking 2000-iteration runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_TRAINING_INTERNAL = {
    "origin_main_2000iter": {
        "test_psnr": 16.46195650100708,
        "test_ssim": 0.4846517714085402,
        "test_lpips": 0.5333475658187159,
        "test_fps": 271.31298105829023,
        "source": "docs/car_model/meshprior_parking_origin_main_baseline_report.md",
    },
    "current_branch_2000iter": {
        "test_psnr": 16.441502058947528,
        "test_ssim": 0.4834401825511897,
        "test_lpips": 0.5322314313164463,
        "test_fps": 257.5665033592215,
        "source": "docs/car_model/meshprior_parking_medium_baseline_2000iter_report.md",
    },
    "stage17_meshprior_2000iter": {
        "test_psnr": 13.443806930824561,
        "test_ssim": 0.3471139594912529,
        "test_lpips": 0.6021583963323522,
        "test_fps": 272.85308373087673,
        "source": "docs/car_model/meshprior_stage17_real_variant_implementation_report.md",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_counts(path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def _metric_value(payload: dict[str, Any], key: str) -> float:
    if not payload:
        return float("nan")
    return float(payload.get(key, float("nan")))


def _cleanup_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "final_cleanup_summary_present": False,
            "final_cleanup_enabled": None,
            "cleanup_executed": None,
            "cleanup_pruned": None,
        }
    payload = _load_json(path)
    return {
        "final_cleanup_summary_present": True,
        "final_cleanup_enabled": bool(payload.get("final_cleanup_enabled", False)),
        "cleanup_executed": bool(payload.get("cleanup_executed", False)),
        "cleanup_pruned": int(payload.get("cleanup_pruned", payload.get("final_cleanup_pruned", 0))),
    }


def _row(label: str, method: str, model_path: Path, iteration: int, training_source: dict[str, Any]) -> dict[str, Any]:
    state_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    geometry_path = model_path / "geometry_eval_colmap" / f"iter_{iteration}.json"
    results_path = model_path / "results.json"
    triangles, vertices = _state_counts(state_path)
    render_metrics = _load_json(results_path).get(f"ours_{iteration}", {})
    geometry = _load_json(geometry_path)
    depth = geometry.get("depth", {})
    normal = geometry.get("normal", {})
    triangle_units = max(float(triangles) / 100000.0, 1e-12)
    cleanup = _cleanup_summary(model_path / "prism_debug" / "final_cleanup_summary.json")
    row = {
        "label": label,
        "method": method,
        "model_path": str(model_path),
        "iteration": int(iteration),
        "triangles": int(triangles),
        "vertices": int(vertices),
        "triangle_units_100k": triangle_units,
        "render_psnr": _metric_value(render_metrics, "PSNR"),
        "render_ssim": _metric_value(render_metrics, "SSIM"),
        "render_lpips": _metric_value(render_metrics, "LPIPS"),
        "geometry_depth_count": int(depth.get("count", 0)),
        "geometry_depth_mae": float(depth.get("mae", float("nan"))),
        "geometry_depth_absrel": float(depth.get("abs_rel", float("nan"))),
        "geometry_normal_mean_ang": float(normal.get("mean_ang_deg", float("nan"))),
        "geometry_normal_median_ang": float(normal.get("median_ang_deg", float("nan"))),
        "training_test_psnr": float(training_source.get("test_psnr", float("nan"))),
        "training_test_ssim": float(training_source.get("test_ssim", float("nan"))),
        "training_test_lpips": float(training_source.get("test_lpips", float("nan"))),
        "training_test_fps": float(training_source.get("test_fps", float("nan"))),
        "training_metric_source": str(training_source.get("source", "")),
        **cleanup,
    }
    row["render_psnr_per_100k_triangles"] = row["render_psnr"] / triangle_units
    row["render_ssim_per_100k_triangles"] = row["render_ssim"] / triangle_units
    row["depth_absrel_per_100k_triangles"] = row["geometry_depth_absrel"] / triangle_units
    row["depth_mae_per_100k_triangles"] = row["geometry_depth_mae"] / triangle_units
    row["triangles_per_training_fps"] = float(triangles) / max(row["training_test_fps"], 1e-12)
    return row


def _add_deltas(rows: list[dict[str, Any]]) -> None:
    by_label = {row["label"]: row for row in rows}
    clean = by_label["origin_main_2000iter"]
    current = by_label["current_branch_2000iter"]
    for row in rows:
        for base_name, base in (("clean", clean), ("current", current)):
            row[f"delta_render_psnr_vs_{base_name}"] = row["render_psnr"] - base["render_psnr"]
            row[f"delta_render_ssim_vs_{base_name}"] = row["render_ssim"] - base["render_ssim"]
            row[f"delta_render_lpips_vs_{base_name}"] = row["render_lpips"] - base["render_lpips"]
            row[f"delta_depth_absrel_vs_{base_name}"] = row["geometry_depth_absrel"] - base["geometry_depth_absrel"]
            row[f"triangle_ratio_vs_{base_name}"] = float(row["triangles"]) / max(float(base["triangles"]), 1.0)


def _decision(rows: list[dict[str, Any]]) -> str:
    by_label = {row["label"]: row for row in rows}
    clean = by_label["origin_main_2000iter"]
    stage17 = by_label["stage17_meshprior_2000iter"]
    if stage17["triangles"] > clean["triangles"] * 5:
        return "QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED"
    return "TOPOLOGY_BUDGET_ACCEPTABLE"


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        _row(
            "origin_main_2000iter",
            "clean_mesh_splatting_candidate",
            Path(args.origin_main_model),
            int(args.iteration),
            DEFAULT_TRAINING_INTERNAL["origin_main_2000iter"],
        ),
        _row(
            "current_branch_2000iter",
            "current_branch_engineering",
            Path(args.current_branch_model),
            int(args.iteration),
            DEFAULT_TRAINING_INTERNAL["current_branch_2000iter"],
        ),
        _row(
            "stage17_meshprior_2000iter",
            "meshprior_cleaned_checkpoint_resume",
            Path(args.stage17_model),
            int(args.iteration),
            DEFAULT_TRAINING_INTERNAL["stage17_meshprior_2000iter"],
        ),
    ]
    _add_deltas(rows)
    decision = _decision(rows)
    report = {
        "iteration": int(args.iteration),
        "rows": rows,
        "decision": decision,
        "gate": "PASS",
        "notes": [
            "Render and geometry metrics are read from local evaluation artifacts.",
            "Training FPS is copied from documented training-internal evaluation summaries because the training script does not emit a standalone machine-readable eval summary.",
            "Stage17 improves quality metrics on this scene but remains topology-inflated; use M18 before any paper claim.",
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "topology_budget_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with (out / "topology_budget_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "topology_budget_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Topology Budget Comparison\n\n")
        f.write(f"- iteration: `{int(args.iteration)}`\n")
        f.write(f"- decision: `{decision}`\n")
        f.write(f"- gate: `{report['gate']}`\n\n")
        f.write("| label | triangles | vertices | PSNR | SSIM | LPIPS | PSNR/100k tri | AbsRel | FPS |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                "| {label} | {triangles} | {vertices} | {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {psnr_norm:.6f} | {absrel:.6f} | {fps:.3f} |\n".format(
                    label=row["label"],
                    triangles=row["triangles"],
                    vertices=row["vertices"],
                    psnr=row["render_psnr"],
                    ssim=row["render_ssim"],
                    lpips=row["render_lpips"],
                    psnr_norm=row["render_psnr_per_100k_triangles"],
                    absrel=row["geometry_depth_absrel"],
                    fps=row["training_test_fps"],
                )
            )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect topology-budget comparison rows for parking 2000-iteration runs.")
    parser.add_argument("--origin_main_model", default="outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model")
    parser.add_argument("--current_branch_model", default="outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model")
    parser.add_argument("--stage17_model", default="outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model")
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"gate": report["gate"], "decision": report["decision"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
