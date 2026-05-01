"""Run or dry-run the MeshPrior experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for meshprior_run_experiment_matrix.py") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs/ss3dm_prior/meshprior/meshprior_experiment_matrix.yaml"


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _load_json(path: str | Path) -> dict | list:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _last_history(payload: list[dict]) -> dict:
    return payload[-1] if payload else {}


def _object_metrics(row: dict, payload: dict | list) -> dict:
    if isinstance(payload, list):
        src = _last_history(payload)
        return {
            "recon_chamfer_l1": src.get("val_recon_chamfer_l1"),
            "hidden_chamfer_l1": src.get("val_hidden_completion_chamfer_l1"),
            "visible_preservation_error": src.get("val_visible_recon_chamfer_l1"),
            "zero_corruption_chamfer": None,
            "free_space_violation": src.get("val_free_space_violation_rate"),
            "mesh_extraction_success": None,
            "inference_time": src.get("val_epoch_time"),
        }
    summary = payload.get("summary", payload)
    return {
        "recon_chamfer_l1": summary.get("recon_chamfer_l1_mean"),
        "hidden_chamfer_l1": summary.get("hidden_chamfer_l1_mean"),
        "visible_preservation_error": summary.get("visible_preservation_error_mean"),
        "zero_corruption_chamfer": summary.get("zero_corruption_recon_chamfer_l1_mean"),
        "free_space_violation": summary.get("free_space_violation_rate_mean"),
        "mesh_extraction_success": summary.get("mesh_extraction_success_rate"),
        "inference_time": summary.get("inference_time"),
    }


def _snap_rows(payload: dict) -> list[dict]:
    rows = []
    for item in payload.get("rows", []):
        rows.append(
            {
                "method": item.get("profile"),
                "damage_type": payload.get("damage_type", "vertex_noise"),
                "hole_closure": None,
                "floater_prune_precision": None,
                "floater_prune_recall": None,
                "valid_surface_protect_recall": item.get("snapped_valid_surface_protect_recall"),
                "visible_preservation": item.get("valid_surface_protect_recall_delta"),
                "free_space_violation": item.get("free_space_violation_delta"),
                "triangle_count_delta": None,
                "snap_surface_distance_delta": item.get("surface_distance_delta_mean"),
                "accepted_by_profile": item.get("accepted_by_profile"),
            }
        )
    return rows


def _scene_metrics(row: dict, payload: dict) -> dict:
    metrics = {
        "scene": row.get("scene"),
        "checkpoint_iteration": row.get("checkpoint"),
        "psnr": payload.get("test/psnr") or payload.get("psnr"),
        "ssim": payload.get("test/ssim") or payload.get("ssim"),
        "lpips": payload.get("test/lpips") or payload.get("lpips"),
        "mae": payload.get("test/l1") or payload.get("mae"),
        "colmap_absrel": payload.get("colmap_sparse_absrel"),
        "sparse_depth_mae": payload.get("sparse_depth_mae"),
        "normal_mean_angle": payload.get("sparse_normal_mean_angle"),
        "triangle_count": payload.get("mesh/triangle_count"),
        "controlled_fps": payload.get("test/fps") or payload.get("controlled_fps"),
        "car_roi_hole_floater_metrics": None,
        "accepted_proposals": payload.get("accepted_count"),
        "rejected_proposals": payload.get("rejected_count"),
    }
    return metrics


def _merge_optional_scene_files(row: dict, metrics: dict) -> dict:
    geometry_path = row.get("geometry_path")
    if geometry_path and Path(geometry_path).is_file():
        geom = _load_json(geometry_path)
        metrics["colmap_absrel"] = (geom.get("depth") or {}).get("abs_rel")
        metrics["sparse_depth_mae"] = (geom.get("depth") or {}).get("mae")
        metrics["normal_mean_angle"] = (geom.get("normal") or {}).get("mean_ang_deg")
    cleanup_path = row.get("cleanup_path")
    if cleanup_path and Path(cleanup_path).is_file():
        cleanup = _load_json(cleanup_path)
        metrics["triangle_count"] = cleanup.get("post_prune_triangle_count", metrics.get("triangle_count"))
        metrics["cleanup_enabled"] = cleanup.get("final_cleanup_enabled")
        metrics["cleanup_pruned"] = cleanup.get("final_cleanup_pruned")
    return metrics


def _evaluate(row: dict) -> dict:
    path = Path(row.get("metrics_path", ""))
    result = {
        "id": row["id"],
        "group": row["group"],
        "method": row["method"],
        "output_type": row.get("output_type", ""),
        "status": "MISSING",
        "metrics_path": str(path),
        "oracle_only": bool(row.get("oracle_only", False)),
        "dry_run": bool(row.get("dry_run", False)),
        "metrics": {},
        "synthetic_rows": [],
        "missing_reason": "",
    }
    if not path.is_file():
        result["missing_reason"] = f"metrics_path not found: {path}"
        return result
    payload = _load_json(path)
    kind = row.get("metric_kind", "")
    result["status"] = "AVAILABLE"
    if row["group"] == "object":
        result["metrics"] = _object_metrics(row, payload)
    elif kind == "snap_calibration":
        result["synthetic_rows"] = _snap_rows(payload)
        result["metrics"] = {
            "calibrated_improves_recall_vs_uncalibrated": payload.get("calibrated_improves_recall_vs_uncalibrated"),
            "calibrated_keeps_baseline_recall": payload.get("calibrated_keeps_baseline_recall"),
            "free_space_safe": payload.get("free_space_safe"),
        }
    elif row["group"] == "scene":
        result["metrics"] = _merge_optional_scene_files(row, _scene_metrics(row, payload))
    elif row["group"] == "synthetic":
        result["metrics"] = payload if isinstance(payload, dict) else {"rows": payload}
    return _clean(result)


def run(args: argparse.Namespace) -> dict:
    registry_path = Path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    experiments = registry.get("experiments", [])
    wanted_groups = {args.group} if args.group != "all" else {"object", "synthetic", "scene"}
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for exp in experiments:
        if args.only and exp["id"] != args.only:
            continue
        if exp["group"] not in wanted_groups:
            continue
        rows.append(_evaluate(exp))
        if args.smoke and len(rows) >= max(1, int(args.max_objects)):
            break
    out_dir = Path(args.output_dir or registry.get("default_output_dir", "outputs/carnet/meshprior/experiment_matrix"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry": str(registry_path),
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
        "group": args.group,
        "seeds": seeds,
        "no_train": bool(args.no_train),
        "eval_only": bool(args.eval_only),
        "counts": {
            "total": len(rows),
            "available": sum(1 for r in rows if r["status"] == "AVAILABLE"),
            "missing": sum(1 for r in rows if r["status"] == "MISSING"),
        },
        "experiments": rows,
    }
    out_path = out_dir / "matrix_results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matrix_results": str(out_path), **payload["counts"]}, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MeshPrior experiment matrix.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--group", default="all", choices=["object", "synthetic", "scene", "all"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max_objects", type=int, default=999)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
