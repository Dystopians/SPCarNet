"""Collect MeshPrior M11 scene experiment metrics from pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    gate = _load_json(run_dir / "scene_gate" / "gate_report.json")
    status = _load_json(run_dir / "pipeline_status.json")
    metrics = {
        "run_name": run_dir.name,
        "mode": gate.get("mode"),
        "checkpoint": "dry_run_synthetic",
        "iteration": 0,
        "gpu_used": args.gpu_used,
        "wandb_url": None,
        "accepted_count": int(gate.get("accepted_count", 0)),
        "rejected_count": int(gate.get("rejected_count", 0)),
        "proposal_count": int(gate.get("proposal_count", 0)),
        "pipeline_status": status.get("status"),
        "colmap_sparse_absrel": None,
        "sparse_depth_mae": None,
        "sparse_normal_mean_angle": None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "mae": None,
        "controlled_fps": None,
        "free_space_violation_delta_max": 0.0,
        "triangle_count_delta_sum": 0.0,
        "boundary_edge_delta_sum": 0.0,
        "component_count_delta_max": 0.0,
        "floater_count_delta_max": 0.0,
        "results": gate.get("results", []),
    }
    for result in metrics["results"]:
        m = result.get("metrics", {})
        metrics["free_space_violation_delta_max"] = max(
            float(metrics["free_space_violation_delta_max"]),
            float(m.get("free_space_violation_delta", 0.0)),
        )
        metrics["triangle_count_delta_sum"] = float(metrics["triangle_count_delta_sum"]) + float(m.get("triangle_count_delta", 0.0))
        metrics["boundary_edge_delta_sum"] = float(metrics["boundary_edge_delta_sum"]) + float(m.get("boundary_edge_delta", 0.0))
        metrics["component_count_delta_max"] = max(float(metrics["component_count_delta_max"]), float(m.get("component_count_delta", 0.0)))
        metrics["floater_count_delta_max"] = max(float(metrics["floater_count_delta_max"]), float(m.get("floater_count_delta", 0.0)))

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior M11 Scene Experiment Summary\n\n")
        f.write(f"run: `{run_dir.name}`\n\n")
        f.write(f"mode: `{metrics['mode']}`\n\n")
        f.write(f"pipeline_status: `{metrics['pipeline_status']}`\n\n")
        f.write(f"accepted/rejected: `{metrics['accepted_count']}` / `{metrics['rejected_count']}`\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        for key in [
            "proposal_count",
            "accepted_count",
            "rejected_count",
            "triangle_count_delta_sum",
            "boundary_edge_delta_sum",
            "component_count_delta_max",
            "floater_count_delta_max",
            "free_space_violation_delta_max",
        ]:
            f.write(f"| `{key}` | `{metrics[key]}` |\n")
        f.write("\nWandb was not started for this dry-run experiment.\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect M11 scene experiment metrics.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--gpu_used", default="none")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
