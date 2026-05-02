"""Collect Stage23.5 integrated topology-control smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_counts(model: Path, iteration: int) -> tuple[int | None, int | None]:
    path = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not path.is_file():
        return None, None
    state = torch.load(path, map_location="cpu")
    return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])


def _round_rows(model: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((model / "prism_round_checkpoints").glob("*_meta.json")):
        payload = _load_json(path)
        rows.append(
            {
                "path": str(path),
                "iteration": payload.get("iteration"),
                "phase": payload.get("phase"),
                "prune_mode": payload.get("prune_mode"),
                "committed": payload.get("committed"),
                "counterfactual_accept": payload.get("counterfactual_accept"),
                "rollback": payload.get("rollback"),
                "no_candidates": payload.get("no_candidates", 0),
                "pre_prune_triangle_count": payload.get("pre_prune_triangle_count"),
                "post_prune_triangle_count": payload.get("post_prune_triangle_count"),
                "recollect_iters_used": payload.get("recollect_iters_used"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    iteration = int(args.iteration)
    results_path = model / "results.json"
    geometry_path = model / "geometry_eval_colmap" / f"iter_{iteration}.json"
    cleanup_path = model / "prism_debug" / "final_cleanup_summary.json"
    triangles, vertices = _state_counts(model, iteration)
    rows = _round_rows(model)
    effective_rows = [row for row in rows if int(row.get("no_candidates") or 0) == 0]
    retry_rows = [row for row in rows if int(row.get("no_candidates") or 0) > 0]
    committed_rows = [row for row in effective_rows if bool(row.get("committed"))]
    rollback_rows = [row for row in effective_rows if int(row.get("rollback") or 0) > 0]
    required_artifacts_exist = results_path.is_file() and geometry_path.is_file() and cleanup_path.is_file()
    if effective_rows and required_artifacts_exist and committed_rows:
        gate = "PASS"
    elif effective_rows and required_artifacts_exist:
        gate = "SOFT PASS"
    else:
        gate = "FAIL"

    metrics = _load_json(results_path).get(f"ours_{iteration}", {}) if results_path.is_file() else {}
    geometry = _load_json(geometry_path) if geometry_path.is_file() else {}
    cleanup = _load_json(cleanup_path) if cleanup_path.is_file() else {}
    report = {
        "model": str(model),
        "iteration": iteration,
        "gate": gate,
        "round_count": len(effective_rows),
        "retry_event_count": len(retry_rows),
        "total_event_count": len(rows),
        "committed_round_count": len(committed_rows),
        "rollback_round_count": len(rollback_rows),
        "triangles": triangles,
        "vertices": vertices,
        "render": {
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "LPIPS": metrics.get("LPIPS"),
        },
        "geometry": {
            "depth_absrel": (geometry.get("depth") or {}).get("abs_rel"),
            "depth_mae": (geometry.get("depth") or {}).get("mae"),
            "normal_mean_ang_deg": (geometry.get("normal") or {}).get("mean_ang_deg"),
        },
        "final_cleanup": cleanup,
        "rounds": rows,
        "notes": [
            "This is a short integrated topology-control smoke, not a paper-budget run.",
            "PASS means at least one training-time PRISM topology edit committed and all eval artifacts exist.",
            "SOFT PASS means PRISM scheduling/gate metadata and eval artifacts exist, but the short smoke did not commit a topology edit.",
            "no_candidates retry events are excluded from round_count because they do not consume a PRISM candidate round.",
        ],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage23_5_integrated_topology_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["path"]
    with (out / "stage23_5_prism_rounds.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "stage23_5_integrated_topology_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Stage23.5 Integrated Topology-Control Smoke Summary\n\n")
        f.write(f"Gate: `{report['gate']}`\n\n")
        f.write(f"- iteration: `{iteration}`\n")
        f.write(f"- effective rounds: `{len(effective_rows)}`\n")
        f.write(f"- retry events: `{len(retry_rows)}`\n")
        f.write(f"- committed rounds: `{len(committed_rows)}`\n")
        f.write(f"- rollback rounds: `{len(rollback_rows)}`\n")
        f.write(f"- triangles: `{triangles}`\n")
        f.write(f"- vertices: `{vertices}`\n")
        f.write(f"- PSNR / SSIM / LPIPS: `{report['render']['PSNR']}` / `{report['render']['SSIM']}` / `{report['render']['LPIPS']}`\n")
        f.write(f"- depth AbsRel: `{report['geometry']['depth_absrel']}`\n")
        f.write(f"- final cleanup enabled: `{cleanup.get('final_cleanup_enabled')}`\n")
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "rounds": len(effective_rows),
                "retry_events": len(retry_rows),
                "committed": len(committed_rows),
            },
            indent=2,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Stage23.5 integrated topology-control smoke outputs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, default=800)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
