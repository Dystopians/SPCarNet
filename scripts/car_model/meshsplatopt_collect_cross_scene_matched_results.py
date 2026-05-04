#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.evaluation_contracts import (
    MethodResult,
    MetricTargets,
    clean_json,
    compare_to_baseline,
    load_geometry_metrics,
    load_render_metrics,
)


def _row(
    row_id: str,
    scene: str,
    method: str,
    model_path: str,
    iteration: int,
    role: str,
    triangles: int,
    vertices: int,
    wandb_run: str,
    notes: str = "",
) -> MethodResult:
    render = load_render_metrics(ROOT / model_path, iteration)
    geom = load_geometry_metrics(ROOT / model_path, iteration)
    available = all(v == v for v in [*render.values(), *geom.values()])
    return MethodResult(
        row_id=row_id,
        scene=scene,
        method=method,
        model_path=model_path,
        iteration=iteration,
        role=role,
        wandb_run=wandb_run,
        psnr=render["psnr"],
        ssim=render["ssim"],
        lpips=render["lpips"],
        abs_rel=geom["abs_rel"],
        depth_mae=geom["depth_mae"],
        normal_mean_ang_deg=geom["normal_mean_ang_deg"],
        triangles=triangles,
        vertices=vertices,
        status="AVAILABLE" if available else "MISSING",
        notes=notes,
    )


def default_rows() -> list[MethodResult]:
    return [
        _row(
            "R57.clean9k",
            "courtyard",
            "clean_continue_7000to9000",
            "outputs/carnet/meshsplatopt/stageR57_02_courtyard_clean_continue_7000to9000/recovery_model",
            9000,
            "matched_clean_baseline",
            410254,
            444301,
            "ucqyn1ym",
        ),
        _row(
            "R57.compact70",
            "courtyard",
            "clean_to_compact_prune70_recovery_7000to9000",
            "outputs/carnet/meshsplatopt/stageR57_01_courtyard_prune70_recovery_7000to9000/recovery_model",
            9000,
            "negative_public_scene_check",
            123076,
            190787,
            "kgazucjj",
            "Fails matched clean on render and depth; keep as documented failure mode.",
        ),
        _row(
            "R58.clean9k",
            "bonsai",
            "clean_continue_7000to9000",
            "outputs/carnet/meshsplatopt/stageR58_02_bonsai_clean_continue_7000to9000/recovery_model",
            9000,
            "matched_clean_baseline",
            2487474,
            2478890,
            "ulv6dpku",
        ),
        _row(
            "R58.compact70",
            "bonsai",
            "clean_to_compact_prune70_recovery_7000to9000",
            "outputs/carnet/meshsplatopt/stageR58_01_bonsai_prune70_recovery_7000to9000/recovery_model",
            9000,
            "public_scene_quality_dominating",
            746242,
            720177,
            "82v2cg9z",
            "Dominates matched clean on render, sparse depth, normal, and topology.",
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[MethodResult], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-scene matched clean-to-compact results",
        "",
        "All rows use independent `render.py`, `metrics.py`, and `evaluate_geometry_colmap.py` outputs at iteration 9000.",
        "",
        "| row | scene | role | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal deg | triangles | W&B | notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.row_id} | {row.scene} | {row.role} | {row.psnr:.6f} | {row.ssim:.6f} | "
            f"{row.lpips:.6f} | {row.abs_rel:.6f} | {row.depth_mae:.6f} | "
            f"{row.normal_mean_ang_deg:.6f} | {row.triangles or ''} | `{row.wandb_run}` | {row.notes} |"
        )

    lines.extend(["", "## Matched comparisons", ""])
    lines.append("| candidate | baseline | pass | failed targets | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepthMAE | dNormal | triangle reduction |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in comparisons:
        delta = item["deltas"]
        lines.append(
            f"| {item['candidate_id']} | {item['baseline_id']} | {item['pass_all_targets']} | "
            f"{','.join(item['failed_targets'])} | {delta['psnr']:.6f} | {delta['ssim']:.6f} | "
            f"{delta['lpips']:.6f} | {delta['abs_rel']:.6f} | {delta['depth_mae']:.6f} | "
            f"{delta['normal_mean_ang_deg']:.6f} | {delta['triangle_reduction']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public-scene matched clean-to-compact results.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/cross_scene_clean_to_compact_tables")
    parser.add_argument("--triangle-reduction-min", type=float, default=0.5)
    args = parser.parse_args()

    rows = default_rows()
    by_id = {row.row_id: row for row in rows}
    targets = MetricTargets(triangle_reduction_min=args.triangle_reduction_min)
    comparisons = [
        compare_to_baseline(by_id["R57.compact70"], by_id["R57.clean9k"], targets).to_dict(),
        compare_to_baseline(by_id["R58.compact70"], by_id["R58.clean9k"], targets).to_dict(),
    ]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [row.to_dict() for row in rows], "comparisons": comparisons}
    (out_dir / "cross_scene_clean_to_compact_results.json").write_text(
        json.dumps(clean_json(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "cross_scene_clean_to_compact_results.csv", [row.to_dict() for row in rows])
    _write_md(out_dir / "cross_scene_clean_to_compact_results.md", rows, comparisons)
    print(f"Wrote {len(rows)} rows and {len(comparisons)} matched comparisons to {out_dir}")
    return 0 if all(row.status == "AVAILABLE" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
