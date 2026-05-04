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
    wandb_run: str = "",
    notes: str = "",
) -> MethodResult:
    render = load_render_metrics(ROOT / model_path, iteration)
    geom = load_geometry_metrics(ROOT / model_path, iteration)
    status = "AVAILABLE" if all(v == v for v in [*render.values(), *geom.values()]) else "MISSING"
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
        status=status,
        notes=notes,
    )


def parking_rows() -> list[MethodResult]:
    rows = [
        _row(
            "clean22k",
            "parking_phone_tiny",
            "clean_long",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model",
            22000,
            "strongest_clean_render_baseline",
            8548242,
            2286499,
            "uus7fi39",
        ),
        _row(
            "clean30k",
            "parking_phone_tiny",
            "clean_long",
            "outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_22000to30000/model",
            30000,
            "clean_long_continuation_baseline",
            8548242,
            2286499,
            "2q807xuf",
        ),
        _row(
            "R48.01",
            "parking_phone_tiny",
            "clean_to_compact_prune80_recovery",
            "outputs/carnet/meshsplatopt/stageR48_01_prune80_clean_recovery_22000to26000/recovery_model",
            26000,
            "compact_pareto",
            1709648,
            1322214,
            "1n6jv232",
        ),
        _row(
            "R53.01",
            "parking_phone_tiny",
            "clean_to_compact_prune70_recovery",
            "outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model",
            26000,
            "headline_quality_dominating",
            2564473,
            1661616,
            "q15qg2b8",
        ),
        _row(
            "R55.01",
            "parking_phone_tiny",
            "clean_to_compact_prune65_recovery",
            "outputs/carnet/meshsplatopt/stageR55_01_prune65_clean_recovery_22000to26000/recovery_model",
            26000,
            "lpips_normal_pareto",
            2991885,
            1783669,
            "ja7t57cx",
        ),
    ]
    # R56 was evaluated inside train.py only; keep it out of paper-facing rows until independent geometry exists.
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[MethodResult], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Clean-to-compact result table",
        "",
        "| row | role | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal deg | triangles | W&B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.row_id} | {row.role} | {row.psnr:.6f} | {row.ssim:.6f} | {row.lpips:.6f} | "
            f"{row.abs_rel:.6f} | {row.depth_mae:.6f} | {row.normal_mean_ang_deg:.6f} | "
            f"{row.triangles or ''} | `{row.wandb_run}` |"
        )
    lines.extend(["", "## Comparisons against clean22k", ""])
    lines.append("| candidate | pass | failed targets | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepthMAE | dNormal | triangle reduction |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in comparisons:
        delta = item["deltas"]
        lines.append(
            f"| {item['candidate_id']} | {item['pass_all_targets']} | {','.join(item['failed_targets'])} | "
            f"{delta['psnr']:.6f} | {delta['ssim']:.6f} | {delta['lpips']:.6f} | "
            f"{delta['abs_rel']:.6f} | {delta['depth_mae']:.6f} | {delta['normal_mean_ang_deg']:.6f} | "
            f"{delta['triangle_reduction']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/clean_to_compact_tables")
    parser.add_argument("--psnr-margin", type=float, default=0.0)
    parser.add_argument("--ssim-margin", type=float, default=0.0)
    parser.add_argument("--lpips-margin", type=float, default=0.0)
    parser.add_argument("--triangle-reduction-min", type=float, default=0.5)
    args = parser.parse_args()

    rows = parking_rows()
    baseline = next(row for row in rows if row.row_id == "clean22k")
    targets = MetricTargets(
        psnr_margin=args.psnr_margin,
        ssim_margin=args.ssim_margin,
        lpips_margin=args.lpips_margin,
        triangle_reduction_min=args.triangle_reduction_min,
    )
    comparisons = [
        compare_to_baseline(row, baseline, targets).to_dict()
        for row in rows
        if row.row_id != baseline.row_id and row.scene == baseline.scene
    ]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = [row.to_dict() for row in rows]
    (out_dir / "clean_to_compact_results.json").write_text(
        json.dumps(clean_json({"rows": rows_payload, "comparisons": comparisons}), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "clean_to_compact_results.csv", rows_payload)
    _write_md(out_dir / "clean_to_compact_results.md", rows, comparisons)
    print(f"Wrote {len(rows)} rows and {len(comparisons)} comparisons to {out_dir}")
    return 0 if all(row.status == "AVAILABLE" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
