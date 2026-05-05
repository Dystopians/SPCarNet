#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.final_collect_stageF76_fixed_adaptive_policy_multiscene import (  # noqa: E402
    CLEAN_MODELS,
    _build_specs,
    _row,
)


def _stage_rows(stage_id: str, policy_tag: str, scenes: list[str]) -> list[dict[str, Any]]:
    stage_group = f"final_stage{stage_id}_fixed_adaptive_policy_multiscene"
    rows = []
    for row in (_row(spec) for spec in _build_specs(stage_group, policy_tag, scenes)):
        payload = row.to_dict()
        payload["stage_id"] = stage_id
        payload["policy_tag"] = policy_tag
        rows.append(payload)
    return rows


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        scene_rows = [row for row in rows if row["scene"] == scene and row["status"] != "PENDING_OR_MISSING_EVAL"]
        if not scene_rows:
            continue
        pass_count = sum(row["status"] == "PASS_ALL_METRIC_CLEAN_WIN" for row in scene_rows)
        summary: dict[str, Any] = {
            "scene": scene,
            "available": len(scene_rows),
            "pass_count": pass_count,
            "all_pass": pass_count == len(scene_rows),
        }
        for key in ("d_psnr", "d_ssim", "d_lpips", "d_abs_rel", "d_depth_mae", "d_normal"):
            values = _finite([row[key] for row in scene_rows])
            summary[f"{key}_mean"] = statistics.fmean(values) if values else math.nan
            summary[f"{key}_min"] = min(values) if values else math.nan
            summary[f"{key}_max"] = max(values) if values else math.nan
        out.append(summary)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}" if math.isfinite(value) else "nan"
    return str(value)


def _write_md(path: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Fixed Policy Robustness",
        "",
        "Rows repeat the same selector and recovery recipe across the listed stages / seeds.",
        "",
        "## Rows",
        "",
        "| stage | scene | W&B | prune | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage_id']} | {row['scene']} | `{row['wandb']}` | {float(row['adaptive_fraction']):.4f} | "
            f"{float(row['d_psnr']):+.6f} | {float(row['d_ssim']):+.6f} | {float(row['d_lpips']):+.6f} | "
            f"{float(row['d_abs_rel']):+.6f} | {float(row['d_depth_mae']):+.6f} | {float(row['d_normal']):+.6f} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Stability",
            "",
            "| scene | available | pass | all pass | min dPSNR | max dLPIPS | max dDepth | max dNormal |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['scene']} | {row['available']} | {row['pass_count']} | {row['all_pass']} | "
            f"{_fmt(row['d_psnr_min'])} | {_fmt(row['d_lpips_max'])} | {_fmt(row['d_depth_mae_max'])} | {_fmt(row['d_normal_max'])} |"
        )
    total_available = sum(1 for row in rows if row["status"] != "PENDING_OR_MISSING_EVAL")
    total_pass = sum(1 for row in rows if row["status"] == "PASS_ALL_METRIC_CLEAN_WIN")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- available rows: `{total_available}` / `{len(rows)}`",
            f"- all-metric clean wins: `{total_pass}` / `{len(rows)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect fixed-policy seed robustness evidence.")
    parser.add_argument("--scenes", default=",".join(CLEAN_MODELS.keys()))
    parser.add_argument(
        "--runs",
        default="F81:adaptive_global_policy_v5_seed1,F82:adaptive_global_policy_v5_seed0",
        help="Comma-separated stage:policy_tag pairs.",
    )
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF82_policy_v5_robustness")
    args = parser.parse_args()
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    rows = []
    for item in args.runs.split(","):
        if not item.strip():
            continue
        stage_id, policy_tag = item.split(":", 1)
        rows.extend(_stage_rows(stage_id.strip(), policy_tag.strip(), scenes))
    summaries = _summaries(rows)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_robustness_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (out / "policy_robustness_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "policy_robustness_rows.csv", rows)
    _write_csv(out / "policy_robustness_summary.csv", summaries)
    _write_md(out / "policy_robustness_results.md", rows, summaries)
    print(f"Wrote robustness evidence to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
