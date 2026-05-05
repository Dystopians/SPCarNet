#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402


HIGHER_IS_BETTER = ("psnr", "ssim")
LOWER_IS_BETTER = ("lpips", "abs_rel", "depth_mae", "normal_mean_ang_deg")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _per_view_psnr(model: Path, iteration: int) -> dict[str, float]:
    payload = _read_json(model / "per_view.json")
    values = payload.get(f"ours_{iteration}", {}).get("PSNR", {})
    return {str(k): float(v) for k, v in values.items()}


def _render_geom(model: Path, iteration: int) -> dict[str, float]:
    render = load_render_metrics(model, iteration)
    geom = load_geometry_metrics(model, iteration)
    return {
        "psnr": float(render["psnr"]),
        "ssim": float(render["ssim"]),
        "lpips": float(render["lpips"]),
        "abs_rel": float(geom["abs_rel"]),
        "depth_mae": float(geom["depth_mae"]),
        "normal_mean_ang_deg": float(geom["normal_mean_ang_deg"]),
    }


def _delta(candidate: float, parent: float, higher: bool) -> float:
    return candidate - parent if higher else parent - candidate


def _status(metrics: dict[str, dict[str, float]], per_view: dict[str, Any]) -> str:
    checks = []
    for key in HIGHER_IS_BETTER:
        checks.append(metrics[key]["candidate_minus_parent"] >= 0.0)
    for key in LOWER_IS_BETTER:
        checks.append(metrics[key]["parent_minus_candidate"] >= 0.0)
    checks.append(float(per_view["candidate_minus_parent_min_delta"]) >= 0.0)
    checks.append(float(per_view["candidate_minus_parent_median_delta"]) >= 0.0)
    return "ACCEPT_STRICT_PARETO" if all(checks) else "REJECT_NOT_PARENT_PARETO"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a candidate continuation against its parent checkpoint line.")
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--parent-iteration", type=int, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-iteration", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parent = ROOT / args.parent_model
    candidate = ROOT / args.candidate_model
    parent_metrics = _render_geom(parent, args.parent_iteration)
    candidate_metrics = _render_geom(candidate, args.candidate_iteration)
    metrics: dict[str, dict[str, float]] = {}
    for key in HIGHER_IS_BETTER:
        metrics[key] = {
            "parent": parent_metrics[key],
            "candidate": candidate_metrics[key],
            "candidate_minus_parent": candidate_metrics[key] - parent_metrics[key],
        }
    for key in LOWER_IS_BETTER:
        metrics[key] = {
            "parent": parent_metrics[key],
            "candidate": candidate_metrics[key],
            "parent_minus_candidate": parent_metrics[key] - candidate_metrics[key],
        }

    parent_pv = _per_view_psnr(parent, args.parent_iteration)
    candidate_pv = _per_view_psnr(candidate, args.candidate_iteration)
    common = sorted(set(parent_pv) & set(candidate_pv))
    deltas = [candidate_pv[name] - parent_pv[name] for name in common]
    deltas_sorted = sorted(deltas)
    per_view = {
        "common": len(common),
        "candidate_minus_parent_min_delta": min(deltas_sorted) if deltas_sorted else math.nan,
        "candidate_minus_parent_median_delta": deltas_sorted[len(deltas_sorted) // 2] if deltas_sorted else math.nan,
        "candidate_minus_parent_max_delta": max(deltas_sorted) if deltas_sorted else math.nan,
        "num_negative": sum(1 for value in deltas if value < 0.0),
    }
    payload = {
        "parent_model": args.parent_model,
        "parent_iteration": args.parent_iteration,
        "candidate_model": args.candidate_model,
        "candidate_iteration": args.candidate_iteration,
        "metrics": metrics,
        "per_view_psnr": per_view,
    }
    payload["status"] = _status(metrics, per_view)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "per_view": per_view}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
