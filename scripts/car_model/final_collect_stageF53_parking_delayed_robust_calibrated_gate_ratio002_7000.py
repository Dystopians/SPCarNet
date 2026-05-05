#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000_report.md"


def _load_helper():
    spec = importlib.util.spec_from_file_location("final_collect_stageF39_gate_removed_ablation", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(row: dict[str, Any], ref: dict[str, Any], key: str) -> Any:
    a = row["metrics"].get(key)
    b = ref["metrics"].get(key)
    if a is None or b is None:
        return None
    return a - b


def _metric_delta_row(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return {key: _delta(row, ref, key) for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")}


def _all_metric_win(d: dict[str, Any]) -> bool:
    return d["psnr"] > 0 and d["ssim"] > 0 and d["lpips"] < 0 and d["absrel"] < 0 and d["depth_mae"] < 0 and d["normal"] < 0


def _interpret(payload: dict[str, Any]) -> str:
    if payload["decision"] != "F53_COMPLETE":
        return "F53 is incomplete. Wait for train, render, metrics, geometry, checkpoint, and W&B before judging it."
    d_strict = payload["deltas_delayed_ratio002_minus_strict"]
    d_f50 = payload["deltas_delayed_ratio002_minus_calibrated"]
    d_f51 = payload["deltas_delayed_ratio002_minus_early_robust"]
    d_nogate = payload["deltas_delayed_ratio002_minus_no_gate"]
    parts = []
    if _all_metric_win(d_strict):
        parts.append("F53 is a strict all-metric win over the strict gate reference.")
    else:
        parts.append("F53 is not a strict all-metric win over the strict gate reference.")
    if _all_metric_win(d_f50):
        parts.append("It strictly repairs F50 calibrated gate.")
    else:
        parts.append("It is mixed versus F50 calibrated gate.")
    if _all_metric_win(d_f51):
        parts.append("It strictly repairs F51 early robust gate.")
    else:
        parts.append("It is mixed versus F51 early robust gate.")
    if _all_metric_win(d_nogate):
        parts.append("It strictly beats no-gate on all tracked render and sparse-geometry metrics.")
    else:
        parts.append("It is mixed versus no-gate, so further work is required before claiming universal dominance.")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delayed_ratio002_root", default="outputs/carnet/meshprior/stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000/parking_7000iter_ratio002_delayed_robust_calibrated_gate")
    parser.add_argument("--early_robust_root", default="outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_robust_calibrated_gate_gpu4_retry")
    parser.add_argument("--calibrated_root", default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_calibrated_gate")
    parser.add_argument("--strict_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_gated")
    parser.add_argument("--nogate_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_no_gate")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    rows = [
        helper._summarize("strict_gate_ratio004_7000", Path(args.strict_root), args.iteration, True),
        helper._summarize("calibrated_gate_ratio004_7000", Path(args.calibrated_root), args.iteration, True),
        helper._summarize("early_robust_gate_ratio004_7000", Path(args.early_robust_root), args.iteration, True),
        helper._summarize("delayed_robust_gate_ratio002_7000", Path(args.delayed_ratio002_root), args.iteration, True),
        helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False),
    ]
    refs = {row["label"]: row for row in rows}
    f53 = refs["delayed_robust_gate_ratio002_7000"]
    payload = {
        "decision": "F53_COMPLETE" if all(row["results_ready"] for row in rows) else "F53_IN_PROGRESS",
        "purpose": "parking delayed robust calibrated gate with lower ratio0.02 after F51/F52 exposed early-timing and over-aggressive-ratio weaknesses",
        "rows": rows,
        "deltas_delayed_ratio002_minus_strict": _metric_delta_row(f53, refs["strict_gate_ratio004_7000"]),
        "deltas_delayed_ratio002_minus_calibrated": _metric_delta_row(f53, refs["calibrated_gate_ratio004_7000"]),
        "deltas_delayed_ratio002_minus_early_robust": _metric_delta_row(f53, refs["early_robust_gate_ratio004_7000"]),
        "deltas_delayed_ratio002_minus_no_gate": _metric_delta_row(f53, refs["no_gate_ratio004_7000"]),
    }
    payload["interpretation"] = _interpret(payload)

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F53 - Parking Delayed Robust Calibrated Gate Ratio0.02 7000-Iteration Repair",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F53 delays candidate selection until geometry has become reliable and lowers the prune ratio from 0.04 to 0.02. This directly tests the F51/F52 diagnosis: early ratio0.04 commits can improve appearance but leave sparse geometry mixed, while delayed ratio0.04 is too aggressive to pass a reliable gate.",
        "",
        f"- summary JSON: `{json_path}`",
        "",
        "## Runs",
        "",
        "| row | W&B | ready | candidate rounds | commits | rollbacks | selected | committed selected | final triangles |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['wandb_url'] or 'NA'} | `{row['results_ready']}` | {row['candidate_rounds']} | {row['committed_rounds']} | {row['rollback_rounds']} | {row['selected_candidates']} | {row['committed_selected_candidates']} | {_fmt(row['checkpoint']['triangles'], 0)} |"
        )
    lines += ["", "## Metrics", "", "| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['label']} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | {_fmt(m.get('lpips'))} | {_fmt(m.get('absrel'))} | {_fmt(m.get('depth_mae'))} | {_fmt(m.get('normal'))} |")
    for label, d in (
        ("F53 - strict", payload["deltas_delayed_ratio002_minus_strict"]),
        ("F53 - calibrated", payload["deltas_delayed_ratio002_minus_calibrated"]),
        ("F53 - early_robust", payload["deltas_delayed_ratio002_minus_early_robust"]),
        ("F53 - no_gate", payload["deltas_delayed_ratio002_minus_no_gate"]),
    ):
        lines.append(f"| {label} | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |")
    lines += ["", "## First Candidate Round", "", "| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        first = row["first_round"]
        lines.append(
            f"| {row['label']} | {_fmt(first.get('iteration'), 0)} | `{first.get('committed')}` | {_fmt(first.get('counterfactual_accept'), 0)} | {_fmt(first.get('rollback'), 0)} | {_fmt(first.get('candidate_selected_count'), 0)} | {_fmt(first.get('pre_prune_triangle_count'), 0)} | {_fmt(first.get('post_prune_triangle_count'), 0)} |"
        )
    lines += ["", "## Interpretation", "", payload["interpretation"], ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "f53_ready": f53["results_ready"], "interpretation": payload["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
