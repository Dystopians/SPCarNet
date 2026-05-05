#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF52_parking_delayed_robust_calibrated_gate_ratio004_7000_report.md"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delayed_root", default="outputs/carnet/meshprior/stageF52_parking_delayed_robust_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_delayed_robust_calibrated_gate")
    parser.add_argument("--robust_root", default="outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_robust_calibrated_gate_gpu4_retry")
    parser.add_argument("--calibrated_root", default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_calibrated_gate")
    parser.add_argument("--strict_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_gated")
    parser.add_argument("--nogate_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_no_gate")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stageF52_parking_delayed_robust_calibrated_gate_ratio004_7000/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    rows = [
        helper._summarize("strict_gate_ratio004_7000", Path(args.strict_root), args.iteration, True),
        helper._summarize("calibrated_gate_ratio004_7000", Path(args.calibrated_root), args.iteration, True),
        helper._summarize("robust_gate_early_ratio004_7000", Path(args.robust_root), args.iteration, True),
        helper._summarize("delayed_robust_gate_ratio004_7000", Path(args.delayed_root), args.iteration, True),
        helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False),
    ]
    refs = {row["label"]: row for row in rows}
    delayed = refs["delayed_robust_gate_ratio004_7000"]
    payload = {
        "decision": "F52_COMPLETE" if all(row["results_ready"] for row in rows) else "F52_IN_PROGRESS",
        "purpose": "parking ratio0.04 delayed robust calibrated-gate test after F51 showed early-accept geometry tradeoff",
        "rows": rows,
        "deltas_delayed_minus_strict": _metric_delta_row(delayed, refs["strict_gate_ratio004_7000"]),
        "deltas_delayed_minus_calibrated": _metric_delta_row(delayed, refs["calibrated_gate_ratio004_7000"]),
        "deltas_delayed_minus_early_robust": _metric_delta_row(delayed, refs["robust_gate_early_ratio004_7000"]),
        "deltas_delayed_minus_no_gate": _metric_delta_row(delayed, refs["no_gate_ratio004_7000"]),
    }
    if payload["decision"] == "F52_COMPLETE" and _all_metric_win(payload["deltas_delayed_minus_calibrated"]):
        payload["interpretation"] = "F52 is a strict all-metric repair over the F50 calibrated-gate reference."
    elif payload["decision"] == "F52_COMPLETE":
        payload["interpretation"] = "F52 is complete but not a strict all-metric repair; inspect metric deltas to decide the next schedule/ratio fix."
    else:
        payload["interpretation"] = "F52 is still in progress."

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF52_parking_delayed_robust_calibrated_gate_ratio004_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F52 - Parking Delayed Robust Calibrated Gate Ratio0.04 7000-Iteration Repair",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F52 tests whether F51's remaining weakness is early topology timing: it delays candidate selection until after a 1400-iteration geometry acquisition window and 100 stats iterations, while keeping the robust AbsRel reliability gate.",
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
        ("delayed - strict", payload["deltas_delayed_minus_strict"]),
        ("delayed - calibrated", payload["deltas_delayed_minus_calibrated"]),
        ("delayed - early_robust", payload["deltas_delayed_minus_early_robust"]),
        ("delayed - no_gate", payload["deltas_delayed_minus_no_gate"]),
    ):
        lines.append(f"| {label} | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |")
    lines += ["", "## Interpretation", "", payload["interpretation"], ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "delayed_ready": delayed["results_ready"]}, indent=2))


if __name__ == "__main__":
    main()
