#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF62_parking_adaptive_arbitrated_policy_7000_report.md"


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


def _deltas(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return {key: _delta(row, ref, key) for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")}


def _all_metric_win(d: dict[str, Any]) -> bool:
    return d["psnr"] > 0 and d["ssim"] > 0 and d["lpips"] < 0 and d["absrel"] < 0 and d["depth_mae"] < 0 and d["normal"] < 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f62_root", default="outputs/carnet/meshprior/stageF62_parking_adaptive_arbitrated_policy_7000/parking_7000iter_adaptive_arbitrated_policy")
    parser.add_argument("--f58_root", default="outputs/carnet/meshprior/stageF58_parking_adaptive_pareto_policy_7000/parking_7000iter_adaptive_pareto_policy")
    parser.add_argument("--f56_root", default="outputs/carnet/meshprior/stageF56_parking_delayed_robust_calibrated_gate_ratio00125_7000/parking_7000iter_ratio00125_delayed_robust_calibrated_gate")
    parser.add_argument("--f55_root", default="outputs/carnet/meshprior/stageF55_parking_delayed_robust_calibrated_gate_ratio0015_7000/parking_7000iter_ratio0015_delayed_robust_calibrated_gate")
    parser.add_argument("--f54_root", default="outputs/carnet/meshprior/stageF54_parking_delayed_robust_calibrated_gate_ratio001_7000/parking_7000iter_ratio001_delayed_robust_calibrated_gate")
    parser.add_argument("--f53_root", default="outputs/carnet/meshprior/stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000/parking_7000iter_ratio002_delayed_robust_calibrated_gate")
    parser.add_argument("--nogate_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_no_gate")
    parser.add_argument("--strict_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_gated")
    parser.add_argument("--calibrated_root", default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_calibrated_gate")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stageF62_parking_adaptive_arbitrated_policy_7000/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    rows = [
        helper._summarize("strict_gate_ratio004_7000", Path(args.strict_root), args.iteration, True),
        helper._summarize("calibrated_gate_ratio004_7000", Path(args.calibrated_root), args.iteration, True),
        helper._summarize("delayed_ratio002_F53", Path(args.f53_root), args.iteration, True),
        helper._summarize("delayed_ratio001_F54", Path(args.f54_root), args.iteration, True),
        helper._summarize("delayed_ratio0015_F55", Path(args.f55_root), args.iteration, True),
        helper._summarize("delayed_ratio00125_F56", Path(args.f56_root), args.iteration, True),
        helper._summarize("adaptive_pareto_policy_F58", Path(args.f58_root), args.iteration, True),
        helper._summarize("adaptive_arbitrated_policy_F62", Path(args.f62_root), args.iteration, True),
        helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False),
    ]
    refs = {row["label"]: row for row in rows}
    f62 = refs["adaptive_arbitrated_policy_F62"]
    f58 = refs["adaptive_pareto_policy_F58"]
    f56 = refs["delayed_ratio00125_F56"]
    nogate = refs["no_gate_ratio004_7000"]
    strict = refs["strict_gate_ratio004_7000"]
    calibrated = refs["calibrated_gate_ratio004_7000"]
    required = [f62, f58, f56, nogate, strict, calibrated, refs["delayed_ratio002_F53"], refs["delayed_ratio001_F54"]]
    payload = {
        "decision": "F62_COMPLETE" if all(row["results_ready"] for row in required) else "F62_IN_PROGRESS",
        "purpose": "adaptive CSEF policy evaluation against fixed-ratio parking rows and fair no-gate baselines",
        "rows": rows,
        "deltas_f62_minus_no_gate": _deltas(f62, nogate),
        "deltas_f62_minus_strict": _deltas(f62, strict),
        "deltas_f62_minus_calibrated": _deltas(f62, calibrated),
        "deltas_f62_minus_f58": _deltas(f62, f58),
        "deltas_f62_minus_f56": _deltas(f62, f56),
    }
    if payload["decision"] == "F62_COMPLETE" and _all_metric_win(payload["deltas_f62_minus_no_gate"]):
        if _all_metric_win(payload["deltas_f62_minus_f56"]):
            payload["interpretation"] = "F62 adaptive CSEF policy strictly beats both no-gate and the strongest fixed-ratio F56 on all six tracked metrics."
        else:
            payload["interpretation"] = "F62 adaptive CSEF policy strictly beats no-gate on all six tracked metrics but remains mixed versus F56."
    elif payload["decision"] == "F62_COMPLETE":
        payload["interpretation"] = "F62 is complete but still mixed versus no-gate; inspect deltas before promoting the adaptive policy."
    else:
        payload["interpretation"] = "F62 is still in progress."

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF62_parking_adaptive_arbitrated_policy_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F62 - Parking Adaptive Arbitrated CSEF Policy 7000",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F62 replaces the fixed ratio sweep with an Adaptive Arbitrated CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.",
        "",
        f"- summary JSON: `{json_path}`",
        "",
        "## Runs",
        "",
        "| row | W&B | ready | commits | rollbacks | selected | committed selected | final triangles |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['label']} | {row['wandb_url'] or 'NA'} | `{row['results_ready']}` | {row['committed_rounds']} | {row['rollback_rounds']} | {row['selected_candidates']} | {row['committed_selected_candidates']} | {_fmt(row['checkpoint']['triangles'], 0)} |")
    lines += ["", "## Metrics", "", "| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['label']} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | {_fmt(m.get('lpips'))} | {_fmt(m.get('absrel'))} | {_fmt(m.get('depth_mae'))} | {_fmt(m.get('normal'))} |")
    for label, d in (
        ("F62 - no_gate", payload["deltas_f62_minus_no_gate"]),
        ("F62 - strict", payload["deltas_f62_minus_strict"]),
        ("F62 - calibrated", payload["deltas_f62_minus_calibrated"]),
        ("F62 - F58", payload["deltas_f62_minus_f58"]),
        ("F62 - F56", payload["deltas_f62_minus_f56"]),
    ):
        lines.append(f"| {label} | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |")
    lines += ["", "## Interpretation", "", payload["interpretation"], ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "f62_ready": f62["results_ready"], "interpretation": payload["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
