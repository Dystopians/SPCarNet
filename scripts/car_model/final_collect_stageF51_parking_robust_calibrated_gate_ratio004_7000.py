#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF51_parking_robust_calibrated_gate_ratio004_7000_report.md"


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


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return a - b


def _deltas(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return {key: _delta(row["metrics"].get(key), ref["metrics"].get(key)) for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")}


def _all_render_better(d: dict[str, Any]) -> bool:
    return d["psnr"] > 0 and d["ssim"] > 0 and d["lpips"] < 0


def _all_geometry_better(d: dict[str, Any]) -> bool:
    return d["absrel"] < 0 and d["depth_mae"] < 0 and d["normal"] < 0


def _interpret(payload: dict[str, Any]) -> str:
    if payload["decision"] != "F51_COMPLETE":
        return "F51 is incomplete. Wait for robust calibrated train, render, metrics, geometry, checkpoint, and W&B before judging the repair."
    robust = payload["robust_calibrated_gate"]
    f50 = payload["calibrated_gate_reference"]
    strict = payload["strict_gate_reference"]
    nogate = payload["no_gate_reference"]
    d_strict = payload["deltas_robust_minus_strict_gate"]
    d_f50 = payload["deltas_robust_minus_calibrated_gate"]
    d_nogate = payload["deltas_robust_minus_no_gate"]
    parts = []
    if robust["committed_rounds"] > f50["committed_rounds"] and robust["committed_selected_candidates"] > 0:
        parts.append("F51 fixes the F50 mechanism weakness: robust calibrated gating commits the early ratio0.04 candidate that F50/strict gate rolled back.")
    else:
        parts.append("F51 does not fix the F50 mechanism weakness; it still fails to commit more candidates than the calibrated reference.")
    if _all_render_better(d_strict):
        parts.append("It beats strict gate on all render metrics.")
    elif _all_geometry_better(d_strict):
        parts.append("It beats strict gate on all sparse-geometry proxies, but render remains mixed.")
    if _all_render_better(d_f50) and _all_geometry_better(d_f50):
        parts.append("It is a strict all-metric improvement over F50 calibrated gate.")
    if _all_render_better(d_nogate) and _all_geometry_better(d_nogate):
        parts.append("It is a strict all-metric improvement over no-gate.")
    elif _all_render_better(d_nogate):
        parts.append("It remains render-positive versus no-gate, but sparse geometry is not fully dominant.")
    return " ".join(parts)


def write_report(
    out: Path,
    robust: dict[str, Any],
    f50: dict[str, Any],
    strict: dict[str, Any],
    nogate: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "decision": "F51_COMPLETE" if robust["results_ready"] and f50["results_ready"] and strict["results_ready"] and nogate["results_ready"] else "F51_IN_PROGRESS",
        "purpose": "parking ratio0.04 7000-iteration robust calibrated-gate repair after F50 AbsRel reliability diagnosis",
        "robust_calibrated_gate": robust,
        "calibrated_gate_reference": f50,
        "strict_gate_reference": strict,
        "no_gate_reference": nogate,
        "deltas_robust_minus_strict_gate": _deltas(robust, strict),
        "deltas_robust_minus_calibrated_gate": _deltas(robust, f50),
        "deltas_robust_minus_no_gate": _deltas(robust, nogate),
    }
    payload["interpretation"] = _interpret(payload)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF51_parking_robust_calibrated_gate_ratio004_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = (strict, f50, robust, nogate)
    lines = [
        "# Final Stage F51 - Parking Robust Calibrated Gate Ratio0.04 7000-Iteration Repair",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F51 tests the AbsRel-reliability repair: when baseline AbsRel is above the reliability threshold, the counterfactual gate does not let that unstable AbsRel delta alone reject an otherwise visually tiny edit. F51 is compared against F42 strict/no-gate and F50 calibrated-gate references.",
        "",
        f"- summary JSON: `{json_path}`",
        "",
        "## Runs",
        "",
        "| row | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {wandb} | `{ready}` | {rounds} | {commits} | {rollbacks} | {selected} | {committed_selected} | {triangles} |".format(
                label=row["label"],
                wandb=row["wandb_url"] or "NA",
                ready=row["results_ready"],
                rounds=row["candidate_rounds"],
                commits=row["committed_rounds"],
                rollbacks=row["rollback_rounds"],
                selected=row["selected_candidates"],
                committed_selected=row["committed_selected_candidates"],
                triangles=_fmt(row["checkpoint"]["triangles"], 0),
            )
        )
    lines += ["", "## Metrics", "", "| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['label']} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | {_fmt(m.get('lpips'))} | {_fmt(m.get('absrel'))} | {_fmt(m.get('depth_mae'))} | {_fmt(m.get('normal'))} |")
    for label, d in (
        ("robust - strict_gate", payload["deltas_robust_minus_strict_gate"]),
        ("robust - calibrated_gate", payload["deltas_robust_minus_calibrated_gate"]),
        ("robust - no_gate", payload["deltas_robust_minus_no_gate"]),
    ):
        lines.append(f"| {label} | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |")
    lines += ["", "## First Candidate Round", "", "| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        first = row["first_round"]
        lines.append(
            "| {label} | {iteration} | `{committed}` | {accept} | {rollback} | {selected} | {pre} | {post} |".format(
                label=row["label"],
                iteration=_fmt(first.get("iteration"), 0),
                committed=first.get("committed"),
                accept=_fmt(first.get("counterfactual_accept"), 0),
                rollback=_fmt(first.get("rollback"), 0),
                selected=_fmt(first.get("candidate_selected_count"), 0),
                pre=_fmt(first.get("pre_prune_triangle_count"), 0),
                post=_fmt(first.get("post_prune_triangle_count"), 0),
            )
        )
    lines += ["", "## Interpretation", "", payload["interpretation"], ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust_root", default="outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_robust_calibrated_gate_gpu4_retry")
    parser.add_argument("--calibrated_root", default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_calibrated_gate")
    parser.add_argument("--strict_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_gated")
    parser.add_argument("--nogate_root", default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_no_gate")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    robust = helper._summarize("robust_calibrated_gate_ratio004_7000", Path(args.robust_root), args.iteration, True)
    f50 = helper._summarize("calibrated_gate_ratio004_7000", Path(args.calibrated_root), args.iteration, True)
    strict = helper._summarize("strict_gate_ratio004_7000", Path(args.strict_root), args.iteration, True)
    nogate = helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False)
    payload = write_report(ROOT / args.output_dir, robust, f50, strict, nogate)
    print(json.dumps({"decision": payload["decision"], "robust_ready": robust["results_ready"]}, indent=2))


if __name__ == "__main__":
    main()
