#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF50_parking_calibrated_gate_ratio004_7000_report.md"


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


def _interpret(payload: dict[str, Any]) -> str:
    if payload["decision"] != "F50_COMPLETE":
        return "F50 is incomplete. Wait for calibrated train, render, metrics, geometry, checkpoint, and W&B before judging cross-scene calibrated gating."
    calibrated = payload["calibrated_gate"]
    strict = payload["strict_gate_reference"]
    d_strict = payload["deltas_calibrated_minus_strict_gate"]
    d_nogate = payload["deltas_calibrated_minus_no_gate"]
    same_mechanism = calibrated["committed_rounds"] == strict["committed_rounds"] and calibrated["rollback_rounds"] == strict["rollback_rounds"]
    better_than_strict_render = d_strict["psnr"] > 0 and d_strict["ssim"] > 0 and d_strict["lpips"] < 0
    better_than_strict_geometry = d_strict["absrel"] < 0 and d_strict["depth_mae"] < 0 and d_strict["normal"] < 0
    better_than_nogate_render = d_nogate["psnr"] > 0 and d_nogate["ssim"] > 0 and d_nogate["lpips"] < 0
    better_than_nogate_geometry = d_nogate["absrel"] < 0 and d_nogate["depth_mae"] < 0 and d_nogate["normal"] < 0
    parts = []
    if same_mechanism:
        parts.append("The calibrated gate follows the same mechanism as the strict F42 gate on parking: it rejects and rolls back the same no-accept ratio0.04 candidate round, so F50 does not replicate the F44 bonsai behavior of accepting recoverable edits.")
    if better_than_strict_render:
        parts.append(
            "It improves over the strict gate on all render metrics: "
            f"PSNR {d_strict['psnr']:+.6f}, SSIM {d_strict['ssim']:+.6f}, LPIPS {d_strict['lpips']:+.6f}."
        )
    else:
        parts.append(
            "Relative to strict gate, calibrated gate gives back render quality "
            f"(PSNR {d_strict['psnr']:+.6f}, SSIM {d_strict['ssim']:+.6f}, LPIPS {d_strict['lpips']:+.6f})."
        )
    if better_than_strict_geometry:
        parts.append(
            "It does improve sparse geometry proxies versus strict gate "
            f"(AbsRel {d_strict['absrel']:+.6f}, Depth MAE {d_strict['depth_mae']:+.6f}, Normal {d_strict['normal']:+.6f})."
        )
    if better_than_nogate_render:
        parts.append(
            "Against no-gate, calibrated gate still wins all render metrics while preserving rollback metadata "
            f"(PSNR {d_nogate['psnr']:+.6f}, SSIM {d_nogate['ssim']:+.6f}, LPIPS {d_nogate['lpips']:+.6f})."
        )
    if not better_than_nogate_geometry:
        parts.append(
            "No-gate remains better on sparse geometry proxies, so the paper claim should stay narrow: F50 supports render-quality rollback safety on parking, but not broad calibrated-gate superiority."
        )
    return " ".join(parts)


def write_report(out: Path, calibrated: dict[str, Any], strict: dict[str, Any], nogate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "decision": "F50_COMPLETE" if calibrated["results_ready"] and strict["results_ready"] and nogate["results_ready"] else "F50_IN_PROGRESS",
        "purpose": "parking ratio0.04 7000-iteration calibrated-gate cross-scene replication against F42 strict/no-gate references",
        "calibrated_gate": calibrated,
        "strict_gate_reference": strict,
        "no_gate_reference": nogate,
        "deltas_calibrated_minus_strict_gate": _deltas(calibrated, strict),
        "deltas_calibrated_minus_no_gate": _deltas(calibrated, nogate),
    }
    payload["interpretation"] = _interpret(payload)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF50_parking_calibrated_gate_ratio004_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = (strict, calibrated, nogate)
    lines = [
        "# Final Stage F50 - Parking Calibrated Gate Ratio0.04 7000-Iteration Replication",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F50 tests whether the F44 calibrated counterfactual-gate thresholds replicate on the parking F42 ratio0.04 7000-step schedule. The strict-gate and no-gate rows are the completed F42 references; the calibrated row is a new online-W&B long run with the same candidate schedule and relaxed immediate gate thresholds.",
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
    lines += [
        "",
        "## Metrics",
        "",
        "| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['label']} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | {_fmt(m.get('lpips'))} | {_fmt(m.get('absrel'))} | {_fmt(m.get('depth_mae'))} | {_fmt(m.get('normal'))} |")
    for label, d in (
        ("calibrated - strict_gate", payload["deltas_calibrated_minus_strict_gate"]),
        ("calibrated - no_gate", payload["deltas_calibrated_minus_no_gate"]),
    ):
        lines.append(f"| {label} | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |")
    lines += [
        "",
        "## First Candidate Round",
        "",
        "| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
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
    parser.add_argument(
        "--calibrated_root",
        default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/parking_7000iter_ratio004_calibrated_gate",
    )
    parser.add_argument(
        "--strict_root",
        default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_gated",
    )
    parser.add_argument(
        "--nogate_root",
        default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/parking_7000iter_ratio004_no_gate",
    )
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    calibrated = helper._summarize("calibrated_gate_ratio004_7000", Path(args.calibrated_root), args.iteration, True)
    strict = helper._summarize("strict_gate_ratio004_7000", Path(args.strict_root), args.iteration, True)
    nogate = helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False)
    payload = write_report(ROOT / args.output_dir, calibrated, strict, nogate)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "calibrated_ready": calibrated["results_ready"],
                "strict_ready": strict["results_ready"],
                "no_gate_ready": nogate["results_ready"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
