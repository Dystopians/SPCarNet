#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF43_bonsai_gate_removed_7000_report.md"


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


def _wins_from_delta(delta: dict[str, Any]) -> dict[str, bool]:
    return {
        "render_no_gate": (
            delta.get("psnr") is not None
            and delta["psnr"] > 0
            and delta.get("ssim") is not None
            and delta["ssim"] > 0
            and delta.get("lpips") is not None
            and delta["lpips"] < 0
        ),
        "geometry_no_gate": (
            delta.get("absrel") is not None
            and delta["absrel"] < 0
            and delta.get("depth_mae") is not None
            and delta["depth_mae"] < 0
            and delta.get("normal") is not None
            and delta["normal"] < 0
        ),
    }


def _interpret(payload: dict[str, Any]) -> str:
    if payload["decision"] != "F43_COMPLETE":
        return (
            "F43 is incomplete. Do not draw conclusions until both bonsai rows have train, render, "
            "image metrics, geometry metrics, checkpoints, and W&B records."
        )

    gated = payload["gated"]
    nogate = payload["no_gate"]
    d = payload["deltas_no_gate_minus_gated"]
    wins = _wins_from_delta(d)
    mechanism_diverged = gated["committed_rounds"] != nogate["committed_rounds"] or gated["rollback_rounds"] != nogate["rollback_rounds"]

    parts = []
    if mechanism_diverged:
        parts.append(
            "Mechanism divergence is real: the same bonsai ratio0.02 7000-step schedule produces different "
            "commit/rollback behavior when the counterfactual gate is enabled versus removed."
        )
    if wins["render_no_gate"] and wins["geometry_no_gate"]:
        parts.append(
            "This is a negative gate-generalization result for the current bonsai schedule: no-gate wins all "
            "three render metrics and all three sparse geometry proxies, while also ending with a much smaller mesh."
        )
    elif wins["render_no_gate"]:
        parts.append(
            "This is at least a render-negative gate-generalization result: no-gate wins PSNR, SSIM, and LPIPS."
        )
    elif wins["geometry_no_gate"]:
        parts.append(
            "Sparse geometry favors no-gate on this bonsai schedule even if render metrics are mixed."
        )
    parts.append(
        "The safe paper conclusion is not broad multi-scene final-budget gate superiority. F43 should be used "
        "as a weakness-finding ablation: the current gate/rollback policy can be too conservative or can steer "
        "the recovery trajectory poorly on bonsai, so any final claim must emphasize the validated compact-recovery "
        "main table and the parking/synthetic unsafe-edit evidence, while listing adaptive scene-aware gate calibration "
        "as required future/fix work before claiming universal gate dominance."
    )
    return " ".join(parts)


def write_report(out: Path, gated: dict[str, Any], nogate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "decision": "F43_COMPLETE" if gated["results_ready"] and nogate["results_ready"] else "F43_IN_PROGRESS",
        "purpose": "7000-iteration bonsai ratio0.02 gate-on/gate-off multi-scene ablation after F42",
        "gated": gated,
        "no_gate": nogate,
        "deltas_no_gate_minus_gated": {
            key: _delta(nogate["metrics"].get(key), gated["metrics"].get(key))
            for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")
        },
    }
    payload["interpretation"] = _interpret(payload)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF43_bonsai_gate_removed_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F43 - Bonsai Gate-Removed 7000-Iteration Ablation",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F43 extends the real gate-removed ablation beyond parking. Both rows use the same bonsai ratio0.02 PRISM schedule, 7000 training iterations, online W&B, independent test-set rendering metrics, and sparse COLMAP geometry evaluation.",
        "",
        f"- summary JSON: `{json_path}`",
        "",
        "## Runs",
        "",
        "| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (gated, nogate):
        lines.append(
            "| {label} | `{gate}` | {wandb} | `{ready}` | {rounds} | {commits} | {rollbacks} | {selected} | {committed_selected} | {triangles} |".format(
                label=row["label"],
                gate=row["gate_enabled"],
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
    for row in (gated, nogate):
        m = row["metrics"]
        lines.append(
            f"| {row['label']} | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'))} | {_fmt(m.get('lpips'))} | {_fmt(m.get('absrel'))} | {_fmt(m.get('depth_mae'))} | {_fmt(m.get('normal'))} |"
        )
    d = payload["deltas_no_gate_minus_gated"]
    lines.append(
        f"| no_gate - gated | {_fmt(d.get('psnr'))} | {_fmt(d.get('ssim'))} | {_fmt(d.get('lpips'))} | {_fmt(d.get('absrel'))} | {_fmt(d.get('depth_mae'))} | {_fmt(d.get('normal'))} |"
    )
    lines += [
        "",
        "## First Candidate Round",
        "",
        "| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (gated, nogate):
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
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated_root", required=True)
    parser.add_argument("--nogate_root", required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshprior/stageF43_bonsai_gate_removed_7000/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    gated = helper._summarize("gated_bonsai_ratio002_7000", Path(args.gated_root), args.iteration, True)
    nogate = helper._summarize("no_gate_bonsai_ratio002_7000", Path(args.nogate_root), args.iteration, False)
    payload = write_report(ROOT / args.output_dir, gated, nogate)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "gated_ready": gated["results_ready"],
                "no_gate_ready": nogate["results_ready"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
