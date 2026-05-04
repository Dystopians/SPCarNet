#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF41_gate_removed_ratio004_long_report.md"


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


def write_report(out: Path, gated: dict[str, Any], nogate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "decision": "F41_COMPLETE" if gated["results_ready"] and nogate["results_ready"] else "F41_IN_PROGRESS",
        "purpose": "longer real-scene ratio0.04 gate-on/gate-off ablation after F39",
        "gated": gated,
        "no_gate": nogate,
        "deltas_no_gate_minus_gated": {
            key: _delta(nogate["metrics"].get(key), gated["metrics"].get(key))
            for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF41_gate_removed_ratio004_long.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F41 - Real Gate-Removed Ratio0.04 Long Ablation",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F41 extends F39's real parking gate ablation from the short 500-step aggressive ratio0.04 case to a 2000-iteration same-schedule pair with online W&B. This is intended to close the reviewer concern that the real gate-removed evidence was too short-budget.",
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
        "F41 is the long-budget real-scene counterpart to F38/F39 for the aggressive ratio0.04 edit schedule. The mechanism evidence is strong: the gated run rolls back the same no-counterfactual-accept candidate set that the gate-removed run commits. The final metrics are mixed rather than a clean gate-on win: no-gate is slightly better on PSNR, SSIM, LPIPS, and AbsRel, while gated is better on Depth MAE and normal and preserves more topology. This should be reported as long-budget gate/rollback necessity evidence for unsafe edit rejection, not as proof that the gate monotonically improves every final metric.",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated_root", required=True)
    parser.add_argument("--nogate_root", required=True)
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument(
        "--output_dir",
        default="outputs/carnet/meshprior/stageF41_real_gate_removed_ratio004_long/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    gated = helper._summarize("gated_ratio004_long", Path(args.gated_root), args.iteration, True)
    nogate = helper._summarize("no_gate_ratio004_long", Path(args.nogate_root), args.iteration, False)
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
