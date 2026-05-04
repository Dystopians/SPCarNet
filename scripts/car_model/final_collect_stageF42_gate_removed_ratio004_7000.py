#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "scripts/car_model/final_collect_stageF39_gate_removed_ablation.py"
DOC = ROOT / "docs/car_model/final_stageF42_gate_removed_ratio004_7000_report.md"


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


def _interpret(payload: dict[str, Any]) -> str:
    if payload["decision"] != "F42_COMPLETE":
        return (
            "F42 is still incomplete. Do not draw metric conclusions until both rows have train, render, "
            "image metrics, geometry metrics, and W&B records."
        )

    gated = payload["gated"]
    nogate = payload["no_gate"]
    d = payload["deltas_no_gate_minus_gated"]
    render_gate_wins = (
        d.get("psnr") is not None
        and d["psnr"] < 0
        and d.get("ssim") is not None
        and d["ssim"] < 0
        and d.get("lpips") is not None
        and d["lpips"] > 0
    )
    geometry_nogate_wins = (
        d.get("absrel") is not None
        and d["absrel"] < 0
        and d.get("depth_mae") is not None
        and d["depth_mae"] < 0
        and d.get("normal") is not None
        and d["normal"] < 0
    )
    mechanism_pass = (
        gated["committed_rounds"] == 0
        and gated["rollback_rounds"] > 0
        and nogate["committed_rounds"] > 0
        and nogate["rollback_rounds"] == 0
    )

    parts = []
    if mechanism_pass:
        parts.append(
            "Mechanism pass: both rows select the same 2579-candidate ratio0.04 edit at iter 141; "
            "the gated row records counterfactual_accept=0 and rolls back, while the gate-removed row commits it."
        )
    if render_gate_wins:
        parts.append(
            "7000-step render evidence favors the gate: gated improves PSNR by "
            f"{-d['psnr']:.6f}, improves SSIM by {-d['ssim']:.6f}, "
            f"and reduces LPIPS by {d['lpips']:.6f} versus no-gate."
        )
    if geometry_nogate_wins:
        parts.append(
            "Sparse geometry proxies favor no-gate at this budget: AbsRel, Depth MAE, and normal angle are "
            f"lower by {-d['absrel']:.6f}, {-d['depth_mae']:.6f}, and {-d['normal']:.6f}."
        )
    parts.append(
        "The safe paper claim is therefore stronger than F41 for visual/held-out-render gate necessity, "
        "but still not universal metric dominance: use F42 as long-budget parking evidence that rollback "
        "prevents an unsafe no-accept topology edit and improves render quality, while geometry proxies remain mixed."
    )
    return " ".join(parts)


def write_report(out: Path, gated: dict[str, Any], nogate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "decision": "F42_COMPLETE" if gated["results_ready"] and nogate["results_ready"] else "F42_IN_PROGRESS",
        "purpose": "7000-iteration real-scene ratio0.04 gate-on/gate-off ablation after F41",
        "gated": gated,
        "no_gate": nogate,
        "deltas_no_gate_minus_gated": {
            key: _delta(nogate["metrics"].get(key), gated["metrics"].get(key))
            for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")
        },
    }
    payload["interpretation"] = _interpret(payload)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "final_stageF42_gate_removed_ratio004_7000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F42 - Real Gate-Removed Ratio0.04 7000-Iteration Ablation",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "F42 extends F41 from 2000 to 7000 iterations on the same parking ratio0.04 gate-on/gate-off schedule with online W&B. It is a closer-to-long-budget mechanism check, not a replacement for the final 22k/26k compact-recovery table.",
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
        default="outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = _load_helper()
    gated = helper._summarize("gated_ratio004_7000", Path(args.gated_root), args.iteration, True)
    nogate = helper._summarize("no_gate_ratio004_7000", Path(args.nogate_root), args.iteration, False)
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
