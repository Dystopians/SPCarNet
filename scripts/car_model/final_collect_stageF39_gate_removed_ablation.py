#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/car_model/final_stageF39_real_gate_removed_ablation_report.md"


WANDB_RE = re.compile(r"https://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _wandb_url(run_root: Path) -> str:
    log = run_root / "logs/train.log"
    if not log.is_file():
        return ""
    match = WANDB_RE.search(log.read_text(encoding="utf-8", errors="replace"))
    return match.group(0) if match else ""


def _checkpoint_counts(model: Path, iteration: int) -> dict[str, int | None]:
    ckpt = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not ckpt.is_file():
        return {"triangles": None, "vertices": None}
    state = torch.load(ckpt, map_location="cpu")
    return {"triangles": int(state["_triangle_indices"].shape[0]), "vertices": int(state["triangles_points"].shape[0])}


def _candidate_rows(model: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((model / "prism_round_checkpoints").glob("*_candidate_meta.json")):
        payload = _load_json(path)
        payload["path"] = str(path)
        rows.append(payload)
    return rows


def _results(model: Path, iteration: int) -> dict[str, Any]:
    render = _load_json(model / "results.json").get(f"ours_{iteration}", {})
    geometry = _load_json(model / "geometry_eval_colmap" / f"iter_{iteration}.json")
    return {
        "psnr": render.get("PSNR"),
        "ssim": render.get("SSIM"),
        "lpips": render.get("LPIPS"),
        "absrel": (geometry.get("depth") or {}).get("abs_rel"),
        "depth_mae": (geometry.get("depth") or {}).get("mae"),
        "normal": (geometry.get("normal") or {}).get("mean_ang_deg"),
    }


def _summarize(label: str, run_root: Path, iteration: int, gate_enabled: bool) -> dict[str, Any]:
    model = run_root / "model"
    rows = _candidate_rows(model)
    effective = [r for r in rows if int(r.get("no_candidates", 0) or 0) == 0]
    committed = [r for r in effective if bool(r.get("committed"))]
    rollback = [r for r in effective if int(r.get("rollback", 0) or 0) > 0]
    selected = sum(int(r.get("candidate_selected_count", 0) or 0) for r in effective)
    committed_selected = sum(int(r.get("candidate_selected_count", 0) or 0) for r in committed)
    pre = next((int(r.get("pre_prune_triangle_count")) for r in effective if r.get("pre_prune_triangle_count") is not None), None)
    post_last = next(
        (int(r.get("post_prune_triangle_count")) for r in reversed(effective) if r.get("post_prune_triangle_count") is not None),
        None,
    )
    return {
        "label": label,
        "run_root": str(run_root),
        "model": str(model),
        "iteration": int(iteration),
        "gate_enabled": bool(gate_enabled),
        "wandb_url": _wandb_url(run_root),
        "checkpoint": _checkpoint_counts(model, iteration),
        "metrics": _results(model, iteration),
        "candidate_rounds": len(effective),
        "committed_rounds": len(committed),
        "rollback_rounds": len(rollback),
        "selected_candidates": selected,
        "committed_selected_candidates": committed_selected,
        "first_round": effective[0] if effective else {},
        "pre_prune_triangle_count": pre,
        "last_post_prune_triangle_count": post_last,
        "results_ready": (model / "results.json").is_file()
        and (model / "geometry_eval_colmap" / f"iter_{iteration}.json").is_file()
        and (model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt").is_file(),
    }


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
        "decision": "F39_COMPLETE" if gated["results_ready"] and nogate["results_ready"] else "F39_IN_PROGRESS",
        "gated": gated,
        "no_gate": nogate,
        "deltas_no_gate_minus_gated": {
            key: _delta(nogate["metrics"].get(key), gated["metrics"].get(key))
            for key in ("psnr", "ssim", "lpips", "absrel", "depth_mae", "normal")
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_stageF39_gate_removed_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Stage F39 - Real Gate-Removed Ablation",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        f"F39 runs a same-schedule parking PRISM ablation with the counterfactual gate enabled and removed. Both runs use online W&B and the same {gated['iteration']}-iteration integrated topology-control configuration.",
        "",
        "## Runs",
        "",
        "| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | selected candidates | committed selected | final triangles |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (gated, nogate):
        lines.append(
            "| {label} | `{gate}` | {wandb} | `{ready}` | {rounds} | {commits} | {selected} | {committed_selected} | {triangles} |".format(
                label=row["label"],
                gate=row["gate_enabled"],
                wandb=row["wandb_url"] or "NA",
                ready=row["results_ready"],
                rounds=row["candidate_rounds"],
                commits=row["committed_rounds"],
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
        "If complete, this is the real-scene counterpart to F38: the gate-removed run can commit a candidate set that has no counterfactual acceptance, while the gated control exposes whether the same schedule rejects or rolls back the edit. The row remains a medium-budget ablation, not a replacement for final long-budget compact-recovery results.",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated_root", required=True)
    parser.add_argument("--nogate_root", required=True)
    parser.add_argument("--iteration", type=int, default=2000)
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/stageF39_real_no_counterfactual_gate/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gated = _summarize("gated_control", Path(args.gated_root), args.iteration, True)
    nogate = _summarize("no_gate", Path(args.nogate_root), args.iteration, False)
    payload = write_report(ROOT / args.output_dir, gated, nogate)
    print(json.dumps({"decision": payload["decision"], "gated_ready": gated["results_ready"], "no_gate_ready": nogate["results_ready"]}, indent=2))


if __name__ == "__main__":
    main()
