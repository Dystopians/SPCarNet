#!/usr/bin/env python3
"""Collect per-scene Phase-S face-local coupled selector decisions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_TEMPLATE = (
    "outputs/carnet/meshsplatopt/ecsr_phase_s/"
    "facelocal_coupled_selector_v1_pilot_20260513_{scene}/{scene}/coupled_selector_decision.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="bicycle,flowers,garden,treehill,counter")
    parser.add_argument("--decision_path_template", default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary.json"),
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary.md"),
    )
    return parser.parse_args()


def parse_scenes(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any, digits: int = 9) -> str:
    v = num(value)
    return "n/a" if not math.isfinite(v) else f"{v:+.{digits}f}"


def metric_block(payload: dict[str, Any] | None) -> dict[str, float]:
    payload = payload or {}
    return {key: num(payload.get(key)) for key in METRICS}


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def best_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("present")]
    if not valid:
        return {}
    return max(valid, key=lambda row: num(row.get("trainval_balanced_delta"), -math.inf))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in parse_scenes(args.scenes):
        path = Path(args.decision_path_template.format(scene=scene))
        decision = read_json(path)
        trials = decision.get("trials") if isinstance(decision.get("trials"), list) else []
        best = best_trial(trials)
        rows.append(
            {
                "scene": scene,
                "decision_path": rel(path),
                "present": bool(decision),
                "candidate_count": int(decision.get("candidate_count", 0)) if decision else 0,
                "accepted": bool(decision.get("accepted", False)),
                "selected_trial": decision.get("selected_trial", "") if decision else "",
                "selected_trainval_balanced_delta": num(decision.get("selected_trainval_balanced_delta")) if decision else math.nan,
                "effective_report_only_test_delta": metric_block(decision.get("effective_report_only_test_delta") if decision else {}),
                "trial_count": len(trials),
                "best_trial_by_trainval": best.get("trial", ""),
                "best_trial_trainval_balanced_delta": num(best.get("trainval_balanced_delta")),
                "best_trial_accepted": bool(best.get("accepted", False)) if best else False,
                "best_trial_test_delta": metric_block(best.get("report_only_test_delta") if best else {}),
            }
        )
    present = [row for row in rows if row["present"] and row["candidate_count"] > 0]
    accepted = [row for row in present if row["accepted"]]
    payload = {
        "requested_scene_count": len(rows),
        "present_scene_count": len([row for row in rows if row["present"]]),
        "present_candidate_scene_count": len(present),
        "accepted_count": len(accepted),
        "mean_effective_report_only_test_delta": {
            key: mean([row["effective_report_only_test_delta"][key] for row in present])
            for key in METRICS
        },
        "rows": rows,
    }
    return payload


def write_md(path: Path, payload: dict[str, Any]) -> None:
    eff = payload["mean_effective_report_only_test_delta"]
    lines = [
        "# Phase-S Face-Local Coupled Selector Collected Summary",
        "",
        "Selection uses train-val render metrics only. Held-out test deltas are report-only; rejected scenes fall back to Phase-J with zero effective test delta.",
        "",
        f"- requested scenes: `{payload['requested_scene_count']}`",
        f"- present scenes: `{payload['present_scene_count']}`",
        f"- present candidate scenes: `{payload['present_candidate_scene_count']}`",
        f"- accepted scenes: `{payload['accepted_count']}`",
        f"- mean effective report-only dPSNR: `{fmt(eff['PSNR'])}`",
        f"- mean effective report-only dSSIM: `{fmt(eff['SSIM'])}`",
        f"- mean effective report-only dLPIPS: `{fmt(eff['LPIPS'])}`",
        "",
        "| scene | present | candidates | trials | selected | accepted | selected train-val balanced | effective dPSNR | effective dSSIM | effective dLPIPS | best trial | best trial train-val balanced | best trial test dPSNR | best trial test dSSIM | best trial test dLPIPS |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        eff = row["effective_report_only_test_delta"]
        best_test = row["best_trial_test_delta"]
        lines.append(
            f"| {row['scene']} | {str(row['present']).lower()} | {row['candidate_count']} | {row['trial_count']} | "
            f"{row['selected_trial'] or 'n/a'} | {str(row['accepted']).lower()} | {fmt(row['selected_trainval_balanced_delta'])} | "
            f"{fmt(eff['PSNR'])} | {fmt(eff['SSIM'])} | {fmt(eff['LPIPS'])} | "
            f"{row['best_trial_by_trainval'] or 'n/a'} | {fmt(row['best_trial_trainval_balanced_delta'])} | "
            f"{fmt(best_test['PSNR'])} | {fmt(best_test['SSIM'])} | {fmt(best_test['LPIPS'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.output_md, payload)
    print(json.dumps({"present": payload["present_scene_count"], "accepted": payload["accepted_count"], "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
