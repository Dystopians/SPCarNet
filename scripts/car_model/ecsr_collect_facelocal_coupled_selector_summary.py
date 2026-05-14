#!/usr/bin/env python3
"""Collect per-scene Phase-S face-local coupled selector decisions."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary.csv"),
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


def find_trial(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("trial") == name:
            return row
    return {}


def trial_path(row: dict[str, Any]) -> str:
    path = str(row.get("decision_path", "") or "")
    return rel(path) if path else ""


def joined_trial_paths(rows: list[dict[str, Any]], skip_trial: str = "") -> str:
    paths = [
        trial_path(row)
        for row in rows
        if row.get("present") and row.get("trial") != skip_trial and trial_path(row)
    ]
    return ";".join(paths)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in parse_scenes(args.scenes):
        path = Path(args.decision_path_template.format(scene=scene))
        decision = read_json(path)
        trials = decision.get("trials") if isinstance(decision.get("trials"), list) else []
        best = best_trial(trials)
        selected_trial = decision.get("selected_trial", "") if decision else ""
        selected = find_trial(trials, selected_trial)
        top1 = find_trial(trials, "top1_s2")
        selected_path = trial_path(selected)
        current_path = rel(path) if decision else ""
        baseline_path = trial_path(top1) or selected_path
        improved_path = selected_path if bool(decision.get("accepted", False)) else ""
        rows.append(
            {
                "scene": scene,
                "decision_path": rel(path),
                "baseline_path": baseline_path,
                "current_path": current_path,
                "improved_path": improved_path,
                "ablation_paths": joined_trial_paths(trials, selected_trial),
                "plan_path": rel(decision.get("plan_path", "")) if decision and decision.get("plan_path") else "",
                "present": bool(decision),
                "candidate_count": int(decision.get("candidate_count", 0)) if decision else 0,
                "accepted": bool(decision.get("accepted", False)),
                "selected_trial": selected_trial,
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


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    eff = row["effective_report_only_test_delta"]
    best_test = row["best_trial_test_delta"]
    return {
        "scene": row["scene"],
        "present": row["present"],
        "candidate_count": row["candidate_count"],
        "accepted": row["accepted"],
        "selected_trial": row["selected_trial"],
        "selected_trainval_balanced_delta": row["selected_trainval_balanced_delta"],
        "effective_report_only_test_dPSNR": eff["PSNR"],
        "effective_report_only_test_dSSIM": eff["SSIM"],
        "effective_report_only_test_dLPIPS": eff["LPIPS"],
        "trial_count": row["trial_count"],
        "best_trial_by_trainval": row["best_trial_by_trainval"],
        "best_trial_trainval_balanced_delta": row["best_trial_trainval_balanced_delta"],
        "best_trial_accepted": row["best_trial_accepted"],
        "best_trial_test_dPSNR": best_test["PSNR"],
        "best_trial_test_dSSIM": best_test["SSIM"],
        "best_trial_test_dLPIPS": best_test["LPIPS"],
        "plan_path": row["plan_path"],
        "baseline_path": row["baseline_path"],
        "current_path": row["current_path"],
        "improved_path": row["improved_path"],
        "ablation_paths": row["ablation_paths"],
        "decision_path": row["decision_path"],
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [flatten_row(row) for row in payload["rows"]]
    fieldnames = list(rows[0].keys()) if rows else [
        "scene",
        "present",
        "candidate_count",
        "accepted",
        "selected_trial",
        "decision_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    lines.extend(
        [
            "",
            "## Role Paths",
            "",
            "`baseline_path` points to the top1/scale2 trial decision when present, because that row carries the Phase-J baseline metrics used by the coupled decision. `current_path` is the coupled selector decision JSON. `improved_path` is the selected promoted trial decision when the scene is accepted. `ablation_paths` are present non-selected trial decision JSONs.",
            "",
            "| scene | plan | baseline | current | improved | ablations |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['scene']} | {row['plan_path'] or 'n/a'} | {row['baseline_path'] or 'n/a'} | "
            f"{row['current_path'] or 'n/a'} | {row['improved_path'] or 'n/a'} | "
            f"{row['ablation_paths'] or 'n/a'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.output_md, payload)
    write_csv(args.output_csv, payload)
    print(
        json.dumps(
            {
                "present": payload["present_scene_count"],
                "accepted": payload["accepted_count"],
                "output_md": str(args.output_md),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
