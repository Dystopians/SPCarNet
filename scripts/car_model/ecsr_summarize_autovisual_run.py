#!/usr/bin/env python3
"""Summarize AutoVisual PhaseK/filter/selector artifacts for one or more runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def fmt(value: Any, digits: int = 6) -> str:
    value = num(value)
    return "n/a" if not math.isfinite(value) else f"{value:+.{digits}f}"


def metric_block(payload: dict[str, Any] | None) -> dict[str, float]:
    payload = payload or {}
    return {key: num(payload.get(key)) for key in METRICS}


def first_value(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    value = next(iter(payload.values()))
    return value if isinstance(value, dict) else {}


def method_metrics(path: Path) -> dict[str, float]:
    return metric_block(first_value(read_json(path)))


def decision_summary(path: Path) -> dict[str, Any]:
    decision = read_json(path)
    if not decision:
        return {"present": False, "path": str(path)}
    render_gate = decision.get("render_region_gate") or {}
    tail = decision.get("trainval_per_view_tail") or {}
    return {
        "present": True,
        "path": str(path),
        "accepted": bool(decision.get("accepted", False)),
        "selected_label": decision.get("selected_label", ""),
        "selection_uses_test": bool(decision.get("selection_uses_test", False)),
        "decision_reasons": decision.get("decision_reasons", []),
        "test_delta_report_only": metric_block(decision.get("test_delta_report_only")),
        "trainval_delta": metric_block(decision.get("trainval_delta")),
        "trainval_balanced_delta": num(decision.get("trainval_balanced_delta")),
        "render_region_accepted": render_gate.get("accepted"),
        "render_region_tail": render_gate.get("tail") or {},
        "trainval_tail": {
            key: tail.get(key)
            for key in (
                "balanced_cvar_delta",
                "balanced_negative_fraction",
                "lpips_positive_fraction",
                "worst_lpips_regression",
            )
        },
    }


def selector_summary(path: Path) -> dict[str, Any]:
    decision = read_json(path)
    if not decision:
        return {"present": False, "path": str(path)}
    trials = decision.get("trials") if isinstance(decision.get("trials"), list) else []
    best = {}
    if trials:
        best = max(trials, key=lambda row: num(row.get("trainval_balanced_delta"), -math.inf))
    return {
        "present": True,
        "path": str(path),
        "accepted": bool(decision.get("accepted", False)),
        "selected_trial": decision.get("selected_trial", ""),
        "selected_label": decision.get("selected_label", ""),
        "candidate_count": decision.get("candidate_count"),
        "trial_count": len(trials),
        "selected_trainval_balanced_delta": num(decision.get("selected_trainval_balanced_delta")),
        "effective_report_only_test_delta": metric_block(decision.get("effective_report_only_test_delta")),
        "best_trial": {
            "name": best.get("trial", ""),
            "accepted": bool(best.get("accepted", False)) if best else False,
            "trainval_balanced_delta": num(best.get("trainval_balanced_delta")),
            "report_only_test_delta": metric_block(best.get("report_only_test_delta")),
        },
    }


def collect_run(root: Path) -> dict[str, Any]:
    scene_names: set[str] = set()
    if root.is_dir():
        for stage in ("plan_generation", "candidate_owned_refit"):
            stage_root = root / stage
            if not stage_root.is_dir():
                continue
            for path in stage_root.iterdir():
                if path.is_dir() and (path / "model").is_dir():
                    scene_names.add(path.name)
    scenes = sorted(scene_names)
    stages: dict[str, Any] = {}
    for stage in ("plan_generation", "candidate_owned_refit"):
        stage_root = root / stage
        stage_rows = {}
        for scene in scenes:
            scene_root = stage_root / scene
            decision_path = stage_root / "decisions" / f"{scene}_decision.json"
            stage_rows[scene] = {
                "base_metrics": method_metrics(scene_root / "model" / "results.json"),
                "test_metrics": method_metrics(scene_root / "model" / "test_results.json"),
                "trainval_metrics": method_metrics(scene_root / "model" / "trainval_gate_results.json"),
                "phasej_test_metrics": method_metrics(scene_root / "phasej_test_results.json"),
                "phasej_trainval_metrics": method_metrics(scene_root / "phasej_trainval_gate_results.json"),
                "decision": decision_summary(decision_path),
            }
        stages[stage] = stage_rows
    selector_rows = {}
    for scene in scenes:
        selector_rows[scene] = selector_summary(root / "selector" / scene / "coupled_selector_decision.json")
    return {"root": str(root), "scenes": scenes, "stages": stages, "selector": selector_rows}


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AutoVisual Run Summary",
        "",
        "All deltas are train-val/test deltas reported by each decision file. Held-out test deltas remain report-only.",
        "",
    ]
    for run in payload["runs"]:
        lines.extend([f"## `{run['root']}`", ""])
        for stage in ("plan_generation", "candidate_owned_refit"):
            rows = run["stages"].get(stage, {})
            if not rows:
                continue
            lines.extend(
                [
                    f"### {stage}",
                    "",
                    "| scene | decision | selected | reasons | test dPSNR | test dSSIM | test dLPIPS | trainval dPSNR | trainval dSSIM | trainval dLPIPS | trainval balanced | tail CVaR |",
                    "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for scene, row in rows.items():
                decision = row["decision"]
                if not decision["present"]:
                    lines.append(f"| {scene} | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                test = decision["test_delta_report_only"]
                train = decision["trainval_delta"]
                tail = decision["trainval_tail"]
                reasons = ", ".join(str(item) for item in decision["decision_reasons"]) or "none"
                lines.append(
                    f"| {scene} | {str(decision['accepted']).lower()} | {decision['selected_label']} | {reasons} | "
                    f"{fmt(test['PSNR'])} | {fmt(test['SSIM'])} | {fmt(test['LPIPS'])} | "
                    f"{fmt(train['PSNR'])} | {fmt(train['SSIM'])} | {fmt(train['LPIPS'])} | "
                    f"{fmt(decision['trainval_balanced_delta'])} | {fmt(tail.get('balanced_cvar_delta'))} |"
                )
            lines.append("")
        selector = run.get("selector", {})
        if selector:
            lines.extend(
                [
                    "### selector",
                    "",
                    "| scene | present | accepted | selected trial | candidates | trials | effective dPSNR | effective dSSIM | effective dLPIPS | selected trainval balanced | best trial | best trial trainval balanced |",
                    "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
                ]
            )
            for scene, row in selector.items():
                if not row["present"]:
                    lines.append(f"| {scene} | false | false | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                eff = row["effective_report_only_test_delta"]
                best = row["best_trial"]
                lines.append(
                    f"| {scene} | true | {str(row['accepted']).lower()} | {row['selected_trial'] or 'n/a'} | "
                    f"{row.get('candidate_count')} | {row.get('trial_count')} | "
                    f"{fmt(eff['PSNR'])} | {fmt(eff['SSIM'])} | {fmt(eff['LPIPS'])} | "
                    f"{fmt(row['selected_trainval_balanced_delta'])} | {best.get('name') or 'n/a'} | "
                    f"{fmt(best.get('trainval_balanced_delta'))} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = {"runs": [collect_run(root) for root in args.run_roots]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.output_md, payload)
    print(json.dumps({"runs": len(payload["runs"]), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
