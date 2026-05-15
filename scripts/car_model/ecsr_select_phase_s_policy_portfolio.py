#!/usr/bin/env python3
"""Select a fixed Phase-S portfolio using train-val decisions only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scene-level Phase-S portfolio policy from already-run candidate "
            "decision JSON files. Selection is based only on each candidate's train-val "
            "gate outcome and train-val balanced delta; held-out test deltas remain "
            "report-only after the policy choice is fixed."
        )
    )
    parser.add_argument("--scenes", required=True, help="Comma/space-separated scene names.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Candidate in label=path_template form. The template may contain {scene}. "
            "Example: georisk=outputs/..._{scene}/{scene}/coupled_selector_decision.json"
        ),
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--min_trainval_balanced_delta", type=float, default=0.0)
    return parser.parse_args()


def split_scenes(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_candidate_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--candidate must be label=path_template, got {raw!r}")
        label, template = raw.split("=", 1)
        label = label.strip()
        template = template.strip()
        if not label or not template:
            raise ValueError(f"empty label/template in candidate spec {raw!r}")
        specs.append((label, template))
    if not specs:
        raise ValueError("at least one --candidate is required")
    return specs


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def test_delta(decision: dict[str, Any], metric: str) -> float:
    effective = decision.get("effective_report_only_test_delta")
    if isinstance(effective, dict) and metric in effective:
        return number(effective.get(metric), 0.0)
    deltas = decision.get("test_delta_report_only")
    if isinstance(deltas, dict) and metric in deltas:
        return number(deltas.get(metric), 0.0)
    base = decision.get("base_test_metrics_report_only")
    cand = decision.get("candidate_test_metrics_report_only")
    if isinstance(base, dict) and isinstance(cand, dict):
        return number(cand.get(metric), 0.0) - number(base.get(metric), 0.0)
    return 0.0


def candidate_row(scene: str, label: str, path: Path, decision: dict[str, Any]) -> dict[str, Any]:
    trainval_balanced = number(
        decision.get("trainval_balanced_delta", decision.get("selected_trainval_balanced_delta")),
        -math.inf,
    )
    accepted = bool(decision.get("accepted", False))
    selection_flag_present = "selection_uses_test" in decision
    uses_test = bool(decision.get("selection_uses_test", True))
    selected_label = str(decision.get("selected_label", decision.get("selected_trial", "")))
    reasons = decision.get("decision_reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    if not selection_flag_present:
        reasons = list(reasons) + ["missing_selection_uses_test_field"]
    return {
        "scene": scene,
        "label": label,
        "path": str(path),
        "present": True,
        "accepted": accepted,
        "selection_uses_test_present": selection_flag_present,
        "selection_uses_test": uses_test,
        "selected_label": selected_label,
        "trainval_balanced_delta": trainval_balanced,
        "decision_reasons": reasons,
        "test_delta_report_only": {metric: test_delta(decision, metric) for metric in METRICS},
        "test_balanced_delta_report_only": number(decision.get("test_balanced_delta_report_only"), 0.0),
    }


def missing_candidate_row(scene: str, label: str, path: Path) -> dict[str, Any]:
    return {
        "scene": scene,
        "label": label,
        "path": str(path),
        "present": False,
        "accepted": False,
        "selection_uses_test_present": False,
        "selection_uses_test": False,
        "selected_label": "",
        "trainval_balanced_delta": -math.inf,
        "decision_reasons": ["missing_decision_json"],
        "test_delta_report_only": {metric: 0.0 for metric in METRICS},
        "test_balanced_delta_report_only": 0.0,
    }


def select_scene(
    scene: str,
    specs: list[tuple[str, str]],
    min_trainval_balanced_delta: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for label, template in specs:
        path = Path(template.format(scene=scene))
        decision = load_json(path)
        if decision is None:
            candidates.append(missing_candidate_row(scene, label, path))
        else:
            candidates.append(candidate_row(scene, label, path, decision))

    eligible = [
        row
        for row in candidates
        if row["present"]
        and row["accepted"]
        and row["selection_uses_test_present"]
        and not row["selection_uses_test"]
        and float(row["trainval_balanced_delta"]) >= float(min_trainval_balanced_delta)
    ]
    eligible.sort(key=lambda row: (float(row["trainval_balanced_delta"]), str(row["label"])), reverse=True)
    selected = eligible[0] if eligible else None
    effective_delta = {metric: 0.0 for metric in METRICS}
    effective_balanced = 0.0
    if selected is not None:
        effective_delta = dict(selected["test_delta_report_only"])
        effective_balanced = float(selected["test_balanced_delta_report_only"])

    return {
        "scene": scene,
        "selection_uses_test": False,
        "selected_label": selected["label"] if selected else "phasej_fallback",
        "selected_candidate_label": selected["selected_label"] if selected else "phasej_guarded_adaptedge",
        "accepted": selected is not None,
        "selected_trainval_balanced_delta": float(selected["trainval_balanced_delta"]) if selected else 0.0,
        "effective_test_delta_report_only": effective_delta,
        "effective_test_balanced_delta_report_only": effective_balanced,
        "candidate_count": len([row for row in candidates if row["present"]]),
        "eligible_count": len(eligible),
        "candidates": candidates,
    }


def fmt(value: float) -> str:
    return f"{value:+.9f}" if math.isfinite(value) else "n/a"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Phase-S Train-Val Portfolio Policy",
        "",
        "Selection uses candidate train-val decisions only. Held-out test deltas are report-only after selection; rejected or missing scenes fall back to Phase-J with zero effective delta.",
        "A candidate must explicitly set `selection_uses_test=false`; missing selection provenance is treated as ineligible.",
        "",
        f"- scenes: `{payload['scene_count']}`",
        f"- accepted scenes: `{payload['accepted_count']}`",
        f"- mean effective report-only dPSNR: `{fmt(payload['mean_effective_test_delta_report_only']['PSNR'])}`",
        f"- mean effective report-only dSSIM: `{fmt(payload['mean_effective_test_delta_report_only']['SSIM'])}`",
        f"- mean effective report-only dLPIPS: `{fmt(payload['mean_effective_test_delta_report_only']['LPIPS'])}`",
        "",
        "| scene | selected policy | accepted | train-val balanced | effective dPSNR | effective dSSIM | effective dLPIPS | candidates | eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["effective_test_delta_report_only"]
        lines.append(
            "| {scene} | {selected} | {accepted} | {train} | {psnr} | {ssim} | {lpips} | {candidates} | {eligible} |".format(
                scene=row["scene"],
                selected=row["selected_label"],
                accepted=str(bool(row["accepted"])).lower(),
                train=fmt(float(row["selected_trainval_balanced_delta"])),
                psnr=fmt(float(delta["PSNR"])),
                ssim=fmt(float(delta["SSIM"])),
                lpips=fmt(float(delta["LPIPS"])),
                candidates=int(row["candidate_count"]),
                eligible=int(row["eligible_count"]),
            )
        )
    lines.extend(["", "## Candidate Paths", ""])
    lines.append("| scene | candidate | present | accepted | train-val balanced | path | reasons |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for row in rows:
        for candidate in row["candidates"]:
            reasons = candidate.get("decision_reasons", [])
            if isinstance(reasons, list):
                reason_text = ",".join(str(item) for item in reasons[:5])
            else:
                reason_text = str(reasons)
            lines.append(
                "| {scene} | {label} | {present} | {accepted} | {train} | `{path}` | {reasons} |".format(
                    scene=row["scene"],
                    label=candidate["label"],
                    present=str(bool(candidate["present"])).lower(),
                    accepted=str(bool(candidate["accepted"])).lower(),
                    train=fmt(float(candidate["trainval_balanced_delta"])),
                    path=candidate["path"],
                    reasons=reason_text,
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "selected_label",
                "accepted",
                "selected_trainval_balanced_delta",
                "effective_dPSNR",
                "effective_dSSIM",
                "effective_dLPIPS",
                "candidate_count",
                "eligible_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            delta = row["effective_test_delta_report_only"]
            writer.writerow(
                {
                    "scene": row["scene"],
                    "selected_label": row["selected_label"],
                    "accepted": row["accepted"],
                    "selected_trainval_balanced_delta": row["selected_trainval_balanced_delta"],
                    "effective_dPSNR": delta["PSNR"],
                    "effective_dSSIM": delta["SSIM"],
                    "effective_dLPIPS": delta["LPIPS"],
                    "candidate_count": row["candidate_count"],
                    "eligible_count": row["eligible_count"],
                }
            )


def main() -> int:
    args = parse_args()
    scenes = split_scenes(args.scenes)
    specs = parse_candidate_specs(args.candidate)
    rows = [select_scene(scene, specs, float(args.min_trainval_balanced_delta)) for scene in scenes]
    mean_delta = {
        metric: sum(float(row["effective_test_delta_report_only"][metric]) for row in rows) / max(len(rows), 1)
        for metric in METRICS
    }
    payload = {
        "selection_uses_test": False,
        "scene_count": int(len(rows)),
        "accepted_count": int(sum(1 for row in rows if row["accepted"])),
        "candidate_specs": [{"label": label, "path_template": template} for label, template in specs],
        "min_trainval_balanced_delta": float(args.min_trainval_balanced_delta),
        "mean_effective_test_delta_report_only": mean_delta,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    print(json.dumps({"accepted_count": payload["accepted_count"], "scene_count": payload["scene_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
