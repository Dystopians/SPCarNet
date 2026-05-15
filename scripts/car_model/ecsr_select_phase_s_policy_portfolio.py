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
    parser.add_argument(
        "--min_trainval_psnr_delta",
        type=float,
        default=-math.inf,
        help="Optional non-test effect-size gate on train-val PSNR delta.",
    )
    parser.add_argument(
        "--max_trainval_ssim_regression",
        type=float,
        default=math.inf,
        help="Optional non-test effect-size gate on train-val SSIM regression.",
    )
    parser.add_argument(
        "--max_trainval_lpips_regression",
        type=float,
        default=math.inf,
        help="Optional non-test effect-size gate on train-val LPIPS regression.",
    )
    parser.add_argument(
        "--min_trainval_effect_score",
        type=float,
        default=-math.inf,
        help=(
            "Optional non-test effect-size gate on train-val balanced score "
            "dPSNR + 20*dSSIM - 20*dLPIPS."
        ),
    )
    parser.add_argument(
        "--require_operator_audit",
        action="store_true",
        help="Require the selected candidate or selected nested trial to expose candidate_operator_audit.",
    )
    parser.add_argument(
        "--require_operator_policy_pass",
        action="store_true",
        help="Require candidate_operator_audit.policy_pass=true when an operator audit is available.",
    )
    parser.add_argument(
        "--reject_no_op_operator",
        action="store_true",
        help="Reject candidates whose operator audit reports no_op_copy=true.",
    )
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


def selected_trial_row(decision: dict[str, Any]) -> dict[str, Any]:
    selected = str(decision.get("selected_trial", ""))
    trials = decision.get("trials")
    if not selected or not isinstance(trials, list):
        return {}
    for row in trials:
        if not isinstance(row, dict):
            continue
        if str(row.get("trial", "")) == selected or str(row.get("selected_label", "")) == selected:
            return row
    return {}


def trainval_delta(decision: dict[str, Any], metric: str) -> float:
    deltas = decision.get("trainval_delta")
    if isinstance(deltas, dict) and metric in deltas:
        return number(deltas.get(metric), 0.0)
    selected = selected_trial_row(decision)
    deltas = selected.get("trainval_delta")
    if isinstance(deltas, dict) and metric in deltas:
        return number(deltas.get(metric), 0.0)
    return 0.0


def trainval_effect_score(deltas: dict[str, float]) -> float:
    return float(deltas.get("PSNR", 0.0)) + 20.0 * float(deltas.get("SSIM", 0.0)) - 20.0 * float(
        deltas.get("LPIPS", 0.0)
    )


def operator_audit(decision: dict[str, Any]) -> dict[str, Any]:
    audit = decision.get("candidate_operator_audit")
    if isinstance(audit, dict):
        return dict(audit)
    selected = selected_trial_row(decision)
    nested_path = selected.get("decision_path")
    if isinstance(nested_path, str) and nested_path.strip():
        nested = load_json(Path(nested_path))
        if isinstance(nested, dict):
            nested_audit = nested.get("candidate_operator_audit")
            if isinstance(nested_audit, dict):
                out = dict(nested_audit)
                out.setdefault("nested_decision_path", nested_path)
                return out
    return {}


def portfolio_rejection_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if not row["present"]:
        reasons.append("missing_decision_json")
    if not row["accepted"]:
        reasons.append("candidate_not_accepted")
    if not row["selection_uses_test_present"]:
        reasons.append("missing_selection_uses_test_field")
    if row["selection_uses_test"]:
        reasons.append("selection_uses_test_true")
    if float(row["trainval_balanced_delta"]) < float(args.min_trainval_balanced_delta):
        reasons.append("trainval_balanced_below_threshold")
    deltas = row.get("trainval_delta", {})
    if float(deltas.get("PSNR", 0.0)) < float(args.min_trainval_psnr_delta):
        reasons.append("trainval_psnr_delta_below_threshold")
    if float(deltas.get("SSIM", 0.0)) < -float(args.max_trainval_ssim_regression):
        reasons.append("trainval_ssim_regression_exceeds_threshold")
    if float(deltas.get("LPIPS", 0.0)) > float(args.max_trainval_lpips_regression):
        reasons.append("trainval_lpips_regression_exceeds_threshold")
    if float(row.get("trainval_effect_score", 0.0)) < float(args.min_trainval_effect_score):
        reasons.append("trainval_effect_score_below_threshold")
    audit = row.get("candidate_operator_audit", {})
    operator_gate_needs_audit = bool(args.require_operator_audit) or bool(args.require_operator_policy_pass) or bool(
        args.reject_no_op_operator
    )
    if operator_gate_needs_audit and not audit:
        reasons.append("missing_candidate_operator_audit")
    if audit:
        if bool(args.require_operator_policy_pass) and not bool(audit.get("policy_pass", False)):
            reasons.append("operator_policy_pass_not_true")
        if bool(args.reject_no_op_operator) and bool(audit.get("no_op_copy", False)):
            reasons.append("operator_no_op_copy_true")
    return reasons


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
    tv_delta = {metric: trainval_delta(decision, metric) for metric in METRICS}
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
        "trainval_delta": tv_delta,
        "trainval_effect_score": trainval_effect_score(tv_delta),
        "candidate_operator_audit": operator_audit(decision),
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
        "trainval_delta": {metric: 0.0 for metric in METRICS},
        "trainval_effect_score": 0.0,
        "candidate_operator_audit": {},
        "decision_reasons": ["missing_decision_json"],
        "test_delta_report_only": {metric: 0.0 for metric in METRICS},
        "test_balanced_delta_report_only": 0.0,
    }


def select_scene(
    scene: str,
    specs: list[tuple[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for label, template in specs:
        path = Path(template.format(scene=scene))
        decision = load_json(path)
        if decision is None:
            candidates.append(missing_candidate_row(scene, label, path))
        else:
            candidates.append(candidate_row(scene, label, path, decision))

    for row in candidates:
        row["portfolio_rejection_reasons"] = portfolio_rejection_reasons(row, args)
    eligible = [row for row in candidates if not row["portfolio_rejection_reasons"]]
    eligible.sort(
        key=lambda row: (
            float(row["trainval_effect_score"]),
            float(row["trainval_balanced_delta"]),
            float(row["trainval_delta"].get("PSNR", 0.0)),
            str(row["label"]),
        ),
        reverse=True,
    )
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
        "selected_trainval_delta": dict(selected["trainval_delta"]) if selected else {metric: 0.0 for metric in METRICS},
        "selected_trainval_effect_score": float(selected["trainval_effect_score"]) if selected else 0.0,
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
        f"- min train-val balanced delta: `{fmt(payload['thresholds']['min_trainval_balanced_delta'])}`",
        f"- min train-val PSNR delta: `{fmt(payload['thresholds']['min_trainval_psnr_delta'])}`",
        f"- max train-val SSIM regression: `{fmt(payload['thresholds']['max_trainval_ssim_regression'])}`",
        f"- max train-val LPIPS regression: `{fmt(payload['thresholds']['max_trainval_lpips_regression'])}`",
        f"- min train-val effect score: `{fmt(payload['thresholds']['min_trainval_effect_score'])}`",
        f"- require operator audit: `{payload['thresholds']['require_operator_audit']}`",
        f"- require operator policy pass: `{payload['thresholds']['require_operator_policy_pass']}`",
        f"- reject no-op operator: `{payload['thresholds']['reject_no_op_operator']}`",
        f"- mean effective report-only dPSNR: `{fmt(payload['mean_effective_test_delta_report_only']['PSNR'])}`",
        f"- mean effective report-only dSSIM: `{fmt(payload['mean_effective_test_delta_report_only']['SSIM'])}`",
        f"- mean effective report-only dLPIPS: `{fmt(payload['mean_effective_test_delta_report_only']['LPIPS'])}`",
        "",
        "| scene | selected policy | accepted | train-val balanced | train-val dPSNR | train-val dSSIM | train-val dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS | candidates | eligible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["effective_test_delta_report_only"]
        tv = row["selected_trainval_delta"]
        lines.append(
            "| {scene} | {selected} | {accepted} | {train} | {tv_psnr} | {tv_ssim} | {tv_lpips} | {psnr} | {ssim} | {lpips} | {candidates} | {eligible} |".format(
                scene=row["scene"],
                selected=row["selected_label"],
                accepted=str(bool(row["accepted"])).lower(),
                train=fmt(float(row["selected_trainval_balanced_delta"])),
                tv_psnr=fmt(float(tv["PSNR"])),
                tv_ssim=fmt(float(tv["SSIM"])),
                tv_lpips=fmt(float(tv["LPIPS"])),
                psnr=fmt(float(delta["PSNR"])),
                ssim=fmt(float(delta["SSIM"])),
                lpips=fmt(float(delta["LPIPS"])),
                candidates=int(row["candidate_count"]),
                eligible=int(row["eligible_count"]),
            )
        )
    lines.extend(["", "## Candidate Paths", ""])
    lines.append("| scene | candidate | present | accepted | train-val balanced | train-val dPSNR | audit pass | no-op | path | portfolio reasons | decision reasons |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        for candidate in row["candidates"]:
            reasons = candidate.get("decision_reasons", [])
            if isinstance(reasons, list):
                reason_text = ",".join(str(item) for item in reasons[:5])
            else:
                reason_text = str(reasons)
            portfolio_reasons = candidate.get("portfolio_rejection_reasons", [])
            if isinstance(portfolio_reasons, list):
                portfolio_reason_text = ",".join(str(item) for item in portfolio_reasons[:8])
            else:
                portfolio_reason_text = str(portfolio_reasons)
            audit = candidate.get("candidate_operator_audit", {})
            tv = candidate.get("trainval_delta", {})
            lines.append(
                "| {scene} | {label} | {present} | {accepted} | {train} | {tv_psnr} | {audit_pass} | {noop} | `{path}` | {portfolio_reasons} | {reasons} |".format(
                    scene=row["scene"],
                    label=candidate["label"],
                    present=str(bool(candidate["present"])).lower(),
                    accepted=str(bool(candidate["accepted"])).lower(),
                    train=fmt(float(candidate["trainval_balanced_delta"])),
                    tv_psnr=fmt(float(tv.get("PSNR", 0.0))),
                    audit_pass=str(bool(audit.get("policy_pass", False))) if audit else "n/a",
                    noop=str(bool(audit.get("no_op_copy", False))) if audit else "n/a",
                    path=candidate["path"],
                    portfolio_reasons=portfolio_reason_text,
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
                "selected_trainval_dPSNR",
                "selected_trainval_dSSIM",
                "selected_trainval_dLPIPS",
                "selected_trainval_effect_score",
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
            tv = row["selected_trainval_delta"]
            writer.writerow(
                {
                    "scene": row["scene"],
                    "selected_label": row["selected_label"],
                    "accepted": row["accepted"],
                    "selected_trainval_balanced_delta": row["selected_trainval_balanced_delta"],
                    "selected_trainval_dPSNR": tv["PSNR"],
                    "selected_trainval_dSSIM": tv["SSIM"],
                    "selected_trainval_dLPIPS": tv["LPIPS"],
                    "selected_trainval_effect_score": row["selected_trainval_effect_score"],
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
    rows = [select_scene(scene, specs, args) for scene in scenes]
    mean_delta = {
        metric: sum(float(row["effective_test_delta_report_only"][metric]) for row in rows) / max(len(rows), 1)
        for metric in METRICS
    }
    payload = {
        "selection_uses_test": False,
        "scene_count": int(len(rows)),
        "accepted_count": int(sum(1 for row in rows if row["accepted"])),
        "candidate_specs": [{"label": label, "path_template": template} for label, template in specs],
        "thresholds": {
            "min_trainval_balanced_delta": float(args.min_trainval_balanced_delta),
            "min_trainval_psnr_delta": float(args.min_trainval_psnr_delta),
            "max_trainval_ssim_regression": float(args.max_trainval_ssim_regression),
            "max_trainval_lpips_regression": float(args.max_trainval_lpips_regression),
            "min_trainval_effect_score": float(args.min_trainval_effect_score),
            "require_operator_audit": bool(args.require_operator_audit),
            "require_operator_policy_pass": bool(args.require_operator_policy_pass),
            "reject_no_op_operator": bool(args.reject_no_op_operator),
        },
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
