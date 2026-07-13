#!/usr/bin/env python3
"""Audit view-tail alpha shrink diagnostics in an AutoVisual/PhaseK run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _summarize_ela_report(path: Path, run_root: Path) -> dict[str, Any]:
    report = _load_json(path) or {}
    calibrator = report.get("alpha_calibrator")
    if not isinstance(calibrator, dict):
        calibrator = {}

    candidate_stats = calibrator.get("view_tail_candidate_stats")
    if not isinstance(candidate_stats, list):
        candidate_stats = []

    scale = _as_float(calibrator.get("view_tail_scale"))
    enabled = bool(calibrator.get("view_tail_enabled", False))
    safe_found = calibrator.get("view_tail_safe_scale_found")
    fallback_used = calibrator.get("view_tail_fallback_used")
    selected_candidate: dict[str, Any] = {}
    if scale is not None:
        for candidate in candidate_stats:
            if not isinstance(candidate, dict):
                continue
            candidate_scale = _as_float(candidate.get("scale"))
            if candidate_scale is not None and abs(candidate_scale - scale) <= 1.0e-9:
                selected_candidate = candidate
                break

    return {
        "path": str(path),
        "relative_path": _relative_or_absolute(path, run_root),
        "split": report.get("split"),
        "method_name": report.get("method_name"),
        "base_method_name": report.get("base_method_name"),
        "alpha_policy": report.get("alpha_policy"),
        "alpha_view_tail_scale_grid": report.get("alpha_view_tail_scale_grid"),
        "alpha_view_tail_cvar_fraction": report.get("alpha_view_tail_cvar_fraction"),
        "alpha_view_tail_min_gain": report.get("alpha_view_tail_min_gain"),
        "alpha_view_tail_max_negative_fraction": report.get("alpha_view_tail_max_negative_fraction"),
        "view_tail_objective": calibrator.get("view_tail_objective"),
        "view_tail_compute_lpips": calibrator.get("view_tail_compute_lpips"),
        "view_tail_ssim_weight": calibrator.get("view_tail_ssim_weight"),
        "view_tail_lpips_weight": calibrator.get("view_tail_lpips_weight"),
        "view_tail_metric_max_side": calibrator.get("view_tail_metric_max_side"),
        "view_tail_enabled": enabled,
        "view_tail_scale": scale,
        "view_tail_scale_lt_1": enabled and scale is not None and scale < 1.0,
        "view_tail_scale_eq_1": enabled and scale == 1.0,
        "view_tail_safe_scale_found": safe_found,
        "view_tail_fallback_used": fallback_used,
        "view_tail_candidate_count": len(candidate_stats),
        "view_tail_cvar_gain": calibrator.get("view_tail_cvar_gain"),
        "view_tail_mean_gain": calibrator.get("view_tail_mean_gain"),
        "view_tail_negative_fraction": calibrator.get("view_tail_negative_fraction"),
        "selected_mean_score": selected_candidate.get("mean_score"),
        "selected_mean_psnr_gain": selected_candidate.get("mean_psnr_gain"),
        "selected_mean_ssim_gain": selected_candidate.get("mean_ssim_gain"),
        "selected_mean_lpips_gain": selected_candidate.get("mean_lpips_gain"),
        "selected_mean_mse_gain": selected_candidate.get("mean_mse_gain"),
        "selected_lpips_regression_fraction": selected_candidate.get("lpips_regression_fraction"),
        "local_trust_mode": report.get("local_trust_mode"),
        "local_trust_mean_weight": report.get("local_trust_mean_weight"),
        "local_trust_active_fraction": report.get("local_trust_active_fraction"),
    }


def _summarize_decision(path: Path, run_root: Path) -> dict[str, Any]:
    decision = _load_json(path) or {}
    return {
        "path": str(path),
        "relative_path": _relative_or_absolute(path, run_root),
        "accepted": decision.get("accepted"),
        "selected_label": decision.get("selected_label"),
        "selected_trial": decision.get("selected_trial"),
        "decision_reasons": decision.get("decision_reasons"),
        "selection_uses_test": decision.get("selection_uses_test"),
        "test_delta_report_only": decision.get("test_delta_report_only"),
        "trainval_delta": decision.get("trainval_delta"),
        "trainval_balanced_delta": decision.get("trainval_balanced_delta"),
        "trainval_tail": decision.get("trainval_tail"),
        "render_region_accepted": decision.get("render_region_accepted"),
        "render_region_tail": decision.get("render_region_tail"),
    }


def build_summary(run_root: Path, extra_ela_reports: list[Path] | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    report_paths = {path.resolve() for path in run_root.rglob("ela_report.json")}
    for path in extra_ela_reports or []:
        if path.exists():
            report_paths.add(path.resolve())

    ela_reports = [_summarize_ela_report(path, run_root) for path in sorted(report_paths)]
    decisions = [
        _summarize_decision(path, run_root)
        for path in sorted(run_root.rglob("*decision.json"))
    ]

    enabled_reports = [row for row in ela_reports if row["view_tail_enabled"]]
    shrink_reports = [row for row in enabled_reports if row["view_tail_scale_lt_1"]]
    scale_one_reports = [row for row in enabled_reports if row["view_tail_scale_eq_1"]]
    fallback_reports = [row for row in enabled_reports if row["view_tail_fallback_used"] is True]
    missing_stats = [
        row
        for row in enabled_reports
        if row["view_tail_candidate_count"] == 0
    ]

    return {
        "run_root": str(run_root),
        "ela_report_count": len(ela_reports),
        "view_tail_enabled_count": len(enabled_reports),
        "view_tail_shrink_count": len(shrink_reports),
        "view_tail_scale_one_count": len(scale_one_reports),
        "view_tail_fallback_count": len(fallback_reports),
        "view_tail_enabled_missing_stats_count": len(missing_stats),
        "decision_count": len(decisions),
        "accepted_decision_count": sum(1 for row in decisions if row.get("accepted") is True),
        "ela_reports": ela_reports,
        "decisions": decisions,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# View-Tail Alpha Run Audit")
    lines.append("")
    lines.append(f"Run root: `{summary['run_root']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- ELA reports: `{summary['ela_report_count']}`")
    lines.append(f"- View-tail enabled reports: `{summary['view_tail_enabled_count']}`")
    lines.append(f"- Actual shrink reports (`scale < 1`): `{summary['view_tail_shrink_count']}`")
    lines.append(f"- Scale-one reports (`scale == 1`): `{summary['view_tail_scale_one_count']}`")
    lines.append(f"- Fallback-used reports: `{summary['view_tail_fallback_count']}`")
    lines.append(
        f"- Enabled reports missing candidate stats: `{summary['view_tail_enabled_missing_stats_count']}`"
    )
    lines.append(f"- Decisions: `{summary['decision_count']}`")
    lines.append(f"- Accepted decisions: `{summary['accepted_decision_count']}`")
    lines.append("")

    lines.append("## ELA Reports")
    lines.append("")
    lines.append(
        "| relative path | split | method | alpha policy | objective | LPIPS | enabled | scale | safe | fallback | candidates | cvar gain | neg frac | mean score | dPSNR | dSSIM | dLPIPS | trust mode |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary["ela_reports"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['relative_path']}`",
                    _fmt(row.get("split")),
                    f"`{_fmt(row.get('method_name'))}`",
                    _fmt(row.get("alpha_policy")),
                    _fmt(row.get("view_tail_objective")),
                    _fmt(row.get("view_tail_compute_lpips")),
                    _fmt(row.get("view_tail_enabled")),
                    _fmt(row.get("view_tail_scale")),
                    _fmt(row.get("view_tail_safe_scale_found")),
                    _fmt(row.get("view_tail_fallback_used")),
                    _fmt(row.get("view_tail_candidate_count")),
                    _fmt(row.get("view_tail_cvar_gain")),
                    _fmt(row.get("view_tail_negative_fraction")),
                    _fmt(row.get("selected_mean_score")),
                    _fmt(row.get("selected_mean_psnr_gain")),
                    _fmt(row.get("selected_mean_ssim_gain")),
                    _fmt(row.get("selected_mean_lpips_gain")),
                    _fmt(row.get("local_trust_mode")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Decisions")
    lines.append("")
    lines.append(
        "| relative path | accepted | selected | uses test | trainval balanced | reasons |"
    )
    lines.append("|---|---:|---|---:|---:|---|")
    for row in summary["decisions"]:
        reasons = row.get("decision_reasons")
        if isinstance(reasons, list):
            reasons_text = ", ".join(str(item) for item in reasons)
        else:
            reasons_text = _fmt(reasons)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['relative_path']}`",
                    _fmt(row.get("accepted")),
                    _fmt(row.get("selected_label") or row.get("selected_trial")),
                    _fmt(row.get("selection_uses_test")),
                    _fmt(row.get("trainval_balanced_delta")),
                    reasons_text,
                ]
            )
            + " |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument(
        "--extra_ela_report",
        action="append",
        default=[],
        type=Path,
        help="Additional ela_report.json path outside run_root. May be repeated.",
    )
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_md", required=True, type=Path)
    args = parser.parse_args()

    summary = build_summary(args.run_root, extra_ela_reports=args.extra_ela_report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.output_md)
    print(
        json.dumps(
            {
                "run_root": summary["run_root"],
                "ela_report_count": summary["ela_report_count"],
                "view_tail_enabled_count": summary["view_tail_enabled_count"],
                "view_tail_shrink_count": summary["view_tail_shrink_count"],
                "decision_count": summary["decision_count"],
                "accepted_decision_count": summary["accepted_decision_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
