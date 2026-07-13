#!/usr/bin/env python3
"""Summarize a single v78 target-support / target-footprint certificate run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COUNTER_REFERENCES: dict[str, dict[str, float]] = {
    "v64_v56_counter_reference": {
        "PSNR": 26.756130219,
        "SSIM": 0.862126231,
        "LPIPS": 0.251691371,
    },
    "v75_zero_blend_local_patch": {
        "PSNR": 26.753995895,
        "SSIM": 0.862119257,
        "LPIPS": 0.251853049,
    },
    "v76_policyval_bin_gain_hybrid": {
        "PSNR": 26.753532410,
        "SSIM": 0.862111092,
        "LPIPS": 0.251881331,
    },
    "v77_strict_bin_gain_hybrid": {
        "PSNR": 26.753528595,
        "SSIM": 0.862111032,
        "LPIPS": 0.251881331,
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_int(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload or payload.get(key) is None:
        return None
    return int(payload[key])


def optional_max(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return int(max(present))


def md_value(value: Any, *, precision: int | None = None) -> str:
    if value is None:
        return "`not recorded`"
    if isinstance(value, float) and precision is not None:
        return f"`{value:.{precision}f}`"
    return f"`{value}`"


def first_method_metrics(results: dict[str, Any]) -> tuple[str, dict[str, float]]:
    if len(results) != 1:
        raise ValueError(f"expected one method in results.json, found {len(results)}")
    method, payload = next(iter(results.items()))
    return method, {
        "PSNR": float(payload.get("PSNR", 0.0)),
        "SSIM": float(payload.get("SSIM", 0.0)),
        "LPIPS": float(payload.get("LPIPS", 0.0)),
    }


def metric_delta(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "dPSNR": float(metrics["PSNR"] - reference["PSNR"]),
        "dSSIM": float(metrics["SSIM"] - reference["SSIM"]),
        "dLPIPS": float(metrics["LPIPS"] - reference["LPIPS"]),
    }


def strict_win(delta: dict[str, float]) -> bool:
    return bool(delta["dPSNR"] > 0.0 and delta["dSSIM"] > 0.0 and delta["dLPIPS"] < 0.0)


def infer_target_support_certificate(
    profile: dict[str, Any],
    *,
    enabled: bool,
    min_changed: float = 1.0e-12,
) -> dict[str, Any]:
    changed = float(profile.get("changed_fraction", 0.0) or 0.0)
    valid = float(profile.get("valid_fraction", profile.get("target_valid_fraction", 0.0)) or 0.0)
    reasons: list[str] = []
    if not enabled:
        reasons.append("target_support_candidate_selection_not_enabled")
    if changed < float(min_changed):
        reasons.append(f"changed_fraction {changed:.8f} < threshold {float(min_changed):.8f}")
    if valid <= 0.0:
        reasons.append("valid_fraction <= 0")
    return {
        "enabled": bool(enabled),
        "passed": bool(enabled and not reasons),
        "changed_fraction": float(changed),
        "changed_fraction_threshold": float(min_changed),
        "valid_fraction": float(valid),
        "reasons": reasons,
        "inferred_from_profile": True,
    }


def compact_score_order(score_order: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in score_order[:limit]:
        target_score = row.get("target_support_score") or {}
        if not target_score:
            target_score = {
                "changed_fraction": row.get("target_changed_fraction", 0.0),
                "valid_fraction": row.get("target_valid_fraction", 0.0),
            }
        target_cert = row.get("target_support_certificate") or infer_target_support_certificate(
            target_score,
            enabled=bool(row.get("target_support_enabled", True)),
        )
        out.append(
            {
                "blend": float(row.get("surface_multiscale_prior_blend", 0.0) or 0.0),
                "hybrid": bool(row.get("policy_val_prior_bin_gain_hybrid", False)),
                "allowed_bins": int(row.get("prior_bin_gain_hybrid_allowed_bins", 0) or 0),
                "accepted": bool(row.get("accepted", False)),
                "alpha": float(row.get("selected_alpha", 0.0) or 0.0),
                "relative_gain": float(row.get("relative_gain", 0.0) or 0.0),
                "ssim_gain": float(row.get("ssim_gain", 0.0) or 0.0),
                "image_l1_gain": float(row.get("image_l1_gain", 0.0) or 0.0),
                "target_changed_fraction": float(
                    row.get("target_changed_fraction", 0.0) or 0.0
                ),
                "target_cvar20_changed_fraction": float(
                    row.get("target_cvar20_view_changed_fraction", 0.0) or 0.0
                ),
                "target_min_changed_fraction": float(
                    row.get("target_min_view_changed_fraction", 0.0) or 0.0
                ),
                "target_certificate_passed": bool(target_cert.get("passed", False)),
                "target_certificate_inferred": bool(
                    target_cert.get("inferred_from_profile", False)
                ),
                "target_valid_fraction": float(target_score.get("valid_fraction", 0.0) or 0.0),
            }
        )
    return out


def candidate_bin_gain_footprint_summary(audit: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in audit.get("fill_mode_candidates") or []:
        fit_summary = candidate.get("fit_summary") or {}
        bin_gain = fit_summary.get("policy_val_prior_bin_gain_hybrid") or {}
        if not isinstance(bin_gain, dict) or not bin_gain:
            continue
        footprint = bin_gain.get("target_footprint_bin_certificate") or {}
        rows.append(
            {
                "fill_mode": str(candidate.get("fill_mode", "")),
                "blend": float(fit_summary.get("surface_multiscale_prior_blend_candidate", 0.0) or 0.0),
                "hybrid": bool(candidate.get("policy_val_prior_bin_gain_hybrid", False)),
                "bin_gain_enabled": bool(bin_gain.get("enabled", False)),
                "footprint_enabled": bool(footprint.get("enabled", False)),
                "allowed_bin_count": int(bin_gain.get("allowed_bin_count", 0) or 0),
                "candidate_bin_count": int(bin_gain.get("candidate_bin_count", 0) or 0),
                "covered_bin_count": int(footprint.get("covered_bin_count", 0) or 0),
                "candidate_bins_with_target_footprint": int(
                    footprint.get("candidate_bins_with_target_footprint", 0) or 0
                ),
                "allowed_bins_with_target_footprint": int(
                    footprint.get("allowed_bins_with_target_footprint", 0) or 0
                ),
                "pre_trunc_allowed_bins_with_target_footprint": optional_int(
                    footprint, "pre_trunc_allowed_bins_with_target_footprint"
                ),
                "target_views_used": int(footprint.get("target_views_used", 0) or 0),
                "target_views_examined": optional_int(footprint, "target_views_examined"),
                "views_with_target_coverage": optional_int(
                    footprint, "views_with_target_coverage"
                ),
            }
        )

    pre_trunc_values = [
        row["pre_trunc_allowed_bins_with_target_footprint"] for row in rows
    ]
    target_views_examined_values = [row["target_views_examined"] for row in rows]
    views_with_target_coverage_values = [row["views_with_target_coverage"] for row in rows]
    return {
        "profile_count": int(len(rows)),
        "enabled_profile_count": int(sum(1 for row in rows if row["footprint_enabled"])),
        "requested_in_any_candidate": bool(rows),
        "max_covered_bin_count": int(max((row["covered_bin_count"] for row in rows), default=0)),
        "max_candidate_bins_with_target_footprint": int(
            max((row["candidate_bins_with_target_footprint"] for row in rows), default=0)
        ),
        "max_allowed_bins_with_target_footprint": int(
            max((row["allowed_bins_with_target_footprint"] for row in rows), default=0)
        ),
        "max_pre_trunc_allowed_bins_with_target_footprint": optional_max(pre_trunc_values),
        "max_target_views_examined": optional_max(target_views_examined_values),
        "max_views_with_target_coverage": optional_max(views_with_target_coverage_values),
        "rows": rows[:16],
    }


def build_summary(run_dir: Path, references: dict[str, dict[str, float]]) -> dict[str, Any]:
    results_path = run_dir / "results.json"
    audit_path = run_dir / "surface_residual_region_texture_adapter_audit.json"
    per_view_path = run_dir / "per_view.json"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)

    method, metrics = first_method_metrics(load_json(results_path))
    audit = load_json(audit_path)
    fit_summary = audit.get("fit_summary") or {}
    policy_val = audit.get("policy_val") or {}
    fill_selection = policy_val.get("fill_mode_selection") or {}
    target_support = fit_summary.get("target_support_candidate_selection") or {}
    target_selected = target_support.get("selected_profile") or {}
    target_best = target_support.get("best_profile") or {}
    bin_gain = fit_summary.get("policy_val_prior_bin_gain_hybrid") or {}
    target_footprint = bin_gain.get("target_footprint_bin_certificate") or {}
    candidate_footprint = candidate_bin_gain_footprint_summary(audit)

    deltas = {
        name: {
            **metric_delta(metrics, reference),
            "strict_rgb_win": strict_win(metric_delta(metrics, reference)),
        }
        for name, reference in references.items()
    }
    per_view_count = 0
    if per_view_path.exists():
        per_view = load_json(per_view_path)
        if isinstance(per_view, dict) and per_view:
            payload = next(iter(per_view.values()))
            if isinstance(payload, dict) and isinstance(payload.get("PSNR"), dict):
                per_view_count = len(payload["PSNR"])

    target_support_enabled = bool(target_support.get("enabled", False))
    selected_certificate = target_support.get("selected_certificate") or infer_target_support_certificate(
        target_selected,
        enabled=target_support_enabled and bool(target_selected),
    )
    best_certificate = target_support.get("best_certificate") or (
        infer_target_support_certificate(target_best, enabled=target_support_enabled and bool(target_best))
        if target_best
        else {}
    )

    return {
        "run_dir": str(run_dir),
        "method": method,
        "metrics": metrics,
        "deltas": deltas,
        "per_view_count": int(per_view_count),
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0) or 0.0),
        "selected_blend": float(fit_summary.get("selected_surface_multiscale_prior_blend", 0.0) or 0.0),
        "selected_hybrid": bool(fit_summary.get("selected_policy_val_prior_bin_gain_hybrid", False)),
        "selected_support_added_faces": int(fit_summary.get("selected_support_added_faces", 0) or 0),
        "score_order_scope": str(fill_selection.get("score_order_scope", fill_selection.get("mode", ""))),
        "selectable_candidate_count": int(
            fill_selection.get(
                "selectable_candidate_count",
                fill_selection.get("accepted_candidate_count", 0),
            )
            or 0
        ),
        "all_candidate_count": int(
            fill_selection.get(
                "all_candidate_count",
                len(fill_selection.get("score_order") or []),
            )
            or 0
        ),
        "target_support": {
            "enabled": bool(target_support.get("enabled", False)),
            "selected_certificate": selected_certificate,
            "best_certificate": best_certificate,
            "best_scope": str(target_support.get("best_scope", "")),
            "selected_is_best": bool(target_support.get("selected_is_best_target_support", False)),
            "selected_is_global_best": bool(
                target_support.get("selected_is_global_best_target_support", False)
            ),
            "selected_changed_fraction": float(target_selected.get("changed_fraction", 0.0) or 0.0),
            "selected_cvar20_changed_fraction": float(
                target_selected.get("cvar20_view_changed_fraction", 0.0) or 0.0
            ),
            "selected_min_changed_fraction": float(
                target_selected.get("min_view_changed_fraction", 0.0) or 0.0
            ),
            "best_changed_fraction": float(target_best.get("changed_fraction", 0.0) or 0.0),
            "best_cvar20_changed_fraction": float(
                target_best.get("cvar20_view_changed_fraction", 0.0) or 0.0
            ),
            "best_min_changed_fraction": float(
                target_best.get("min_view_changed_fraction", 0.0) or 0.0
            ),
        },
        "bin_gain": {
            "enabled": bool(bin_gain.get("enabled", False)),
            "allowed_bin_count": int(bin_gain.get("allowed_bin_count", 0) or 0),
            "candidate_bin_count": int(bin_gain.get("candidate_bin_count", 0) or 0),
            "allowed_bin_fraction": float(bin_gain.get("allowed_bin_fraction", 0.0) or 0.0),
            "min_views": int(bin_gain.get("min_views", 0) or 0),
            "min_abs_gain": float(bin_gain.get("min_abs_gain", 0.0) or 0.0),
            "min_relative_gain": float(bin_gain.get("min_relative_gain", 0.0) or 0.0),
            "min_positive_view_fraction": float(
                bin_gain.get("min_positive_view_fraction", 0.0) or 0.0
            ),
        },
        "target_footprint": {
            "enabled": bool(target_footprint.get("enabled", False)),
            "selected_enabled": bool(target_footprint.get("enabled", False)),
            "covered_bin_count": int(target_footprint.get("covered_bin_count", 0) or 0),
            "candidate_bins_with_target_footprint": int(
                target_footprint.get("candidate_bins_with_target_footprint", 0) or 0
            ),
            "pre_trunc_allowed_bins_with_target_footprint": optional_int(
                target_footprint, "pre_trunc_allowed_bins_with_target_footprint"
            ),
            "allowed_bins_with_target_footprint": int(
                target_footprint.get("allowed_bins_with_target_footprint", 0) or 0
            ),
            "target_views_used": int(target_footprint.get("target_views_used", 0) or 0),
            "target_views_examined": optional_int(target_footprint, "target_views_examined"),
            "views_with_target_coverage": optional_int(
                target_footprint, "views_with_target_coverage"
            ),
            "candidate_level": candidate_footprint,
        },
        "score_order": compact_score_order(fill_selection.get("score_order") or [], 8),
    }


def write_markdown(summary: dict[str, Any], output_md: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# v78 Target Certificate Run Summary",
        "",
        f"Run dir: `{summary['run_dir']}`",
        f"Method: `{summary['method']}`",
        "",
        "## Metrics",
        "",
        "| PSNR | SSIM | LPIPS | per-view count |",
        "|---:|---:|---:|---:|",
        f"| `{metrics['PSNR']:.9f}` | `{metrics['SSIM']:.9f}` | `{metrics['LPIPS']:.9f}` | `{summary['per_view_count']}` |",
        "",
        "## Reference Deltas",
        "",
        "| reference | dPSNR | dSSIM | dLPIPS | strict RGB win |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, delta in summary["deltas"].items():
        lines.append(
            f"| {name} | `{delta['dPSNR']:+.9f}` | `{delta['dSSIM']:+.9f}` | "
            f"`{delta['dLPIPS']:+.9f}` | `{delta['strict_rgb_win']}` |"
        )

    target = summary["target_support"]
    bin_gain = summary["bin_gain"]
    footprint = summary["target_footprint"]
    footprint_candidates = footprint.get("candidate_level") or {}
    selected_cert = target.get("selected_certificate") or {}
    best_cert = target.get("best_certificate") or {}
    lines.extend(
        [
            "",
            "## Selected Policy",
            "",
            f"- accepted: `{summary['accepted']}`",
            f"- effective policy: `{summary['effective_policy']}`",
            f"- selected alpha: `{summary['selected_alpha']}`",
            f"- selected blend: `{summary['selected_blend']}`",
            f"- selected hybrid: `{summary['selected_hybrid']}`",
            f"- selected support added faces: `{summary['selected_support_added_faces']}`",
            f"- score order scope: `{summary.get('score_order_scope', '')}`",
            f"- selectable candidates: `{summary.get('selectable_candidate_count', 0)}` / `{summary.get('all_candidate_count', 0)}`",
            "",
            "## Target-Support Certificate",
            "",
            f"- enabled: `{target['enabled']}`",
            f"- selected certificate passed: `{bool(selected_cert.get('passed', False))}`",
            f"- selected certificate inferred from profile: `{bool(selected_cert.get('inferred_from_profile', False))}`",
            f"- best certificate passed: `{bool(best_cert.get('passed', False))}`",
            f"- best certificate inferred from profile: `{bool(best_cert.get('inferred_from_profile', False))}`",
            f"- selected is best target support: `{target['selected_is_best']}`",
            f"- selected is global best target support: `{target.get('selected_is_global_best', False)}`",
            f"- best scope: `{target.get('best_scope', '')}`",
            f"- selected changed fraction: `{target['selected_changed_fraction']:.9f}`",
            f"- selected CVaR20 changed fraction: `{target['selected_cvar20_changed_fraction']:.9f}`",
            f"- selected min-view changed fraction: `{target['selected_min_changed_fraction']:.9f}`",
            f"- best changed fraction: `{target['best_changed_fraction']:.9f}`",
            f"- best CVaR20 changed fraction: `{target['best_cvar20_changed_fraction']:.9f}`",
            f"- best min-view changed fraction: `{target['best_min_changed_fraction']:.9f}`",
            "",
            "## Bin-Gain / Target-Footprint Certificate",
            "",
            f"- bin-gain enabled: `{bin_gain['enabled']}`",
            f"- allowed bins: `{bin_gain['allowed_bin_count']}` / `{bin_gain['candidate_bin_count']}`",
            f"- allowed bin fraction: `{bin_gain['allowed_bin_fraction']:.9f}`",
            f"- min views: `{bin_gain['min_views']}`",
            f"- min abs gain: `{bin_gain['min_abs_gain']}`",
            f"- min relative gain: `{bin_gain['min_relative_gain']}`",
            f"- min positive-view fraction: `{bin_gain['min_positive_view_fraction']}`",
            f"- selected target-footprint enabled: `{footprint['selected_enabled']}`",
            f"- selected target covered bins: `{footprint['covered_bin_count']}`",
            f"- selected candidate bins with target footprint: `{footprint['candidate_bins_with_target_footprint']}`",
            f"- selected pre-trunc allowed bins with target footprint: {md_value(footprint['pre_trunc_allowed_bins_with_target_footprint'])}",
            f"- selected allowed bins with target footprint: `{footprint['allowed_bins_with_target_footprint']}`",
            f"- selected target footprint views used: `{footprint['target_views_used']}`",
            f"- selected target footprint views examined: {md_value(footprint['target_views_examined'])}",
            f"- selected views with target coverage: {md_value(footprint['views_with_target_coverage'])}",
            f"- candidate-level footprint profiles: `{footprint_candidates.get('enabled_profile_count', 0)}` / `{footprint_candidates.get('profile_count', 0)}`",
            f"- max candidate-level covered bins: `{footprint_candidates.get('max_covered_bin_count', 0)}`",
            f"- max candidate-level allowed bins with target footprint: `{footprint_candidates.get('max_allowed_bins_with_target_footprint', 0)}`",
            f"- max candidate-level pre-trunc allowed bins with target footprint: {md_value(footprint_candidates.get('max_pre_trunc_allowed_bins_with_target_footprint'))}",
            f"- max candidate-level target views examined: {md_value(footprint_candidates.get('max_target_views_examined'))}",
            f"- max candidate-level views with target coverage: {md_value(footprint_candidates.get('max_views_with_target_coverage'))}",
            "",
            "## Candidate Order",
            "",
            "| rank | blend | hybrid | allowed bins | accepted | alpha | rel gain | SSIM gain | L1 gain | target changed | target cert | cert inferred |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(summary["score_order"], start=1):
        lines.append(
            f"| {idx} | `{row['blend']}` | `{row['hybrid']}` | `{row['allowed_bins']}` | "
            f"`{row['accepted']}` | `{row['alpha']}` | `{row['relative_gain']:.9f}` | "
            f"`{row['ssim_gain']:.9f}` | `{row['image_l1_gain']:.9f}` | "
            f"`{row['target_changed_fraction']:.9f}` | `{row['target_certificate_passed']}` | "
            f"`{row['target_certificate_inferred']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- Promote only if this row beats the current v64/v56 counter reference on PSNR, SSIM, and LPIPS and the certificate fields pass.",
            "- If it only beats v75/v76/v77 but not v64/v56, keep it as a diagnostic.",
            "- If target support improves without metric gain, report it as observability/certificate progress, not as a promoted endpoint.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Optional NAME:PSNR:SSIM:LPIPS reference row. Defaults include counter v64/v75/v76/v77.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = dict(COUNTER_REFERENCES)
    for item in args.reference:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"bad --reference {item!r}; expected NAME:PSNR:SSIM:LPIPS")
        name, psnr, ssim, lpips = parts
        references[name] = {
            "PSNR": float(psnr),
            "SSIM": float(ssim),
            "LPIPS": float(lpips),
        }
    summary = build_summary(args.run_dir, references)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
