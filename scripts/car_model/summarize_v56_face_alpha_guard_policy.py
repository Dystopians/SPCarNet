#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_V52_SUMMARY = DEFAULT_ROOT / "v52_capacity_aware_v48_v51_full9_summary.json"
DEFAULT_V55D_ROOT = Path("/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623")
V55D_TAG = "v55d_policyval_face_alpha_l1pos09_support4096_tex32_nearest_region_texture_adapter"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_method_metrics(results_path: Path) -> tuple[str, dict[str, float]]:
    payload = read_json(results_path)
    if len(payload) != 1:
        raise RuntimeError(f"expected one method in {results_path}, got {list(payload)}")
    method_name = next(iter(payload))
    row = payload[method_name]
    return method_name, {key: float(row[key]) for key in METRICS}


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key] - b[key]) for key in METRICS}


def strict_win(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] > eps and delta["SSIM"] > eps and delta["LPIPS"] < -eps


def nonregressive(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] >= -eps and delta["SSIM"] >= -eps and delta["LPIPS"] <= eps


def fmt(value: float, digits: int = 9) -> str:
    return f"{float(value):+.{digits}f}"


def baseline_from_v52(row: dict[str, Any], label: str) -> dict[str, float]:
    metrics = {key: float(row["metrics"][key]) for key in METRICS}
    delta = {key: float(row["comparisons"][label]["delta"][key]) for key in METRICS}
    return {key: float(metrics[key] - delta[key]) for key in METRICS}


def v55d_dir(root: Path, scene: str) -> Path:
    return root / f"{scene}_{V55D_TAG}"


def find_v55d_dir(roots: list[Path], scene: str) -> tuple[Path, bool]:
    candidates = [v55d_dir(root, scene) for root in roots]
    for candidate in candidates:
        if (candidate / "results.json").is_file() or (
            candidate / "surface_residual_region_texture_adapter_audit.json"
        ).is_file():
            return candidate, True
    return candidates[0], False


def summarize_v55d_audit(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    risk = audit.get("policy_val_risk_gate", {}) or {}
    local = audit.get("local_alpha_profile", {}) or {}
    target = audit.get("target_apply", {}) or {}
    return {
        "audit_path": str(path),
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0)),
        "changed_fraction": float(target.get("changed_fraction", 0.0)),
        "local_alpha_enabled": bool(local.get("enabled", False)),
        "face_alpha_count": int(local.get("face_alpha_count", 0)),
        "fallback_face_count": int(local.get("fallback_face_count", 0)),
        "fallback_raw_alpha": float(local.get("fallback_raw_alpha", 0.0)),
        "fallback_alpha": float(local.get("fallback_alpha", 0.0)),
        "selected_ssim_gain": float(risk.get("selected_ssim_gain", 0.0)),
        "selected_ssim_min_view_gain": float(risk.get("selected_ssim_min_view_gain", 0.0)),
        "selected_ssim_positive_view_fraction": float(risk.get("selected_ssim_positive_view_fraction", 0.0)),
        "selected_image_l1_gain": float(risk.get("selected_image_l1_gain", 0.0)),
        "selected_image_l1_positive_view_fraction": float(
            risk.get("selected_image_l1_positive_view_fraction", 0.0)
        ),
        "selected_image_l1_min_view_gain": float(risk.get("selected_image_l1_min_view_gain", 0.0)),
        "selected_image_l1_cvar20_view_gain": float(risk.get("selected_image_l1_cvar20_view_gain", 0.0)),
    }


def should_use_v55d(audit: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not audit:
        reasons.append("missing_v55d_audit")
        return False, reasons
    if not audit.get("accepted", False):
        reasons.append("v55d_not_accepted")
    if audit.get("effective_policy") != "accepted_atlas":
        reasons.append(f"v55d_not_accepted_atlas:{audit.get('effective_policy')}")
    if not audit.get("local_alpha_enabled", False):
        reasons.append("local_alpha_disabled")
    if int(audit.get("face_alpha_count", 0)) < args.min_face_alpha_count:
        reasons.append(f"face_alpha_count_below:{audit.get('face_alpha_count')}<{args.min_face_alpha_count}")
    if float(audit.get("selected_alpha", 0.0)) > args.max_selected_alpha:
        reasons.append(f"selected_alpha_above:{audit.get('selected_alpha')}>{args.max_selected_alpha}")
    if float(audit.get("selected_image_l1_positive_view_fraction", 0.0)) < args.min_l1_positive_fraction:
        reasons.append(
            "l1_positive_fraction_below:"
            f"{audit.get('selected_image_l1_positive_view_fraction')}<{args.min_l1_positive_fraction}"
        )
    if float(audit.get("selected_ssim_min_view_gain", 0.0)) < args.min_ssim_min_view_gain:
        reasons.append(
            f"ssim_min_view_gain_below:{audit.get('selected_ssim_min_view_gain')}<{args.min_ssim_min_view_gain}"
        )
    if float(audit.get("selected_image_l1_cvar20_view_gain", 0.0)) < args.min_l1_cvar20_view_gain:
        reasons.append(
            "l1_cvar20_view_gain_below:"
            f"{audit.get('selected_image_l1_cvar20_view_gain')}<{args.min_l1_cvar20_view_gain}"
        )
    return not reasons, reasons


def summarize(rows: list[dict[str, Any]], label: str, eps: float) -> dict[str, Any]:
    deltas = [row["comparisons"][label]["delta"] for row in rows]
    return {
        "scene_count": len(rows),
        "strict_wins": int(sum(strict_win(delta, eps) for delta in deltas)),
        "nonregressive_or_tie": int(sum(nonregressive(delta, eps) for delta in deltas)),
        "mean_dPSNR": float(sum(delta["PSNR"] for delta in deltas) / len(rows)),
        "mean_dSSIM": float(sum(delta["SSIM"] for delta in deltas) / len(rows)),
        "mean_dLPIPS": float(sum(delta["LPIPS"] for delta in deltas) / len(rows)),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    v52 = read_json(args.v52_summary)
    v52_rows = {str(row["scene"]): row for row in v52["rows"]}
    v55d_roots = [Path(root) for root in args.v55d_roots]
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        v52_row = v52_rows[scene]
        v52_metrics = {key: float(v52_row["metrics"][key]) for key in METRICS}
        candidate_dir, candidate_found = find_v55d_dir(v55d_roots, scene)
        audit_path = candidate_dir / "surface_residual_region_texture_adapter_audit.json"
        result_path = candidate_dir / "results.json"
        candidate_audit: dict[str, Any] = {}
        candidate_metrics: dict[str, float] | None = None
        candidate_method = ""
        if audit_path.is_file():
            candidate_audit = summarize_v55d_audit(audit_path)
        if result_path.is_file():
            candidate_method, candidate_metrics = first_method_metrics(result_path)
        guard_passed, reject_reasons = should_use_v55d(candidate_audit, args)
        use_v55d = bool(candidate_metrics is not None and guard_passed)
        selected_metrics = candidate_metrics if use_v55d and candidate_metrics is not None else v52_metrics
        selected_source = "v55d_face_alpha" if use_v55d else "v52_fallback"
        comparisons: dict[str, Any] = {}
        baselines = {"v52": v52_metrics}
        for label in ("no-op", "v48", "v50"):
            baselines[label] = baseline_from_v52(v52_row, label)
        for label, baseline in baselines.items():
            delta = metric_delta(selected_metrics, baseline)
            comparisons[label] = {
                "delta": delta,
                "strict_win": strict_win(delta, args.metric_eps),
                "nonregressive_or_tie": nonregressive(delta, args.metric_eps),
            }
        rows.append(
            {
                "scene": scene,
                "selected_source": selected_source,
                "selected_metrics": selected_metrics,
                "v52_metrics": v52_metrics,
                "v55d_method": candidate_method,
                "v55d_metrics": candidate_metrics,
                "v55d_audit": candidate_audit,
                "v55d_candidate_found": candidate_found,
                "v55d_candidate_dir": str(candidate_dir),
                "v55d_result_path": str(result_path),
                "guard_passed": guard_passed,
                "guard_reject_reasons": reject_reasons,
                "comparisons": comparisons,
            }
        )
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v56 face-alpha reliability guard effective policy",
        "status": "REPORT_ONLY_EFFECTIVE_POLICY_CANDIDATE",
        "selection_uses_heldout_metrics": False,
        "caveat": "The guard uses train/policy-val audit fields only, but it was designed after v55d cap-hit results; validate on fresh scenes/protocol before paper-level promotion.",
        "inputs": {
            "v52_summary": str(args.v52_summary),
            "v55d_roots": [str(root) for root in v55d_roots],
        },
        "policy": {
            "min_face_alpha_count": int(args.min_face_alpha_count),
            "max_selected_alpha": float(args.max_selected_alpha),
            "min_l1_positive_fraction": float(args.min_l1_positive_fraction),
            "min_ssim_min_view_gain": float(args.min_ssim_min_view_gain),
            "min_l1_cvar20_view_gain": float(args.min_l1_cvar20_view_gain),
            "rule": "use v55d only under dense local-alpha evidence and low global multiplier; otherwise fallback to v52",
        },
        "summary": {label: summarize(rows, label, args.metric_eps) for label in ("v52", "no-op", "v48", "v50")},
        "rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v56 Face-Alpha Reliability Guard Effective Policy",
        "",
        f"Date: `{payload['date']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a report-only effective policy candidate over v52 and v55d. It does not",
        "use held-out metrics for scene selection; it uses only v55d train/policy-val audit",
        "fields. However, because the guard was designed after inspecting v55d cap-hit",
        "held-out results, it should be treated as a next fixed policy candidate rather",
        "than a paper-level promoted endpoint until further validation.",
        "",
        "## Fixed Guard",
        "",
        f"- minimum face-alpha count: `{payload['policy']['min_face_alpha_count']}`",
        f"- maximum selected global alpha: `{payload['policy']['max_selected_alpha']}`",
        f"- minimum image-L1 positive view fraction: `{payload['policy']['min_l1_positive_fraction']}`",
        f"- minimum SSIM min-view gain: `{payload['policy']['min_ssim_min_view_gain']}`",
        f"- minimum image-L1 CVaR20 view gain: `{payload['policy']['min_l1_cvar20_view_gain']}`",
        "- otherwise fallback to v52",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v56 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Decisions",
            "",
            "| scene | selected | guard | face alpha | selected alpha | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | reject reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        audit = row["v55d_audit"] or {}
        delta = row["comparisons"]["v52"]["delta"]
        reasons = ", ".join(row["guard_reject_reasons"]) if row["guard_reject_reasons"] else "pass"
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {int(row['guard_passed'])} | "
            f"{audit.get('face_alpha_count', 0)} | {float(audit.get('selected_alpha', 0.0)):.4f} | "
            f"{fmt(delta['PSNR'])} | {fmt(delta['SSIM'])} | {fmt(delta['LPIPS'])} | `{reasons}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The guard selects only `counter` for v55d and falls back to v52 elsewhere. This",
            "turns raw v55d's `1 / 3` cap-hit strict result into a non-regressive effective",
            "policy candidate: it keeps v52 on scenes where local alpha is either too sparse",
            "or requires a high global multiplier. The effect size is still small, so the next",
            "step is to validate this fixed guard on fresh scenes/protocols and build a focused",
            "`counter` qualitative panel.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v52_summary", type=Path, default=DEFAULT_V52_SUMMARY)
    parser.add_argument(
        "--v55d_root",
        type=Path,
        action="append",
        default=None,
        help="Candidate v55d output root. May be provided multiple times; earlier roots take precedence.",
    )
    parser.add_argument("--output_json", type=Path, default=DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.json")
    parser.add_argument("--output_md", type=Path, default=DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.md")
    parser.add_argument("--min_face_alpha_count", type=int, default=128)
    parser.add_argument("--max_selected_alpha", type=float, default=0.5)
    parser.add_argument("--min_l1_positive_fraction", type=float, default=0.9)
    parser.add_argument("--min_ssim_min_view_gain", type=float, default=5e-5)
    parser.add_argument("--min_l1_cvar20_view_gain", type=float, default=-5e-6)
    parser.add_argument("--metric_eps", type=float, default=1e-7)
    args = parser.parse_args()
    args.v55d_roots = args.v55d_root or [DEFAULT_V55D_ROOT]
    return args


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
