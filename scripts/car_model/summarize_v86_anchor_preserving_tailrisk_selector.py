#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from summarize_v84_strict_v82_capacity_selector import (
    METRICS,
    SCENES,
    baseline_from_v64,
    copy_if_small,
    first_method_metrics,
    fmt,
    metric_delta,
    nonregressive,
    read_json,
    render_gt_from_source,
    replace_symlink,
    strict_win,
    summarize,
)


DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_V84_SUMMARY = DEFAULT_ROOT / "v84_strict_v82_capacity_selector_full9_summary.json"
DEFAULT_V84_SELECTED_ROOT = DEFAULT_ROOT / "v84_strict_v82_capacity_selector_selected_full9"
DEFAULT_OUTPUT_JSON = DEFAULT_ROOT / "v86_anchor_preserving_tailrisk_selector_full9_summary.json"
DEFAULT_OUTPUT_MD = DEFAULT_ROOT / "v86_anchor_preserving_tailrisk_selector_full9_summary.md"
DEFAULT_MATERIALIZE_ROOT = DEFAULT_ROOT / "v86_anchor_preserving_tailrisk_selector_selected_full9"
DEFAULT_V85_CANDIDATES = (
    "counter="
    "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/"
    "v85_target_tailrisk_counter_20260625",
)


POLICY_KEYS = (
    "selected_cvar20_view_relative_gain",
    "selected_min_view_relative_gain",
    "selected_ssim_gain",
    "selected_ssim_min_view_gain",
    "selected_image_l1_gain",
    "selected_image_l1_min_view_gain",
    "selected_image_l1_cvar20_view_gain",
)


def parse_candidate_specs(items: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"candidate spec must be scene=path, got {item!r}")
        scene, path = item.split("=", 1)
        scene = scene.strip()
        if scene not in SCENES:
            raise ValueError(f"unknown scene in candidate spec: {scene}")
        paths[scene] = Path(path)
    return paths


def audit_policy_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    risk = audit.get("policy_val_risk_gate", {}) or {}
    target = audit.get("target_apply", {}) or {}
    tail = audit.get("target_footprint_tail_risk_gate", {}) or {}
    tail_profile = tail.get("selected_profile", {}) or {}
    fit_summary = audit.get("fit_summary", {}) or {}
    policy_val = audit.get("policy_val", {}) or {}
    fill_mode = policy_val.get("fill_mode_selection", {}) or {}
    score_order = fill_mode.get("score_order", []) or []
    selected = score_order[0] if score_order else {}
    out = {
        "audit_path": str(path),
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "policy_val_risk_gate_passed": bool(risk.get("passed", False)),
        "selected_alpha": float(audit.get("selected_alpha", 0.0) or 0.0),
        "changed_fraction": float(target.get("changed_fraction", 0.0) or 0.0),
        "selected_positive_view_fraction": float(
            risk.get("selected_positive_view_fraction", 0.0) or 0.0
        ),
        "selected_ssim_positive_view_fraction": float(
            risk.get("selected_ssim_positive_view_fraction", 0.0) or 0.0
        ),
        "selected_image_l1_positive_view_fraction": float(
            risk.get("selected_image_l1_positive_view_fraction", 0.0) or 0.0
        ),
        "tail_risk_enabled": bool(tail.get("enabled", False)),
        "tail_risk_selected_enabled": bool(tail_profile.get("enabled", False)),
        "tail_risk_allowed_keep_bin_count": int(tail_profile.get("allowed_keep_bin_count", 0) or 0),
        "tail_risk_rejected_bin_count": int(tail_profile.get("rejected_bin_count", 0) or 0),
        "tail_risk_candidate_bins_with_target_footprint": int(
            tail_profile.get("candidate_bins_with_target_footprint", 0) or 0
        ),
        "selected_candidate_label": str(fit_summary.get("selected_candidate_label", "")),
        "selected_policy_val_prior_bin_gain_hybrid": bool(
            fit_summary.get("selected_policy_val_prior_bin_gain_hybrid", False)
        ),
        "policy_candidate_support": str(selected.get("support_label", selected.get("support", ""))),
        "policy_candidate_texture": int(selected.get("texture_size", 0) or 0),
        "policy_candidate_prior_blend": float(
            selected.get("surface_multiscale_prior_blend", 0.0) or 0.0
        ),
    }
    for key in POLICY_KEYS:
        out[key] = float(risk.get(key, 0.0) or 0.0)
    return out


def anchor_policy_summary(v84_row: dict[str, Any]) -> dict[str, Any]:
    if v84_row.get("selected_source") == "v82_capacity_prerank":
        audit = dict(v84_row.get("v82_audit") or {})
        return {
            "source": "v84_v82_capacity_prerank",
            "has_train_policy": bool(audit),
            "selected_alpha": float(audit.get("selected_alpha", 0.0) or 0.0),
            "changed_fraction": float(audit.get("changed_fraction", 0.0) or 0.0),
            "selected_positive_view_fraction": 1.0,
            "selected_ssim_positive_view_fraction": float(
                audit.get("selected_ssim_positive_view_fraction", 0.0) or 0.0
            ),
            "selected_image_l1_positive_view_fraction": float(
                audit.get("selected_image_l1_positive_view_fraction", 0.0) or 0.0
            ),
            **{
                "selected_cvar20_view_relative_gain": float(
                    audit.get("selected_cvar20_view_relative_gain", 0.0) or 0.0
                ),
                "selected_min_view_relative_gain": float(
                    audit.get("selected_min_view_relative_gain", 0.0) or 0.0
                ),
                "selected_ssim_gain": float(audit.get("selected_ssim_gain", 0.0) or 0.0),
                "selected_ssim_min_view_gain": float(
                    audit.get("selected_ssim_min_view_gain", 0.0) or 0.0
                ),
                "selected_image_l1_gain": float(
                    audit.get("selected_image_l1_gain", 0.0) or 0.0
                ),
                "selected_image_l1_min_view_gain": float(
                    audit.get("selected_image_l1_min_view_gain", 0.0) or 0.0
                ),
                "selected_image_l1_cvar20_view_gain": float(
                    audit.get("selected_image_l1_cvar20_view_gain", 0.0) or 0.0
                ),
            },
        }
    return {
        "source": str(v84_row.get("selected_source", "v84_anchor")),
        "has_train_policy": False,
    }


def candidate_dominates_anchor(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    eps: float,
) -> tuple[bool, list[str]]:
    if not bool(anchor.get("has_train_policy", False)):
        return True, []
    reasons: list[str] = []
    for key in POLICY_KEYS:
        candidate_value = float(candidate.get(key, 0.0))
        anchor_value = float(anchor.get(key, 0.0))
        if candidate_value + eps < anchor_value:
            reasons.append(f"{key}_below_anchor:{candidate_value}<{anchor_value}")
    return not reasons, reasons


def should_use_v85(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not candidate:
        return False, ["missing_v85_candidate"]
    if not candidate.get("accepted", False):
        reasons.append("v85_not_accepted")
    if candidate.get("effective_policy") != "accepted_atlas":
        reasons.append(f"v85_not_accepted_atlas:{candidate.get('effective_policy')}")
    if not candidate.get("policy_val_risk_gate_passed", False):
        reasons.append("policy_val_risk_gate_not_passed")
    if not candidate.get("selected_policy_val_prior_bin_gain_hybrid", False):
        reasons.append("selected_policy_val_prior_bin_gain_hybrid_not_enabled")
    if not candidate.get("tail_risk_enabled", False):
        reasons.append("tail_risk_not_enabled")
    if not candidate.get("tail_risk_selected_enabled", False):
        reasons.append("tail_risk_selected_profile_not_enabled")
    if int(candidate.get("tail_risk_allowed_keep_bin_count", 0)) < args.min_tailrisk_allowed_bins:
        reasons.append(
            "tailrisk_allowed_bins_below:"
            f"{candidate.get('tail_risk_allowed_keep_bin_count')}<{args.min_tailrisk_allowed_bins}"
        )
    alpha = float(candidate.get("selected_alpha", 0.0))
    if alpha < args.min_selected_alpha:
        reasons.append(f"selected_alpha_below:{alpha}<{args.min_selected_alpha}")
    if alpha > args.max_selected_alpha:
        reasons.append(f"selected_alpha_above:{alpha}>{args.max_selected_alpha}")
    changed = float(candidate.get("changed_fraction", 0.0))
    if changed < args.min_changed_fraction:
        reasons.append(f"changed_fraction_below:{changed}<{args.min_changed_fraction}")
    if float(candidate.get("selected_ssim_gain", 0.0)) < args.min_policy_val_ssim_gain:
        reasons.append(
            f"ssim_gain_below:{candidate.get('selected_ssim_gain')}<{args.min_policy_val_ssim_gain}"
        )
    if float(candidate.get("selected_ssim_positive_view_fraction", 0.0)) < args.min_ssim_positive_fraction:
        reasons.append(
            "ssim_positive_fraction_below:"
            f"{candidate.get('selected_ssim_positive_view_fraction')}<{args.min_ssim_positive_fraction}"
        )
    if float(candidate.get("selected_ssim_min_view_gain", 0.0)) < args.min_ssim_min_view_gain:
        reasons.append(
            f"ssim_min_view_gain_below:{candidate.get('selected_ssim_min_view_gain')}<{args.min_ssim_min_view_gain}"
        )
    if float(candidate.get("selected_image_l1_gain", 0.0)) < args.min_policy_val_l1_gain:
        reasons.append(
            f"l1_gain_below:{candidate.get('selected_image_l1_gain')}<{args.min_policy_val_l1_gain}"
        )
    if float(candidate.get("selected_image_l1_positive_view_fraction", 0.0)) < args.min_l1_positive_fraction:
        reasons.append(
            "l1_positive_fraction_below:"
            f"{candidate.get('selected_image_l1_positive_view_fraction')}<{args.min_l1_positive_fraction}"
        )
    if float(candidate.get("selected_image_l1_min_view_gain", 0.0)) < args.min_l1_min_view_gain:
        reasons.append(
            f"l1_min_view_gain_below:{candidate.get('selected_image_l1_min_view_gain')}<{args.min_l1_min_view_gain}"
        )
    if float(candidate.get("selected_image_l1_cvar20_view_gain", 0.0)) < args.min_l1_cvar20_view_gain:
        reasons.append(
            "l1_cvar20_view_gain_below:"
            f"{candidate.get('selected_image_l1_cvar20_view_gain')}<{args.min_l1_cvar20_view_gain}"
        )
    dominates, dominance_reasons = candidate_dominates_anchor(candidate, anchor, args.policy_dominance_eps)
    if not dominates:
        reasons.extend(dominance_reasons)
    return not reasons, reasons


def materialize_selected_tree(
    payload: dict[str, Any],
    selected_root: Path,
    v84_selected_root: Path,
    max_copy_bytes: int,
) -> dict[str, Any]:
    selected_root.mkdir(parents=True, exist_ok=True)
    small_names = (
        "results.json",
        "per_view.json",
        "surface_residual_region_texture_adapter_audit.json",
        "surface_residual_region_texture_adapter_audit.md",
        "topology_audit.json",
        "topology_audit.md",
        "selection_manifest.json",
    )
    records: list[dict[str, Any]] = []
    for row in payload["rows"]:
        scene = str(row["scene"])
        scene_root = selected_root / scene
        scene_root.mkdir(parents=True, exist_ok=True)
        if row["selected_source"] == "v85_target_tailrisk":
            source_dir = Path(str(row["v85_candidate_dir"]))
        else:
            source_dir = v84_selected_root / scene
        copied_files: list[str] = []
        for name in small_names:
            if copy_if_small(source_dir / name, scene_root / name, max_copy_bytes):
                copied_files.append(name)
        log_path = source_dir / f"apply_metrics_{scene}.log"
        if copy_if_small(log_path, scene_root / log_path.name, max_copy_bytes):
            copied_files.append(log_path.name)
        log_path = source_dir / "logs" / f"apply_metrics_{scene}.log"
        if copy_if_small(log_path, scene_root / log_path.name, max_copy_bytes):
            copied_files.append(log_path.name)
        render_dir, gt_dir, render_source = render_gt_from_source(source_dir)
        render_linked = False
        gt_linked = False
        if render_dir is not None:
            replace_symlink(scene_root / "renders", render_dir)
            render_linked = True
        if gt_dir is not None:
            replace_symlink(scene_root / "gt", gt_dir)
            gt_linked = True
        record = {
            "scene": scene,
            "selected_source": row["selected_source"],
            "anchor_source": row["anchor_source"],
            "source_dir": str(source_dir),
            "copied_files": copied_files,
            "render_source": render_source,
            "render_dir": "" if render_dir is None else str(render_dir),
            "gt_dir": "" if gt_dir is None else str(gt_dir),
            "render_symlink": str(scene_root / "renders") if render_linked else "",
            "gt_symlink": str(scene_root / "gt") if gt_linked else "",
            "render_linked": render_linked,
            "gt_linked": gt_linked,
            "metrics": row["selected_metrics"],
            "anchor_metrics": row["anchor_metrics"],
            "guard_passed": bool(row["guard_passed"]),
            "guard_reject_reasons": row["guard_reject_reasons"],
            "selection_uses_heldout_metrics": False,
        }
        (scene_root / "selection_manifest.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    manifest = {
        "materialized_root": str(selected_root),
        "scene_count": len(records),
        "max_copy_bytes": int(max_copy_bytes),
        "render_linked_scene_count": int(sum(1 for item in records if item["render_linked"] and item["gt_linked"])),
        "selection_uses_heldout_metrics": False,
        "status": "V86_ANCHOR_PRESERVING_TAILRISK_SELECTOR_MATERIALIZED",
        "scenes": records,
    }
    (selected_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    v84 = read_json(args.v84_summary)
    v84_rows = {str(row["scene"]): row for row in v84["rows"]}
    candidate_paths = parse_candidate_specs([str(x) for x in args.v85_candidate])
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        anchor_row = v84_rows[scene]
        anchor_metrics = {key: float(anchor_row["selected_metrics"][key]) for key in METRICS}
        v64_metrics = {key: float(anchor_row["v64_metrics"][key]) for key in METRICS}
        v56_metrics = {key: float(anchor_row["v56_metrics"][key]) for key in METRICS}
        v52_metrics = {key: float(anchor_row["v52_metrics"][key]) for key in METRICS}
        anchor_policy = anchor_policy_summary(anchor_row)

        candidate_dir = candidate_paths.get(scene, args.missing_candidate_dir / scene)
        result_path = candidate_dir / "results.json"
        audit_path = candidate_dir / "surface_residual_region_texture_adapter_audit.json"
        candidate_method = ""
        candidate_metrics: dict[str, float] | None = None
        candidate_audit: dict[str, Any] = {}
        candidate_found = bool(result_path.is_file() or audit_path.is_file())
        if audit_path.is_file():
            candidate_audit = audit_policy_summary(audit_path)
        if result_path.is_file():
            candidate_method, candidate_metrics = first_method_metrics(result_path)
        guard_passed, reject_reasons = should_use_v85(candidate_audit, anchor_policy, args)
        use_v85 = bool(candidate_metrics is not None and guard_passed)
        selected_metrics = candidate_metrics if use_v85 and candidate_metrics is not None else anchor_metrics
        selected_source = "v85_target_tailrisk" if use_v85 else "v84_anchor_fallback"

        baselines = {
            "v84_anchor": anchor_metrics,
            "v64": v64_metrics,
            "v56": v56_metrics,
            "v52": v52_metrics,
        }
        for label in ("no-op", "v48", "v50"):
            baselines[label] = baseline_from_v64(anchor_row, label)
        comparisons: dict[str, Any] = {}
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
                "anchor_source": str(anchor_row["selected_source"]),
                "selected_metrics": selected_metrics,
                "anchor_metrics": anchor_metrics,
                "v64_metrics": v64_metrics,
                "v56_metrics": v56_metrics,
                "v52_metrics": v52_metrics,
                "anchor_policy": anchor_policy,
                "v85_method": candidate_method,
                "v85_metrics": candidate_metrics,
                "v85_audit": candidate_audit,
                "v85_candidate_found": candidate_found,
                "v85_candidate_dir": str(candidate_dir),
                "v85_result_path": str(result_path),
                "guard_passed": guard_passed,
                "guard_reject_reasons": reject_reasons,
                "comparisons": comparisons,
            }
        )

    candidate_complete = int(sum(bool(row["v85_metrics"] is not None and row["v85_audit"]) for row in rows))
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v86 anchor-preserving selector over v85 target-footprint tail-risk and v84 fallback",
        "status": (
            "FULL9_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
            if candidate_complete == len(SCENES)
            else "PARTIAL_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
        ),
        "selection_uses_heldout_metrics": False,
        "caveat": (
            "The selector uses train/policy-val audit fields only. Held-out metrics are used "
            "after selection only to validate whether the selected endpoint improved."
        ),
        "inputs": {
            "v84_summary": str(args.v84_summary),
            "v84_selected_root": str(args.v84_selected_root),
            "v85_candidates": [str(x) for x in args.v85_candidate],
        },
        "policy": {
            "min_selected_alpha": float(args.min_selected_alpha),
            "max_selected_alpha": float(args.max_selected_alpha),
            "min_changed_fraction": float(args.min_changed_fraction),
            "min_tailrisk_allowed_bins": int(args.min_tailrisk_allowed_bins),
            "min_policy_val_ssim_gain": float(args.min_policy_val_ssim_gain),
            "min_ssim_positive_fraction": float(args.min_ssim_positive_fraction),
            "min_ssim_min_view_gain": float(args.min_ssim_min_view_gain),
            "min_policy_val_l1_gain": float(args.min_policy_val_l1_gain),
            "min_l1_positive_fraction": float(args.min_l1_positive_fraction),
            "min_l1_min_view_gain": float(args.min_l1_min_view_gain),
            "min_l1_cvar20_view_gain": float(args.min_l1_cvar20_view_gain),
            "policy_dominance_eps": float(args.policy_dominance_eps),
            "rule": (
                "use v85 target-footprint tail-risk only when its train/policy-val audit "
                "passes fixed safety thresholds and dominates the current v84 selected "
                "anchor audit when that anchor has a comparable train-policy audit; "
                "otherwise preserve v84/v64 fallback"
            ),
        },
        "candidate_complete_scene_count": candidate_complete,
        "summary": {
            label: summarize(rows, label, args.metric_eps)
            for label in ("v84_anchor", "v64", "v56", "v52", "no-op", "v48", "v50")
        },
        "rows": rows,
    }
    if args.materialize_root:
        manifest = materialize_selected_tree(
            payload,
            Path(args.materialize_root),
            Path(args.v84_selected_root),
            int(args.max_copy_bytes),
        )
        payload["materialized_root"] = str(args.materialize_root)
        payload["materialized_scene_count"] = int(manifest["scene_count"])
        payload["render_linked_scene_count"] = int(manifest["render_linked_scene_count"])
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if int(payload.get("candidate_complete_scene_count", 0)) == len(SCENES):
        completion_note = (
            "The v85 tail-risk candidate set is complete for full9. The status remains "
            "report-only until this selector is validated on fresh scenes/protocols."
        )
    else:
        completion_note = (
            "The v85 tail-risk candidate set is not complete for full9. Missing scenes "
            "fall back to v84, so this is a conservative selector validation rather than "
            "a full v85 rerun."
        )
    lines: list[str] = [
        "# v86 Anchor-Preserving Target-Footprint Tail-Risk Selector",
        "",
        f"Date: `{payload['date']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This selector keeps the current v84 selected endpoint as the anchor. It promotes",
        "a v85 target-footprint tail-risk candidate only when fixed train/policy-val audit",
        "fields pass safety thresholds and strictly dominate the current anchor audit when",
        "that anchor has a comparable train-policy audit. Held-out metrics are not used for",
        "branch selection.",
        "",
        "## Fixed Guard",
        "",
        f"- selected alpha range: `[{payload['policy']['min_selected_alpha']}, {payload['policy']['max_selected_alpha']}]`",
        f"- minimum target changed fraction: `{payload['policy']['min_changed_fraction']}`",
        f"- minimum tail-risk allowed bins: `{payload['policy']['min_tailrisk_allowed_bins']}`",
        f"- minimum policy-val SSIM gain: `{payload['policy']['min_policy_val_ssim_gain']}`",
        f"- minimum policy-val SSIM positive fraction: `{payload['policy']['min_ssim_positive_fraction']}`",
        f"- minimum policy-val SSIM min-view gain: `{payload['policy']['min_ssim_min_view_gain']}`",
        f"- minimum policy-val image-L1 gain: `{payload['policy']['min_policy_val_l1_gain']}`",
        f"- minimum policy-val image-L1 positive fraction: `{payload['policy']['min_l1_positive_fraction']}`",
        f"- minimum policy-val image-L1 min-view gain: `{payload['policy']['min_l1_min_view_gain']}`",
        f"- minimum policy-val image-L1 CVaR20 gain: `{payload['policy']['min_l1_cvar20_view_gain']}`",
        f"- policy dominance epsilon: `{payload['policy']['policy_dominance_eps']}`",
        "- otherwise preserve v84/v64 fallback",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v86 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Decisions",
            "",
            "| scene | selected | anchor | v85 found | guard | alpha | changed | tail keep bins | pval ssim | pval l1 | dPSNR vs v84 | dSSIM vs v84 | dLPIPS vs v84 | reject reasons |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        audit = row["v85_audit"] or {}
        delta = row["comparisons"]["v84_anchor"]["delta"]
        reasons = ", ".join(row["guard_reject_reasons"]) if row["guard_reject_reasons"] else "pass"
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {row['anchor_source']} | "
            f"{int(row['v85_candidate_found'])} | {int(row['guard_passed'])} | "
            f"{float(audit.get('selected_alpha', 0.0)):.4f} | "
            f"{float(audit.get('changed_fraction', 0.0)):.6f} | "
            f"{int(audit.get('tail_risk_allowed_keep_bin_count', 0) or 0)} | "
            f"{float(audit.get('selected_ssim_gain', 0.0)):+.9f} | "
            f"{float(audit.get('selected_image_l1_gain', 0.0)):+.9f} | "
            f"{fmt(delta['PSNR'])} | {fmt(delta['SSIM'])} | {fmt(delta['LPIPS'])} | "
            f"`{reasons}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v86 is an anchor-preserving safety wrapper over v85. On the current evidence,",
            "the only materialized v85 tail-risk candidate is `counter`; it passes the basic",
            "tail-risk certificate, but it does not strictly dominate the v84/v82b anchor on",
            "train/policy-val SSIM and L1 audit fields. The selector therefore preserves v84.",
            "",
            "This is a real engineering improvement over the raw v85 diagnostic because it",
            "prevents a safety certificate from replacing an already stronger anchor. It is",
            "not a paper-level representation breakthrough; it is a guardrail needed before",
            "future tail-risk candidates can be evaluated fairly.",
            "",
            completion_note,
            "",
        ]
    )
    if payload.get("materialized_root"):
        lines.extend(
            [
                "## Materialized Artifacts",
                "",
                f"- selected root: `{payload['materialized_root']}`",
                f"- materialized scenes: `{payload.get('materialized_scene_count', 0)}`",
                f"- render/GT linked scenes: `{payload.get('render_linked_scene_count', 0)}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v84_summary", type=Path, default=DEFAULT_V84_SUMMARY)
    parser.add_argument("--v84_selected_root", type=Path, default=DEFAULT_V84_SELECTED_ROOT)
    parser.add_argument("--v85_candidate", action="append", default=list(DEFAULT_V85_CANDIDATES))
    parser.add_argument("--missing_candidate_dir", type=Path, default=DEFAULT_ROOT / "__missing_v85_tailrisk")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--materialize_root", type=Path, default=DEFAULT_MATERIALIZE_ROOT)
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    parser.add_argument("--min_selected_alpha", type=float, default=0.5)
    parser.add_argument("--max_selected_alpha", type=float, default=0.5)
    parser.add_argument("--min_changed_fraction", type=float, default=0.001)
    parser.add_argument("--min_tailrisk_allowed_bins", type=int, default=1)
    parser.add_argument("--min_policy_val_ssim_gain", type=float, default=2.0e-4)
    parser.add_argument("--min_ssim_positive_fraction", type=float, default=1.0)
    parser.add_argument("--min_ssim_min_view_gain", type=float, default=5.0e-5)
    parser.add_argument("--min_policy_val_l1_gain", type=float, default=2.0e-5)
    parser.add_argument("--min_l1_positive_fraction", type=float, default=0.9)
    parser.add_argument("--min_l1_min_view_gain", type=float, default=-1.0e-6)
    parser.add_argument("--min_l1_cvar20_view_gain", type=float, default=1.0e-6)
    parser.add_argument("--policy_dominance_eps", type=float, default=1.0e-12)
    parser.add_argument("--metric_eps", type=float, default=1.0e-7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "status": payload["status"],
                "candidate_complete_scene_count": payload["candidate_complete_scene_count"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
