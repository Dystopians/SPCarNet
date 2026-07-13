#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
DEFAULT_V64_SUMMARY = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_full9_summary.json"
DEFAULT_V64_SELECTED_ROOT = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_selected_full9"
DEFAULT_OUTPUT_JSON = DEFAULT_ROOT / "v84_strict_v82_capacity_selector_full9_summary.json"
DEFAULT_OUTPUT_MD = DEFAULT_ROOT / "v84_strict_v82_capacity_selector_full9_summary.md"
DEFAULT_MATERIALIZE_ROOT = DEFAULT_ROOT / "v84_strict_v82_capacity_selector_selected_full9"
DEFAULT_V82_ROOTS = (
    DEFAULT_ROOT / "v82_capacity_prerank_facealpha_20260624",
    DEFAULT_ROOT / "v82_capacity_prerank_facealpha_triad_20260624",
    Path("/dev/shm/peilincai_spcarnet_v82_capacity_prerank_facealpha_triad_20260624"),
)
DEFAULT_V82_TAG = "v82_capacity_prerank_facealpha_tex32_48_support4096_8192_{scene}_region_texture_adapter"


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


def baseline_from_v64(row: dict[str, Any], label: str) -> dict[str, float]:
    metrics = {key: float(row["selected_metrics"][key]) for key in METRICS}
    delta = {key: float(row["comparisons"][label]["delta"][key]) for key in METRICS}
    return {key: float(metrics[key] - delta[key]) for key in METRICS}


def candidate_name(scene: str, tag_template: str) -> str:
    return f"{scene}_{tag_template.format(scene=scene)}"


def find_candidate_dir(roots: list[Path], scene: str, tag_template: str) -> tuple[Path, bool]:
    name = candidate_name(scene, tag_template)
    expected = [root / name for root in roots]
    for candidate in expected:
        if (candidate / "results.json").is_file() or (
            candidate / "surface_residual_region_texture_adapter_audit.json"
        ).is_file():
            return candidate, True
    for root in roots:
        for candidate in sorted(root.glob(f"**/{name}")):
            if (candidate / "results.json").is_file() or (
                candidate / "surface_residual_region_texture_adapter_audit.json"
            ).is_file():
                return candidate, True
    return expected[0], False


def summarize_v82_audit(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    risk = audit.get("policy_val_risk_gate", {}) or {}
    target = audit.get("target_apply", {}) or {}
    policy_val = audit.get("policy_val", {}) or {}
    fill_mode = policy_val.get("fill_mode_selection", {}) or {}
    score_order = fill_mode.get("score_order", []) or []
    selected = score_order[0] if score_order else {}
    return {
        "audit_path": str(path),
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0) or 0.0),
        "changed_fraction": float(target.get("changed_fraction", 0.0) or 0.0),
        "selected_ssim_gain": float(risk.get("selected_ssim_gain", 0.0) or 0.0),
        "selected_ssim_positive_view_fraction": float(
            risk.get("selected_ssim_positive_view_fraction", 0.0) or 0.0
        ),
        "selected_ssim_min_view_gain": float(risk.get("selected_ssim_min_view_gain", 0.0) or 0.0),
        "selected_image_l1_gain": float(risk.get("selected_image_l1_gain", 0.0) or 0.0),
        "selected_image_l1_positive_view_fraction": float(
            risk.get("selected_image_l1_positive_view_fraction", 0.0) or 0.0
        ),
        "selected_image_l1_min_view_gain": float(
            risk.get("selected_image_l1_min_view_gain", 0.0) or 0.0
        ),
        "selected_image_l1_cvar20_view_gain": float(
            risk.get("selected_image_l1_cvar20_view_gain", 0.0) or 0.0
        ),
        "policy_candidate_support": str(selected.get("support_label", selected.get("support", ""))),
        "policy_candidate_texture": int(selected.get("texture_size", 0) or 0),
        "policy_candidate_prior_blend": float(selected.get("surface_multiscale_prior_blend", 0.0) or 0.0),
    }


def should_use_v82(audit: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not audit:
        reasons.append("missing_v82_audit")
        return False, reasons
    if not audit.get("accepted", False):
        reasons.append("v82_not_accepted")
    if audit.get("effective_policy") != "accepted_atlas":
        reasons.append(f"v82_not_accepted_atlas:{audit.get('effective_policy')}")
    alpha = float(audit.get("selected_alpha", 0.0))
    if alpha < args.min_selected_alpha:
        reasons.append(f"selected_alpha_below:{alpha}<{args.min_selected_alpha}")
    if alpha > args.max_selected_alpha:
        reasons.append(f"selected_alpha_above:{alpha}>{args.max_selected_alpha}")
    changed_fraction = float(audit.get("changed_fraction", 0.0))
    if changed_fraction < args.min_changed_fraction:
        reasons.append(f"changed_fraction_below:{changed_fraction}<{args.min_changed_fraction}")
    if float(audit.get("selected_ssim_gain", 0.0)) < args.min_policy_val_ssim_gain:
        reasons.append(
            f"ssim_gain_below:{audit.get('selected_ssim_gain')}<{args.min_policy_val_ssim_gain}"
        )
    if float(audit.get("selected_ssim_positive_view_fraction", 0.0)) < args.min_ssim_positive_fraction:
        reasons.append(
            "ssim_positive_fraction_below:"
            f"{audit.get('selected_ssim_positive_view_fraction')}<{args.min_ssim_positive_fraction}"
        )
    if float(audit.get("selected_ssim_min_view_gain", 0.0)) < args.min_ssim_min_view_gain:
        reasons.append(
            f"ssim_min_view_gain_below:{audit.get('selected_ssim_min_view_gain')}<{args.min_ssim_min_view_gain}"
        )
    if float(audit.get("selected_image_l1_gain", 0.0)) < args.min_policy_val_l1_gain:
        reasons.append(
            f"l1_gain_below:{audit.get('selected_image_l1_gain')}<{args.min_policy_val_l1_gain}"
        )
    if float(audit.get("selected_image_l1_positive_view_fraction", 0.0)) < args.min_l1_positive_fraction:
        reasons.append(
            "l1_positive_fraction_below:"
            f"{audit.get('selected_image_l1_positive_view_fraction')}<{args.min_l1_positive_fraction}"
        )
    if float(audit.get("selected_image_l1_min_view_gain", 0.0)) < args.min_l1_min_view_gain:
        reasons.append(
            f"l1_min_view_gain_below:{audit.get('selected_image_l1_min_view_gain')}<{args.min_l1_min_view_gain}"
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
        "scene_count": int(len(rows)),
        "strict_wins": int(sum(strict_win(delta, eps) for delta in deltas)),
        "nonregressive_or_tie": int(sum(nonregressive(delta, eps) for delta in deltas)),
        "mean_dPSNR": float(sum(delta["PSNR"] for delta in deltas) / len(deltas)),
        "mean_dSSIM": float(sum(delta["SSIM"] for delta in deltas) / len(deltas)),
        "mean_dLPIPS": float(sum(delta["LPIPS"] for delta in deltas) / len(deltas)),
    }


def _valid_dir(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_dir() else None


def render_gt_from_source(source_dir: Path) -> tuple[Path | None, Path | None, str]:
    manifest_path = source_dir / "selection_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        render_dir = _valid_dir(manifest.get("render_dir"))
        gt_dir = _valid_dir(manifest.get("gt_dir"))
        if render_dir is not None and gt_dir is not None:
            return render_dir, gt_dir, "selection_manifest"
    audit_path = source_dir / "surface_residual_region_texture_adapter_audit.json"
    if audit_path.is_file():
        audit = read_json(audit_path)
        target = audit.get("target_apply", {}) or {}
        render_dir = _valid_dir(target.get("render_dir"))
        gt_dir = _valid_dir(target.get("gt_dir"))
        if render_dir is not None and gt_dir is not None:
            return render_dir, gt_dir, "audit_target_apply"
    for render_dir in sorted(source_dir.resolve().glob("test/*/renders")):
        gt_dir = render_dir.parent / "gt"
        if render_dir.is_dir() and gt_dir.is_dir():
            return render_dir, gt_dir, "source_dir_test_glob"
    return None, None, "not_found"


def copy_if_small(src: Path, dst: Path, max_bytes: int) -> bool:
    if not src.is_file():
        return False
    if src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def materialize_selected_tree(
    payload: dict[str, Any],
    selected_root: Path,
    v64_selected_root: Path,
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
    )
    records: list[dict[str, Any]] = []
    for row in payload["rows"]:
        scene = str(row["scene"])
        scene_root = selected_root / scene
        scene_root.mkdir(parents=True, exist_ok=True)
        if row["selected_source"] == "v82_capacity_prerank":
            source_dir = Path(str(row["v82_candidate_dir"]))
        else:
            source_dir = v64_selected_root / scene
        copied_files: list[str] = []
        for name in small_names:
            if copy_if_small(source_dir / name, scene_root / name, max_copy_bytes):
                copied_files.append(name)
        log_path = source_dir / f"apply_metrics_{scene}.log"
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
            "v64_metrics": row["v64_metrics"],
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
        "status": "V84_STRICT_V82_CAPACITY_SELECTOR_MATERIALIZED",
        "scenes": records,
    }
    (selected_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    v64 = read_json(args.v64_summary)
    v64_rows = {str(row["scene"]): row for row in v64["rows"]}
    roots = [Path(root) for root in args.v82_root]
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        v64_row = v64_rows[scene]
        v64_metrics = {key: float(v64_row["selected_metrics"][key]) for key in METRICS}
        v56_metrics = {key: float(v64_row["v56_metrics"][key]) for key in METRICS}
        v52_metrics = {key: float(v64_row["v52_metrics"][key]) for key in METRICS}
        candidate_dir, candidate_found = find_candidate_dir(roots, scene, str(args.v82_tag_template))
        result_path = candidate_dir / "results.json"
        audit_path = candidate_dir / "surface_residual_region_texture_adapter_audit.json"
        candidate_method = ""
        candidate_metrics: dict[str, float] | None = None
        candidate_audit: dict[str, Any] = {}
        if audit_path.is_file():
            candidate_audit = summarize_v82_audit(audit_path)
        if result_path.is_file():
            candidate_method, candidate_metrics = first_method_metrics(result_path)
        guard_passed, reject_reasons = should_use_v82(candidate_audit, args)
        use_v82 = bool(candidate_metrics is not None and guard_passed)
        selected_metrics = candidate_metrics if use_v82 and candidate_metrics is not None else v64_metrics
        selected_source = "v82_capacity_prerank" if use_v82 else "v64_fallback"
        baselines = {
            "v64": v64_metrics,
            "v56": v56_metrics,
            "v52": v52_metrics,
        }
        for label in ("no-op", "v48", "v50"):
            baselines[label] = baseline_from_v64(v64_row, label)
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
                "selected_metrics": selected_metrics,
                "v64_metrics": v64_metrics,
                "v56_metrics": v56_metrics,
                "v52_metrics": v52_metrics,
                "v82_method": candidate_method,
                "v82_metrics": candidate_metrics,
                "v82_audit": candidate_audit,
                "v82_candidate_found": candidate_found,
                "v82_candidate_dir": str(candidate_dir),
                "v82_result_path": str(result_path),
                "guard_passed": guard_passed,
                "guard_reject_reasons": reject_reasons,
                "comparisons": comparisons,
            }
        )
    candidate_complete = int(sum(bool(row["v82_metrics"] is not None and row["v82_audit"]) for row in rows))
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v84 strict selector over v82 capacity-prerank and v64 fallback",
        "status": (
            "FULL9_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
            if candidate_complete == len(SCENES)
            else "PARTIAL_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
        ),
        "selection_uses_heldout_metrics": False,
        "caveat": (
            "The guard uses train/policy-val audit fields only, but the rule was formed after "
            "the v82 hard-triad diagnosis. Treat it as a fixed next candidate requiring fresh "
            "blind validation, not as a paper-level promoted endpoint."
        ),
        "inputs": {
            "v64_summary": str(args.v64_summary),
            "v64_selected_root": str(args.v64_selected_root),
            "v82_roots": [str(root) for root in roots],
            "v82_tag_template": str(args.v82_tag_template),
        },
        "policy": {
            "min_selected_alpha": float(args.min_selected_alpha),
            "max_selected_alpha": float(args.max_selected_alpha),
            "min_changed_fraction": float(args.min_changed_fraction),
            "min_policy_val_ssim_gain": float(args.min_policy_val_ssim_gain),
            "min_ssim_positive_fraction": float(args.min_ssim_positive_fraction),
            "min_ssim_min_view_gain": float(args.min_ssim_min_view_gain),
            "min_policy_val_l1_gain": float(args.min_policy_val_l1_gain),
            "min_l1_positive_fraction": float(args.min_l1_positive_fraction),
            "min_l1_min_view_gain": float(args.min_l1_min_view_gain),
            "min_l1_cvar20_view_gain": float(args.min_l1_cvar20_view_gain),
            "rule": "use v82 capacity-prerank only under strict train-policy-val evidence and moderate global alpha; otherwise fallback to v64",
        },
        "candidate_complete_scene_count": candidate_complete,
        "summary": {
            label: summarize(rows, label, args.metric_eps)
            for label in ("v64", "v56", "v52", "no-op", "v48", "v50")
        },
        "rows": rows,
    }
    if args.materialize_root:
        manifest = materialize_selected_tree(
            payload,
            Path(args.materialize_root),
            Path(args.v64_selected_root),
            int(args.max_copy_bytes),
        )
        payload["materialized_root"] = str(args.materialize_root)
        payload["materialized_scene_count"] = int(manifest["scene_count"])
        payload["render_linked_scene_count"] = int(manifest["render_linked_scene_count"])
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if int(payload.get("candidate_complete_scene_count", 0)) == len(SCENES):
        completion_note = (
            "The v82 candidate set is complete for full9. The status remains report-only until "
            "this strict selector is validated on fresh scenes/protocols."
        )
    else:
        completion_note = (
            "The v82 candidate set is not complete for full9. Missing scenes fall back to v64, "
            "so this summary is a materialized selector candidate, not a full v82 rerun."
        )
    lines: list[str] = [
        "# v84 Strict Selector: v82 Capacity-Prerank or v64 Fallback",
        "",
        f"Date: `{payload['date']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This policy does not use held-out metrics for scene selection. It promotes the",
        "v82 capacity-prerank + face-alpha candidate only when train/policy-val evidence",
        "is strong and the global alpha remains moderate; otherwise it preserves the v64",
        "selected fallback.",
        "",
        "## Fixed Guard",
        "",
        f"- selected alpha range: `[{payload['policy']['min_selected_alpha']}, {payload['policy']['max_selected_alpha']}]`",
        f"- minimum target changed fraction: `{payload['policy']['min_changed_fraction']}`",
        f"- minimum policy-val SSIM gain: `{payload['policy']['min_policy_val_ssim_gain']}`",
        f"- minimum policy-val SSIM positive fraction: `{payload['policy']['min_ssim_positive_fraction']}`",
        f"- minimum policy-val SSIM min-view gain: `{payload['policy']['min_ssim_min_view_gain']}`",
        f"- minimum policy-val image-L1 gain: `{payload['policy']['min_policy_val_l1_gain']}`",
        f"- minimum policy-val image-L1 positive fraction: `{payload['policy']['min_l1_positive_fraction']}`",
        f"- minimum policy-val image-L1 min-view gain: `{payload['policy']['min_l1_min_view_gain']}`",
        f"- minimum policy-val image-L1 CVaR20 gain: `{payload['policy']['min_l1_cvar20_view_gain']}`",
        "- otherwise fallback to v64",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v84 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Decisions",
            "",
            "| scene | selected | candidate found | guard | alpha | changed | pval ssim | pval l1 | dPSNR vs v64 | dSSIM vs v64 | dLPIPS vs v64 | reject reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        audit = row["v82_audit"] or {}
        delta = row["comparisons"]["v64"]["delta"]
        reasons = ", ".join(row["guard_reject_reasons"]) if row["guard_reject_reasons"] else "pass"
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {int(row['v82_candidate_found'])} | "
            f"{int(row['guard_passed'])} | {float(audit.get('selected_alpha', 0.0)):.4f} | "
            f"{float(audit.get('changed_fraction', 0.0)):.6f} | "
            f"{float(audit.get('selected_ssim_gain', 0.0)):+.9f} | "
            f"{float(audit.get('selected_image_l1_gain', 0.0)):+.9f} | "
            f"{fmt(delta['PSNR'])} | {fmt(delta['SSIM'])} | {fmt(delta['LPIPS'])} | `{reasons}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v84 is a conservative materialized selector over the v82 capacity-prerank probe.",
            "It captures the verified counter-side signal while rejecting the kitchen/bonsai",
            "raw-policy failures through a fixed train/policy-val guard and v64 fallback.",
            "This is useful for engineering closure and ablation hygiene, but it should not be",
            "presented as a paper-clean breakthrough because the rule was formed after the",
            "hard-triad diagnosis.",
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
    parser.add_argument("--v64_summary", type=Path, default=DEFAULT_V64_SUMMARY)
    parser.add_argument("--v64_selected_root", type=Path, default=DEFAULT_V64_SELECTED_ROOT)
    parser.add_argument("--v82_root", type=Path, action="append", default=None)
    parser.add_argument("--v82_tag_template", default=DEFAULT_V82_TAG)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--materialize_root", type=Path, default=DEFAULT_MATERIALIZE_ROOT)
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    parser.add_argument("--min_selected_alpha", type=float, default=0.5)
    parser.add_argument("--max_selected_alpha", type=float, default=0.5)
    parser.add_argument("--min_changed_fraction", type=float, default=0.001)
    parser.add_argument("--min_policy_val_ssim_gain", type=float, default=2.0e-4)
    parser.add_argument("--min_ssim_positive_fraction", type=float, default=1.0)
    parser.add_argument("--min_ssim_min_view_gain", type=float, default=5.0e-5)
    parser.add_argument("--min_policy_val_l1_gain", type=float, default=2.0e-5)
    parser.add_argument("--min_l1_positive_fraction", type=float, default=0.9)
    parser.add_argument("--min_l1_min_view_gain", type=float, default=-1.0e-6)
    parser.add_argument("--min_l1_cvar20_view_gain", type=float, default=1.0e-6)
    parser.add_argument("--metric_eps", type=float, default=1.0e-7)
    args = parser.parse_args()
    args.v82_root = args.v82_root or list(DEFAULT_V82_ROOTS)
    return args


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
