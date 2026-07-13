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
DEFAULT_V56_SUMMARY = DEFAULT_ROOT / "v56_face_alpha_guard_full9_summary.json"
DEFAULT_V56_SELECTED_ROOT = DEFAULT_ROOT / "v56_face_alpha_guard_selected_full9"
DEFAULT_OUTPUT_JSON = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_full9_summary.json"
DEFAULT_OUTPUT_MD = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_full9_summary.md"
DEFAULT_MATERIALIZE_ROOT = DEFAULT_ROOT / "v64_bin_alpha_auto_policy_selected_full9"
DEFAULT_V63B_ROOTS = (
    Path("/dev/shm/peilincai_spcarnet_v63b_bin_alpha_full9_20260624"),
    Path("/dev/shm/peilincai_spcarnet_v63b_bin_alpha_counter_20260624"),
    Path("/dev/shm/peilincai_spcarnet_v63b_bin_alpha_kitchen_20260624"),
)
DEFAULT_V63B_TAG = (
    "v63b_bin_alpha_max035_min32_pos05_shrink128_profile8192_"
    "normal_camera_oodz25_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter"
)


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


def baseline_from_v56(row: dict[str, Any], label: str) -> dict[str, float]:
    metrics = {key: float(row["selected_metrics"][key]) for key in METRICS}
    delta = {key: float(row["comparisons"][label]["delta"][key]) for key in METRICS}
    return {key: float(metrics[key] - delta[key]) for key in METRICS}


def find_candidate_dir(roots: list[Path], scene: str, tag: str) -> tuple[Path, bool]:
    expected = [root / f"{scene}_{tag}" for root in roots]
    for candidate in expected:
        if (candidate / "results.json").is_file() or (
            candidate / "surface_residual_region_texture_adapter_audit.json"
        ).is_file():
            return candidate, True
    for root in roots:
        matches = sorted(root.glob(f"**/{scene}_{tag}"))
        for candidate in matches:
            if (candidate / "results.json").is_file() or (
                candidate / "surface_residual_region_texture_adapter_audit.json"
            ).is_file():
                return candidate, True
    return expected[0], False


def summarize_v63b_audit(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    local = audit.get("local_alpha_profile", {}) or {}
    policy = audit.get("policy_val", {}) or {}
    best = policy.get("best", {}) or {}
    target = audit.get("target_apply", {}) or {}
    view_basis = (audit.get("fit_summary", {}) or {}).get("view_conditioned_basis", {}) or {}
    return {
        "audit_path": str(path),
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0)),
        "changed_fraction": float(target.get("changed_fraction", 0.0)),
        "local_alpha_enabled": bool(local.get("enabled", False)),
        "local_alpha_mode": str(local.get("mode", "")),
        "bin_alpha_count": int(local.get("bin_alpha_count", 0) or 0),
        "candidate_bin_count": int(local.get("candidate_bin_count", 0) or 0),
        "fallback_bin_count": int(local.get("fallback_bin_count", 0) or 0),
        "fallback_alpha": float(local.get("fallback_alpha", 0.0) or 0.0),
        "max_alpha": float(local.get("max_alpha", 0.0) or 0.0),
        "policy_val_alpha": float(best.get("alpha", 0.0) or 0.0),
        "policy_val_relative_gain": float(best.get("relative_gain", 0.0) or 0.0),
        "policy_val_positive_view_fraction": float(best.get("positive_view_fraction", 0.0) or 0.0),
        "policy_val_ssim_gain": float(best.get("ssim_gain", 0.0) or 0.0),
        "policy_val_ssim_positive_view_fraction": float(
            best.get("ssim_positive_view_fraction", 0.0) or 0.0
        ),
        "policy_val_image_l1_gain": float(best.get("image_l1_gain", 0.0) or 0.0),
        "policy_val_image_l1_positive_view_fraction": float(
            best.get("image_l1_positive_view_fraction", 0.0) or 0.0
        ),
        "policy_val_image_l1_min_view_gain": float(best.get("image_l1_min_view_gain", 0.0) or 0.0),
        "policy_val_image_l1_cvar20_view_gain": float(
            best.get("image_l1_cvar20_view_gain", 0.0) or 0.0
        ),
        "view_basis_effective_mode": str(view_basis.get("effective_mode", view_basis.get("mode", "none"))),
        "view_basis_supported_bin_fraction": float(view_basis.get("supported_bin_fraction", 0.0) or 0.0),
        "view_basis_ood_mode": str(view_basis.get("ood_mode", "none")),
    }


def should_use_v63b(audit: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not audit:
        reasons.append("missing_v63b_audit")
        return False, reasons
    if not audit.get("accepted", False):
        reasons.append("v63b_not_accepted")
    if audit.get("effective_policy") != "accepted_atlas":
        reasons.append(f"v63b_not_accepted_atlas:{audit.get('effective_policy')}")
    if not audit.get("local_alpha_enabled", False):
        reasons.append("local_alpha_disabled")
    if audit.get("local_alpha_mode") != "policy_val_bin_alpha":
        reasons.append(f"local_alpha_mode_not_bin:{audit.get('local_alpha_mode')}")
    if int(audit.get("bin_alpha_count", 0)) < args.min_bin_alpha_count:
        reasons.append(f"bin_alpha_count_below:{audit.get('bin_alpha_count')}<{args.min_bin_alpha_count}")
    if int(audit.get("bin_alpha_count", 0)) > args.max_bin_alpha_count:
        reasons.append(f"bin_alpha_count_above:{audit.get('bin_alpha_count')}>{args.max_bin_alpha_count}")
    if float(audit.get("selected_alpha", 0.0)) < args.min_selected_alpha:
        reasons.append(f"selected_alpha_below:{audit.get('selected_alpha')}<{args.min_selected_alpha}")
    if float(audit.get("selected_alpha", 0.0)) > args.max_selected_alpha:
        reasons.append(f"selected_alpha_above:{audit.get('selected_alpha')}>{args.max_selected_alpha}")
    if float(audit.get("policy_val_relative_gain", 0.0)) < args.min_policy_val_relative_gain:
        reasons.append(
            "relative_gain_below:"
            f"{audit.get('policy_val_relative_gain')}<{args.min_policy_val_relative_gain}"
        )
    if float(audit.get("policy_val_ssim_gain", 0.0)) < args.min_policy_val_ssim_gain:
        reasons.append(f"ssim_gain_below:{audit.get('policy_val_ssim_gain')}<{args.min_policy_val_ssim_gain}")
    if float(audit.get("policy_val_ssim_positive_view_fraction", 0.0)) < args.min_ssim_positive_fraction:
        reasons.append(
            "ssim_positive_fraction_below:"
            f"{audit.get('policy_val_ssim_positive_view_fraction')}<{args.min_ssim_positive_fraction}"
        )
    if float(audit.get("policy_val_image_l1_gain", 0.0)) < args.min_policy_val_l1_gain:
        reasons.append(
            f"l1_gain_below:{audit.get('policy_val_image_l1_gain')}<{args.min_policy_val_l1_gain}"
        )
    if float(audit.get("policy_val_image_l1_positive_view_fraction", 0.0)) < args.min_l1_positive_fraction:
        reasons.append(
            "l1_positive_fraction_below:"
            f"{audit.get('policy_val_image_l1_positive_view_fraction')}<{args.min_l1_positive_fraction}"
        )
    if float(audit.get("policy_val_image_l1_min_view_gain", 0.0)) < args.min_l1_min_view_gain:
        reasons.append(
            "l1_min_view_gain_below:"
            f"{audit.get('policy_val_image_l1_min_view_gain')}<{args.min_l1_min_view_gain}"
        )
    return not reasons, reasons


def summarize(rows: list[dict[str, Any]], label: str, eps: float) -> dict[str, Any]:
    deltas = [row["comparisons"][label]["delta"] for row in rows]
    count = len(deltas)
    return {
        "scene_count": int(count),
        "strict_wins": int(sum(strict_win(delta, eps) for delta in deltas)),
        "nonregressive_or_tie": int(sum(nonregressive(delta, eps) for delta in deltas)),
        "mean_dPSNR": float(sum(delta["PSNR"] for delta in deltas) / count),
        "mean_dSSIM": float(sum(delta["SSIM"] for delta in deltas) / count),
        "mean_dLPIPS": float(sum(delta["LPIPS"] for delta in deltas) / count),
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
        source_model = _valid_dir(audit.get("source_model"))
        base_method = str(audit.get("base_method_name", ""))
        if source_model is not None and base_method:
            render_dir = source_model / "test" / base_method / "renders"
            gt_dir = source_model / "test" / base_method / "gt"
            if render_dir.is_dir() and gt_dir.is_dir():
                return render_dir, gt_dir, "source_model_base_method"
    render_candidates = sorted(source_dir.resolve().glob("test/*/renders"))
    for render_dir in render_candidates:
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
    summary: dict[str, Any],
    selected_root: Path,
    v56_selected_root: Path,
    max_copy_bytes: int,
) -> dict[str, Any]:
    selected_root.mkdir(parents=True, exist_ok=True)
    small_names = (
        "results.json",
        "surface_residual_region_texture_adapter_audit.json",
        "surface_residual_region_texture_adapter_audit.md",
        "topology_audit.json",
        "topology_audit.md",
    )
    records: list[dict[str, Any]] = []
    for row in summary["rows"]:
        scene = str(row["scene"])
        scene_root = selected_root / scene
        scene_root.mkdir(parents=True, exist_ok=True)
        if row["selected_source"] == "v63b_bin_alpha":
            source_dir = Path(str(row["v63b_candidate_dir"]))
        else:
            source_dir = v56_selected_root / scene
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
            "v56_metrics": row["v56_metrics"],
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
        "status": "V64_SELECTED_TREE_MATERIALIZED",
        "scenes": records,
    }
    (selected_root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    v56 = read_json(args.v56_summary)
    v56_rows = {str(row["scene"]): row for row in v56["rows"]}
    roots = [Path(root) for root in args.v63b_root]
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        v56_row = v56_rows[scene]
        v56_metrics = {key: float(v56_row["selected_metrics"][key]) for key in METRICS}
        candidate_dir, candidate_found = find_candidate_dir(roots, scene, str(args.v63b_tag))
        result_path = candidate_dir / "results.json"
        audit_path = candidate_dir / "surface_residual_region_texture_adapter_audit.json"
        candidate_method = ""
        candidate_metrics: dict[str, float] | None = None
        candidate_audit: dict[str, Any] = {}
        if audit_path.is_file():
            candidate_audit = summarize_v63b_audit(audit_path)
        if result_path.is_file():
            candidate_method, candidate_metrics = first_method_metrics(result_path)
        guard_passed, reject_reasons = should_use_v63b(candidate_audit, args)
        use_v63b = bool(candidate_metrics is not None and guard_passed)
        selected_metrics = candidate_metrics if use_v63b and candidate_metrics is not None else v56_metrics
        selected_source = "v63b_bin_alpha" if use_v63b else "v56_fallback"
        baselines = {"v56": v56_metrics, "v52": {key: float(v56_row["v52_metrics"][key]) for key in METRICS}}
        for label in ("no-op", "v48", "v50"):
            baselines[label] = baseline_from_v56(v56_row, label)
        comparisons = {}
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
                "v56_metrics": v56_metrics,
                "v52_metrics": baselines["v52"],
                "v63b_method": candidate_method,
                "v63b_metrics": candidate_metrics,
                "v63b_audit": candidate_audit,
                "v63b_candidate_found": candidate_found,
                "v63b_candidate_dir": str(candidate_dir),
                "v63b_result_path": str(result_path),
                "guard_passed": guard_passed,
                "guard_reject_reasons": reject_reasons,
                "comparisons": comparisons,
            }
        )
    candidate_complete = int(sum(bool(row["v63b_metrics"] is not None and row["v63b_audit"]) for row in rows))
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "method": "v64 fixed auto policy over v63b bin-alpha and v56 fallback",
        "status": (
            "FULL9_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
            if candidate_complete == len(SCENES)
            else "PARTIAL_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY"
        ),
        "selection_uses_heldout_metrics": False,
        "caveat": (
            "The guard uses train/policy-val audit fields only. It was designed after v63b "
            "counter/kitchen probes, so it must be validated on fresh full9 candidates before "
            "paper-level promotion."
        ),
        "inputs": {
            "v56_summary": str(args.v56_summary),
            "v56_selected_root": str(args.v56_selected_root),
            "v63b_roots": [str(root) for root in roots],
            "v63b_tag": str(args.v63b_tag),
        },
        "policy": {
            "min_bin_alpha_count": int(args.min_bin_alpha_count),
            "max_bin_alpha_count": int(args.max_bin_alpha_count),
            "min_selected_alpha": float(args.min_selected_alpha),
            "max_selected_alpha": float(args.max_selected_alpha),
            "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
            "min_policy_val_ssim_gain": float(args.min_policy_val_ssim_gain),
            "min_ssim_positive_fraction": float(args.min_ssim_positive_fraction),
            "min_policy_val_l1_gain": float(args.min_policy_val_l1_gain),
            "min_l1_positive_fraction": float(args.min_l1_positive_fraction),
            "min_l1_min_view_gain": float(args.min_l1_min_view_gain),
            "rule": "use v63b bin-alpha only under strong train-policy-val magnitude evidence; otherwise fallback to v56",
        },
        "candidate_complete_scene_count": candidate_complete,
        "summary": {label: summarize(rows, label, args.metric_eps) for label in ("v56", "v52", "no-op", "v48", "v50")},
        "rows": rows,
    }
    if args.materialize_root:
        manifest = materialize_selected_tree(
            payload,
            Path(args.materialize_root),
            Path(args.v56_selected_root),
            int(args.max_copy_bytes),
        )
        payload["materialized_root"] = str(args.materialize_root)
        payload["materialized_scene_count"] = int(manifest["scene_count"])
        payload["render_linked_scene_count"] = int(manifest["render_linked_scene_count"])
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    if int(payload.get("candidate_complete_scene_count", 0)) == len(SCENES):
        completion_note = (
            "The v63b candidate set is complete for full9. The status remains report-only "
            "because the rule was designed after the initial counter/kitchen probes and still "
            "needs fresh blind/long-run validation before paper-level promotion."
        )
    else:
        completion_note = (
            "The v63b candidate set is not complete for full9 yet. Treat the summary as partial "
            "until every scene has a candidate audit and result row."
        )
    lines: list[str] = [
        "# v64 Fixed Auto Policy: v63b Bin-Alpha or v56 Fallback",
        "",
        f"Date: `{payload['date']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This policy does not use held-out metrics for scene selection. It promotes v63b",
        "only when the v63b train/policy-val audit shows strong bin-level residual magnitude",
        "evidence; otherwise it falls back to the already materialized v56 selected policy.",
        "",
        "## Fixed Guard",
        "",
        f"- minimum bin-alpha count: `{payload['policy']['min_bin_alpha_count']}`",
        f"- maximum bin-alpha count: `{payload['policy']['max_bin_alpha_count']}`",
        f"- selected alpha range: `[{payload['policy']['min_selected_alpha']}, {payload['policy']['max_selected_alpha']}]`",
        f"- minimum policy-val relative gain: `{payload['policy']['min_policy_val_relative_gain']}`",
        f"- minimum policy-val SSIM gain: `{payload['policy']['min_policy_val_ssim_gain']}`",
        f"- minimum policy-val SSIM positive fraction: `{payload['policy']['min_ssim_positive_fraction']}`",
        f"- minimum policy-val image-L1 gain: `{payload['policy']['min_policy_val_l1_gain']}`",
        f"- minimum policy-val image-L1 positive fraction: `{payload['policy']['min_l1_positive_fraction']}`",
        f"- minimum policy-val image-L1 min-view gain: `{payload['policy']['min_l1_min_view_gain']}`",
        "- otherwise fallback to v56",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v64 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'])} | {fmt(stats['mean_dLPIPS'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Decisions",
            "",
            "| scene | selected | candidate found | guard | bins | alpha | pval ssim | pval l1 | dPSNR vs v56 | dSSIM vs v56 | dLPIPS vs v56 | reject reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        audit = row["v63b_audit"] or {}
        delta = row["comparisons"]["v56"]["delta"]
        reasons = ", ".join(row["guard_reject_reasons"]) if row["guard_reject_reasons"] else "pass"
        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {int(row['v63b_candidate_found'])} | "
            f"{int(row['guard_passed'])} | {int(audit.get('bin_alpha_count', 0))} | "
            f"{float(audit.get('selected_alpha', 0.0)):.4f} | "
            f"{float(audit.get('policy_val_ssim_gain', 0.0)):+.9f} | "
            f"{float(audit.get('policy_val_image_l1_gain', 0.0)):+.9f} | "
            f"{fmt(delta['PSNR'])} | {fmt(delta['SSIM'])} | {fmt(delta['LPIPS'])} | `{reasons}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v64 is a fixed auto policy candidate. It upgrades v63b from a scene-specific probe",
            "into a deployable decision rule: use bin-level residual magnitude calibration only",
            "when train/policy-val evidence is strong, otherwise preserve the v56 fallback. This",
            "keeps the known counter failure out of the selected endpoint and admits kitchen when",
            "its policy-val signal is much stronger.",
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
    parser.add_argument("--v56_summary", type=Path, default=DEFAULT_V56_SUMMARY)
    parser.add_argument("--v56_selected_root", type=Path, default=DEFAULT_V56_SELECTED_ROOT)
    parser.add_argument("--v63b_root", type=Path, action="append", default=None)
    parser.add_argument("--v63b_tag", default=DEFAULT_V63B_TAG)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--materialize_root", type=Path, default=DEFAULT_MATERIALIZE_ROOT)
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    parser.add_argument("--min_bin_alpha_count", type=int, default=32)
    parser.add_argument("--max_bin_alpha_count", type=int, default=256)
    parser.add_argument("--min_selected_alpha", type=float, default=0.5)
    parser.add_argument("--max_selected_alpha", type=float, default=1.0)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.05)
    parser.add_argument("--min_policy_val_ssim_gain", type=float, default=3.0e-4)
    parser.add_argument("--min_ssim_positive_fraction", type=float, default=1.0)
    parser.add_argument("--min_policy_val_l1_gain", type=float, default=4.0e-5)
    parser.add_argument("--min_l1_positive_fraction", type=float, default=1.0)
    parser.add_argument("--min_l1_min_view_gain", type=float, default=1.0e-5)
    parser.add_argument("--metric_eps", type=float, default=1.0e-7)
    args = parser.parse_args()
    args.v63b_root = args.v63b_root or list(DEFAULT_V63B_ROOTS)
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
