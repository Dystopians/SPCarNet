#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(results_path: Path, method: str) -> dict[str, float]:
    if not results_path.is_file():
        return {}
    payload = _read_json(results_path)
    row = payload.get(method, {})
    return row if isinstance(row, dict) else {}


def _maybe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _get_nested(payload: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _artifact_scene_root(args: argparse.Namespace, scene: str, manifest_path: Path) -> Path:
    if args.compact_artifact_root:
        return Path(args.compact_artifact_root) / scene
    return manifest_path.parent.parent


def _audit_summary(args: argparse.Namespace, scene: str, manifest_path: Path) -> dict[str, Any]:
    scene_root = _artifact_scene_root(args, scene, manifest_path)
    if args.compact_artifact_root:
        adapter_path = scene_root / "model_audits" / "surface_residual_region_texture_adapter_audit.json"
        target_no_gt_path = scene_root / "model_audits" / "target_evidence_no_gt_audit.json"
    else:
        adapter_path = manifest_path.parent.parent / "model" / "surface_residual_region_texture_adapter_audit.json"
        target_no_gt_path = manifest_path.parent.parent / "target_evidence_no_gt" / "target_evidence_no_gt_audit.json"

    adapter = _maybe_read_json(adapter_path)
    target_no_gt = _maybe_read_json(target_no_gt_path)
    best = _get_nested(adapter, ("policy_val_risk_gate",), {}) or {}
    fit = _get_nested(adapter, ("fit_summary",), {}) or {}
    local_alpha = _get_nested(adapter, ("local_alpha_profile",), {}) or {}
    target_apply = _get_nested(adapter, ("target_apply",), {}) or {}
    return {
        "adapter_audit_path": str(adapter_path) if adapter_path.is_file() else "",
        "target_no_gt_audit_path": str(target_no_gt_path) if target_no_gt_path.is_file() else "",
        "accepted": adapter.get("accepted") if adapter else None,
        "effective_policy": adapter.get("effective_policy", "") if adapter else "",
        "fallback_written": adapter.get("fallback_written") if adapter else None,
        "reject_reason": adapter.get("reject_reason", "") if adapter else "",
        "selected_alpha": adapter.get("selected_alpha") if adapter else None,
        "changed_pixels": target_apply.get("changed_pixels"),
        "changed_fraction": target_apply.get("changed_fraction"),
        "atlas_faces": fit.get("atlas_faces"),
        "candidate_faces": fit.get("candidate_faces"),
        "selected_texture_size": fit.get("selected_texture_size") or fit.get("texture_size"),
        "support_added_faces": fit.get("selected_support_added_faces") or fit.get("support_added_faces"),
        "local_alpha_selected_face_count": local_alpha.get("selected_face_count"),
        "local_alpha_fallback_bin_count": local_alpha.get("fallback_bin_count"),
        "mean_profile_abs_delta_from_fallback": local_alpha.get("mean_profile_abs_delta_from_fallback"),
        "policy_val_selected_positive_view_fraction": best.get("selected_positive_view_fraction"),
        "policy_val_selected_cvar20_view_relative_gain": best.get("selected_cvar20_view_relative_gain"),
        "policy_val_selected_min_view_relative_gain": best.get("selected_min_view_relative_gain"),
        "policy_val_selected_ssim_gain": best.get("selected_ssim_gain"),
        "policy_val_selected_ssim_positive_view_fraction": best.get("selected_ssim_positive_view_fraction"),
        "policy_val_selected_image_l1_gain": best.get("selected_image_l1_gain"),
        "policy_val_selected_image_l1_positive_view_fraction": best.get("selected_image_l1_positive_view_fraction"),
        "target_gt_visible_to_apply": target_no_gt.get("target_gt_visible_to_apply") if target_no_gt else None,
        "target_forbidden_keys_removed_total": target_no_gt.get("forbidden_keys_removed_total") if target_no_gt else None,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble vNext certified residual texture scene manifests.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--method_name", default="ours_26000_vnext_certified_residual_surface_texture")
    parser.add_argument(
        "--compact_artifact_root",
        type=Path,
        default=None,
        help="Optional compact artifact root containing per-scene model_audits copied out before cleanup.",
    )
    args = parser.parse_args()

    manifests = sorted(Path(args.run_root).glob("*/reports/*_vnext_certified_residual_texture_manifest.json"))
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        outputs = manifest.get("outputs", {}) or {}
        results_path = Path(str(outputs.get("results_path", "")))
        scene = manifest.get("scene", manifest_path.parent.parent.name)
        row = {
            "scene": scene,
            "status": manifest.get("status", ""),
            "protocol_audit_passed": bool((manifest.get("protocol_audit", {}) or {}).get("passed", False)),
            "protocol_audit": manifest.get("protocol_audit", {}) or {},
            "results_path": str(results_path),
            "report_path": str(outputs.get("report_path", "")),
            "metrics": _metrics(results_path, str(args.method_name)),
            "audit_summary": _audit_summary(args, str(scene), manifest_path),
            "errors": manifest.get("errors", []),
        }
        rows.append(row)

    completed = [row for row in rows if row["status"] == "COMPLETE" and row["metrics"]]
    mean = {}
    if completed:
        for key in ("PSNR", "SSIM", "LPIPS"):
            values = [float(row["metrics"][key]) for row in completed if key in row["metrics"]]
            if values:
                mean[key] = sum(values) / len(values)
    accepted = [row for row in completed if row.get("audit_summary", {}).get("accepted") is True]
    fallback = [row for row in completed if row.get("audit_summary", {}).get("accepted") is False]
    changed_values = [
        float(row["audit_summary"]["changed_fraction"])
        for row in completed
        if row.get("audit_summary", {}).get("changed_fraction") is not None
    ]
    payload = {
        "schema_version": 1,
        "run_root": str(args.run_root),
        "compact_artifact_root": str(args.compact_artifact_root) if args.compact_artifact_root else "",
        "method_name": str(args.method_name),
        "scene_count": int(len(rows)),
        "completed_metric_scene_count": int(len(completed)),
        "accepted_scene_count": int(len(accepted)),
        "fallback_or_rejected_scene_count": int(len(fallback)),
        "mean_changed_fraction": (sum(changed_values) / len(changed_values)) if changed_values else None,
        "mean_metrics": mean,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# vNext Certified Residual Surface Texture Summary",
        "",
        f"- run root: `{args.run_root}`",
        f"- compact artifact root: `{args.compact_artifact_root or ''}`",
        f"- scenes found: `{len(rows)}`",
        f"- completed metric scenes: `{len(completed)}`",
        f"- accepted scenes: `{len(accepted)}`",
        f"- fallback/rejected scenes: `{len(fallback)}`",
        f"- mean changed fraction: `{_fmt(payload.get('mean_changed_fraction'), 9)}`",
        f"- mean PSNR: `{_fmt(mean.get('PSNR'))}`",
        f"- mean SSIM: `{_fmt(mean.get('SSIM'))}`",
        f"- mean LPIPS: `{_fmt(mean.get('LPIPS'))}`",
        "",
        "| scene | status | protocol | accepted | policy | alpha | changed frac | policy gain | SSIM gain | L1 gain | PSNR | SSIM | LPIPS | report |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics", {}) or {}
        audit = row.get("audit_summary", {}) or {}
        lines.append(
            "| {scene} | {status} | {protocol} | {accepted} | {policy} | {alpha} | {changed} | {gain} | {ssim_gain} | {l1_gain} | {psnr} | {ssim} | {lpips} | `{report}` |".format(
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                protocol=row.get("protocol_audit_passed", False),
                accepted=audit.get("accepted", ""),
                policy=audit.get("effective_policy", ""),
                alpha=_fmt(audit.get("selected_alpha")),
                changed=_fmt(audit.get("changed_fraction"), 9),
                gain=_fmt(audit.get("policy_val_selected_cvar20_view_relative_gain"), 9),
                ssim_gain=_fmt(audit.get("policy_val_selected_ssim_gain"), 9),
                l1_gain=_fmt(audit.get("policy_val_selected_image_l1_gain"), 9),
                psnr=_fmt(metrics.get("PSNR")),
                ssim=_fmt(metrics.get("SSIM")),
                lpips=_fmt(metrics.get("LPIPS")),
                report=row.get("report_path", ""),
            )
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
