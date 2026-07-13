#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key] - b[key]) for key in METRICS}


def strict_win(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] > eps and delta["SSIM"] > eps and delta["LPIPS"] < -eps


def nonregressive(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] >= -eps and delta["SSIM"] >= -eps and delta["LPIPS"] <= eps


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):+.{digits}f}"


def metric_dict(row: dict[str, Any]) -> dict[str, float]:
    return {key: float(row["metrics"][key]) for key in METRICS}


def v51_audit_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    policy = audit.get("policy_val", {}) or {}
    best = policy.get("best", {}) or {}
    fill = policy.get("fill_mode_selection", {}) or {}
    selected_support = str(fill.get("selected_support_mode", ""))
    selected_texture = fill.get("selected_texture_size")
    selected_fill = str(fill.get("selected_fill_mode", ""))
    selected_added = 0
    for candidate in fill.get("score_order", []) or []:
        if (
            str(candidate.get("support_mode", "")) == selected_support
            and candidate.get("texture_size") == selected_texture
            and str(candidate.get("fill_mode", "")) == selected_fill
        ):
            selected_added = int(candidate.get("support_added_faces", 0))
            break
    return {
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "selected_support_mode": selected_support,
        "selected_support_added_faces": int(selected_added),
        "selected_texture_size": int(selected_texture or 0),
        "selected_fill": selected_fill,
        "changed_fraction": float((audit.get("target_apply", {}) or {}).get("changed_fraction", 0.0)),
        "policy_val_relative_gain": float(best.get("relative_gain", 0.0)),
        "policy_val_ssim_gain": float(best.get("ssim_gain", 0.0)),
        "policy_val_image_l1_gain": float(best.get("image_l1_gain", 0.0)),
        "policy_val_image_l1_positive_view_fraction": float(best.get("image_l1_positive_view_fraction", 0.0)),
        "policy_val_image_l1_min_view_gain": float(best.get("image_l1_min_view_gain", 0.0)),
        "policy_val_image_l1_cvar20_view_gain": float(best.get("image_l1_cvar20_view_gain", 0.0)),
    }


def should_promote_v51(
    v48_audit: dict[str, Any],
    v51_audit: dict[str, Any],
    support_cap: int,
    min_v51_ssim_gain: float,
    min_v51_relative_gain: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    v48_added = int(v48_audit.get("selected_support_added_faces", 0))
    v51_added = int(v51_audit.get("selected_support_added_faces", 0))
    if not bool(v48_audit.get("accepted", False)):
        reasons.append("v48_not_accepted")
    if v48_added < support_cap:
        reasons.append(f"v48_support_below_cap:{v48_added}<{support_cap}")
    if not bool(v51_audit.get("accepted", False)):
        reasons.append("v51_not_accepted")
    if str(v51_audit.get("effective_policy", "")) != "accepted_atlas":
        reasons.append(f"v51_policy_not_accepted_atlas:{v51_audit.get('effective_policy', '')}")
    if v51_added <= v48_added:
        reasons.append(f"v51_not_larger_support:{v51_added}<={v48_added}")
    if float(v51_audit.get("policy_val_ssim_gain", 0.0)) < min_v51_ssim_gain:
        reasons.append(
            f"v51_ssim_gain_below_threshold:{float(v51_audit.get('policy_val_ssim_gain', 0.0)):.9f}"
        )
    if float(v51_audit.get("policy_val_relative_gain", 0.0)) < min_v51_relative_gain:
        reasons.append(
            f"v51_relative_gain_below_threshold:{float(v51_audit.get('policy_val_relative_gain', 0.0)):.9f}"
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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v52 Capacity-Aware v48/v51 Policy Summary",
        "",
        "Status: `REPORT_ONLY_EFFECTIVE_POLICY`.",
        "",
        "This policy does not use held-out metrics for selection. It promotes v51 only when",
        "v48 reached the fixed support cap and v51 was accepted by train/policy-val evidence",
        "with a larger support footprint. The table below then reports held-out metrics for",
        "the already materialized selected rows.",
        "",
        "## Fixed Selection Rule",
        "",
        f"- v48 support cap threshold: `{payload['policy']['support_cap']}` extra faces",
        f"- minimum v51 policy-val SSIM gain: `{payload['policy']['min_v51_ssim_gain']}`",
        f"- minimum v51 policy-val relative gain: `{payload['policy']['min_v51_relative_gain']}`",
        "- promote to v51 only if v48 is accepted, v48 hits the support cap, v51 is accepted, and v51 uses larger support",
        "- otherwise keep v48",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v52 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'], 9)} | "
            f"{fmt(stats['mean_dSSIM'], 9)} | {fmt(stats['mean_dLPIPS'], 9)} |"
        )
    lines.extend(
        [
            "",
            "## Per-Scene Decisions",
            "",
            "| scene | selected | decision | support | +faces | texture | fill | changed | PSNR | SSIM | LPIPS | vs no-op | vs v48 | vs v50 |",
            "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        audit = row["selected_audit"]
        metrics = row["metrics"]

        def mark(label: str) -> str:
            comp = row["comparisons"][label]
            delta = comp["delta"]
            status = "S" if comp["strict_win"] else ("N" if comp["nonregressive_or_tie"] else "-")
            return f"{fmt(delta['PSNR'])}/{fmt(delta['SSIM'], 8)}/{fmt(delta['LPIPS'], 8)} {status}"

        lines.append(
            f"| {row['scene']} | {row['selected_source']} | {row['decision']} | "
            f"{audit.get('selected_support_mode', '')} | {audit.get('selected_support_added_faces', 0)} | "
            f"{audit.get('selected_texture_size', 0)} | {audit.get('selected_fill', '')} | "
            f"{100.0 * float(audit.get('changed_fraction', 0.0)):.4f}% | "
            f"{metrics['PSNR']:.6f} | {metrics['SSIM']:.8f} | {metrics['LPIPS']:.8f} | "
            f"{mark('no-op')} | {mark('v48')} | {mark('v50')} |"
        )
    lines.extend(["", "## Rejected v51 Reasons", ""])
    for row in payload["rows"]:
        if row["selected_source"] == "v51":
            continue
        lines.append(f"- `{row['scene']}` kept v48: `{', '.join(row['v51_reject_reasons'])}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v52 is the first fixed capacity-aware policy that captures the v51 gains on cap-hit",
            "scenes without sacrificing v48's auto-capacity/auto-fill behavior elsewhere. It is",
            "still an effective policy over already materialized v48/v51 outputs; the next",
            "engineering step is to wire this same rule into the run/eval launcher so selected",
            "renders and metrics can be produced or refreshed in one command.",
            "",
        ]
    )
    if payload.get("materialized_root"):
        lines.extend(
            [
                "## Materialized Small Artifacts",
                "",
                f"- selected small-artifact root: `{payload['materialized_root']}`",
                f"- materialized scene count: `{payload.get('materialized_scene_count', 0)}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_if_small(src: Path, dst: Path, max_bytes: int) -> bool:
    if not src.is_file():
        return False
    if src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _valid_dir(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_dir() else None


def find_render_gt_dirs(source_dir: Path, audit_path: Path) -> tuple[Path | None, Path | None, str]:
    audit: dict[str, Any] = {}
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
        fallback_render = source_model / "test" / base_method / "renders"
        fallback_gt = source_model / "test" / base_method / "gt"
        if fallback_render.is_dir() and fallback_gt.is_dir():
            return fallback_render, fallback_gt, "source_model_base_method"

    resolved_source = source_dir.resolve()
    render_candidates = sorted(resolved_source.glob("test/*/renders"))
    for candidate in render_candidates:
        paired_gt = candidate.parent / "gt"
        if candidate.is_dir() and paired_gt.is_dir():
            return candidate, paired_gt, "source_dir_test_glob"
    return None, None, "not_found"


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def materialize_selected_tree(root: Path, rows: list[dict[str, Any]], max_copy_bytes: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    small_names = (
        "results.json",
        "surface_residual_region_texture_adapter_audit.json",
        "surface_residual_region_texture_adapter_audit.md",
        "topology_audit.json",
        "topology_audit.md",
    )
    for row in rows:
        scene = str(row["scene"])
        scene_dir = root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(row["selected_source_dir"])
        copied_files: list[str] = []
        audit_path = source_dir / "surface_residual_region_texture_adapter_audit.json"
        for name in small_names:
            if copy_if_small(source_dir / name, scene_dir / name, max_copy_bytes):
                copied_files.append(name)
        log_path = source_dir / f"apply_metrics_{scene}.log"
        if copy_if_small(log_path, scene_dir / log_path.name, max_copy_bytes):
            copied_files.append(log_path.name)
        render_dir, gt_dir, render_source = find_render_gt_dirs(source_dir, audit_path)
        render_linked = False
        gt_linked = False
        if render_dir is not None:
            replace_symlink(scene_dir / "renders", render_dir)
            render_linked = True
        if gt_dir is not None:
            replace_symlink(scene_dir / "gt", gt_dir)
            gt_linked = True
        source_note = {
            "scene": scene,
            "selected_source": row["selected_source"],
            "decision": row["decision"],
            "source_dir": str(source_dir),
            "copied_files": copied_files,
            "render_source": render_source,
            "render_dir": "" if render_dir is None else str(render_dir),
            "gt_dir": "" if gt_dir is None else str(gt_dir),
            "render_symlink": str(scene_dir / "renders") if render_linked else "",
            "gt_symlink": str(scene_dir / "gt") if gt_linked else "",
            "render_linked": render_linked,
            "gt_linked": gt_linked,
            "metrics": row["metrics"],
            "selection_uses_heldout_metrics": False,
        }
        (scene_dir / "selection_manifest.json").write_text(
            json.dumps(source_note, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        copied.append(source_note)
    manifest = {
        "materialized_root": str(root),
        "scene_count": len(copied),
        "max_copy_bytes": int(max_copy_bytes),
        "render_linked_scene_count": int(sum(1 for item in copied if item["render_linked"] and item["gt_linked"])),
        "scenes": copied,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed v52 capacity-aware v48/v51 effective policy summary.")
    parser.add_argument(
        "--v48_summary",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.json",
    )
    parser.add_argument(
        "--v50_summary",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623/v50/v50_full9_summary.json",
    )
    parser.add_argument(
        "--v51_summary",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_full9_summary.json",
    )
    parser.add_argument(
        "--v51_artifact_root",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_small_artifacts_20260623",
    )
    parser.add_argument("--support_cap", type=int, default=2048)
    parser.add_argument("--min_v51_ssim_gain", type=float, default=5.0e-5)
    parser.add_argument("--min_v51_relative_gain", type=float, default=0.0)
    parser.add_argument(
        "--output_json",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.json",
    )
    parser.add_argument(
        "--output_md",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md",
    )
    parser.add_argument(
        "--materialize_root",
        default="",
        help="Optional directory for copying selected small artifacts into a canonical v52 tree.",
    )
    parser.add_argument("--max_copy_bytes", type=int, default=5_000_000)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    args = parser.parse_args()

    v48 = read_json(Path(args.v48_summary))
    v50 = read_json(Path(args.v50_summary))
    v51 = read_json(Path(args.v51_summary))
    v48_rows = {row["scene"]: row for row in v48["rows"]}
    v50_rows = {row["scene"]: row for row in v50["rows"]}
    v51_rows = {row["scene"]: row for row in v51["rows"]}
    scenes = list(v48_rows.keys())
    artifact_root = Path(args.v51_artifact_root)

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        v48_row = v48_rows[scene]
        v50_row = v50_rows[scene]
        v51_row = v51_rows[scene]
        v48_audit = dict(v48_row.get("audit", {}) or {})
        v51_audit = v51_audit_summary(artifact_root / scene / "surface_residual_region_texture_adapter_audit.json")
        promote, reject_reasons = should_promote_v51(
            v48_audit,
            v51_audit,
            int(args.support_cap),
            float(args.min_v51_ssim_gain),
            float(args.min_v51_relative_gain),
        )
        selected_source = "v51" if promote else "v48"
        selected_row = v51_row if promote else v48_row
        selected_audit = v51_audit if promote else v48_audit
        selected_source_dir = artifact_root / scene if promote else Path(str(v48_row["method_dir"]))
        selected_metrics = metric_dict(selected_row)
        bases = {
            "no-op": {key: float(v48_row["comparisons"]["no-op"]["metrics"][key]) for key in METRICS},
            "v48": metric_dict(v48_row),
            "v50": metric_dict(v50_row),
        }
        comparisons: dict[str, Any] = {}
        for label, base_metrics in bases.items():
            delta = metric_delta(selected_metrics, base_metrics)
            comparisons[label] = {
                "delta": delta,
                "strict_win": strict_win(delta, float(args.eps)),
                "nonregressive_or_tie": nonregressive(delta, float(args.eps)),
            }
        rows.append(
            {
                "scene": scene,
                "selected_source": selected_source,
                "selected_source_dir": str(selected_source_dir),
                "decision": "promote_v51_capacity_ladder" if promote else "keep_v48_auto_policy",
                "selection_uses_heldout_metrics": False,
                "v51_reject_reasons": reject_reasons,
                "metrics": selected_metrics,
                "selected_audit": selected_audit,
                "v48_audit": v48_audit,
                "v51_audit": v51_audit,
                "comparisons": comparisons,
            }
        )

    summary = {label: summarize(rows, label, float(args.eps)) for label in ("no-op", "v48", "v50")}
    payload = {
        "method": "v52 capacity-aware v48/v51 fixed policy",
        "status": "REPORT_ONLY_EFFECTIVE_POLICY",
        "selection_uses_heldout_metrics": False,
        "policy": {
            "support_cap": int(args.support_cap),
            "min_v51_ssim_gain": float(args.min_v51_ssim_gain),
            "min_v51_relative_gain": float(args.min_v51_relative_gain),
            "rule": "promote v51 only when v48 accepted and hit support cap, v51 accepted_atlas, v51 support is larger, and policy-val gates pass",
        },
        "inputs": {
            "v48_summary": str(args.v48_summary),
            "v50_summary": str(args.v50_summary),
            "v51_summary": str(args.v51_summary),
            "v51_artifact_root": str(args.v51_artifact_root),
        },
        "scenes": scenes,
        "rows": rows,
        "summary": summary,
    }
    materialized: dict[str, Any] | None = None
    if args.materialize_root:
        materialized = materialize_selected_tree(Path(args.materialize_root), rows, int(args.max_copy_bytes))
        payload["status"] = "REPORT_ONLY_EFFECTIVE_POLICY_WITH_SELECTED_SMALL_ARTIFACTS"
        payload["materialized_root"] = str(args.materialize_root)
        payload["materialized_scene_count"] = int(materialized["scene_count"])
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(output_md, payload)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "materialized_root": str(args.materialize_root or ""),
                "materialized_scene_count": None if materialized is None else materialized["scene_count"],
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
