#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
METRICS = ("PSNR", "SSIM", "LPIPS")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _metric_delta(method: dict[str, Any], base: dict[str, Any]) -> dict[str, float | None]:
    return {
        "dPSNR": _none_sub(method.get("PSNR"), base.get("PSNR")),
        "dSSIM": _none_sub(method.get("SSIM"), base.get("SSIM")),
        "dLPIPS": _none_sub(method.get("LPIPS"), base.get("LPIPS")),
    }


def _none_sub(left: Any, right: Any) -> float | None:
    lval = _float(left)
    rval = _float(right)
    if lval is None or rval is None:
        return None
    return lval - rval


def _strict_rgb_win(delta: dict[str, float | None]) -> bool:
    return (
        delta.get("dPSNR") is not None
        and delta.get("dSSIM") is not None
        and delta.get("dLPIPS") is not None
        and float(delta["dPSNR"]) > 0.0
        and float(delta["dSSIM"]) > 0.0
        and float(delta["dLPIPS"]) < 0.0
    )


def _per_view_strict_counts(
    clean_per_view: dict[str, Any] | None,
    method_per_view: dict[str, Any] | None,
    clean_method: str,
    method_name: str,
) -> dict[str, Any]:
    if not clean_per_view or not method_per_view:
        return {"available": False}
    clean = clean_per_view.get(clean_method)
    method = method_per_view.get(method_name)
    if not clean or not method:
        return {"available": False}
    common = sorted(
        set(clean.get("PSNR", {}).keys())
        & set(clean.get("SSIM", {}).keys())
        & set(clean.get("LPIPS", {}).keys())
        & set(method.get("PSNR", {}).keys())
        & set(method.get("SSIM", {}).keys())
        & set(method.get("LPIPS", {}).keys())
    )
    rows = []
    strict = 0
    for frame in common:
        delta = {
            "dPSNR": _none_sub(method["PSNR"].get(frame), clean["PSNR"].get(frame)),
            "dSSIM": _none_sub(method["SSIM"].get(frame), clean["SSIM"].get(frame)),
            "dLPIPS": _none_sub(method["LPIPS"].get(frame), clean["LPIPS"].get(frame)),
        }
        ok = _strict_rgb_win(delta)
        strict += int(ok)
        rows.append({"frame": frame, "strict_rgb_win": ok, **delta})
    return {
        "available": True,
        "frame_count": len(common),
        "strict_rgb_wins": strict,
        "strict_rgb_win_fraction": strict / max(len(common), 1),
        "mean_dPSNR": mean(float(r["dPSNR"]) for r in rows if r["dPSNR"] is not None) if rows else None,
        "mean_dSSIM": mean(float(r["dSSIM"]) for r in rows if r["dSSIM"] is not None) if rows else None,
        "mean_dLPIPS": mean(float(r["dLPIPS"]) for r in rows if r["dLPIPS"] is not None) if rows else None,
        "worst_dPSNR": min((float(r["dPSNR"]) for r in rows if r["dPSNR"] is not None), default=None),
        "worst_dSSIM": min((float(r["dSSIM"]) for r in rows if r["dSSIM"] is not None), default=None),
        "worst_dLPIPS": max((float(r["dLPIPS"]) for r in rows if r["dLPIPS"] is not None), default=None),
        "frames": rows,
    }


def _geometry_row(clean_geo: dict[str, Any] | None, compact_geo: dict[str, Any] | None) -> dict[str, Any]:
    if not clean_geo or not compact_geo:
        return {"available": False}
    clean_depth = clean_geo.get("depth", {})
    compact_depth = compact_geo.get("depth", {})
    clean_normal = clean_geo.get("normal", {})
    compact_normal = compact_geo.get("normal", {})
    delta = {
        "d_abs_rel": _none_sub(compact_depth.get("abs_rel"), clean_depth.get("abs_rel")),
        "d_depth_mae": _none_sub(compact_depth.get("mae"), clean_depth.get("mae")),
        "d_normal_mean_ang_deg": _none_sub(compact_normal.get("mean_ang_deg"), clean_normal.get("mean_ang_deg")),
    }
    strict_eps = 1e-9
    strict_win = (
        delta["d_abs_rel"] is not None
        and delta["d_depth_mae"] is not None
        and delta["d_normal_mean_ang_deg"] is not None
        and float(delta["d_abs_rel"]) < -strict_eps
        and float(delta["d_depth_mae"]) < -strict_eps
        and float(delta["d_normal_mean_ang_deg"]) < -strict_eps
    )
    safe = (
        delta["d_abs_rel"] is not None
        and delta["d_depth_mae"] is not None
        and delta["d_normal_mean_ang_deg"] is not None
        and float(delta["d_abs_rel"]) <= max(0.002, abs(float(clean_depth.get("abs_rel", 0.0))) * 0.10)
        and float(delta["d_depth_mae"]) <= max(0.02, abs(float(clean_depth.get("mae", 0.0))) * 0.10)
        and float(delta["d_normal_mean_ang_deg"]) <= 0.25
    )
    return {
        "available": True,
        "clean_abs_rel": clean_depth.get("abs_rel"),
        "compact_abs_rel": compact_depth.get("abs_rel"),
        "clean_depth_mae": clean_depth.get("mae"),
        "compact_depth_mae": compact_depth.get("mae"),
        "clean_normal_mean_ang_deg": clean_normal.get("mean_ang_deg"),
        "compact_normal_mean_ang_deg": compact_normal.get("mean_ang_deg"),
        **delta,
        "strict_geometry_win": strict_win,
        "geometry_safe": safe,
    }


def _wandb_runs(path: Path) -> list[str]:
    wandb_dir = path / "wandb"
    if not wandb_dir.is_dir():
        return []
    return sorted(p.name.removeprefix("run-") for p in wandb_dir.iterdir() if p.is_dir() and p.name.startswith("run-"))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    phase_root = Path(args.phase_root)
    clean_root = Path(args.clean_root)
    source_root = Path(args.source_root)
    summary = _load_json(Path(args.summary_json))
    decisions = _load_json(Path(args.decisions_json))
    if not summary or not decisions:
        raise FileNotFoundError("Phase-J summary or decision JSON is missing.")
    decision_by_scene = {row["scene"]: row for row in decisions.get("rows", [])}
    rows = []
    per_view_rows = []

    for row in summary.get("rows", []):
        scene = str(row["scene"])
        if args.scenes and scene not in args.scenes:
            continue
        method_name = str(row.get("method_name", summary.get("args", {}).get("method_name", "")))
        clean_method = str(row.get("clean_baseline_method", "ours_26000"))
        model = Path(row["model"])
        compact_topology = _load_json(model / "topology_audit.json") or {}
        clean_scene_root = clean_root / scene
        compact_geo = _load_json(source_root / scene / args.source_policy_tag / "compact_model" / "geometry_eval_colmap" / "iter_26000_max500.json")
        clean_geo = _load_json(clean_scene_root / "geometry_eval_colmap" / "iter_26000_max500.json")
        geo = _geometry_row(clean_geo, compact_geo)
        clean_per_view = _load_json(clean_scene_root / "per_view.json")
        method_per_view = _load_json(model / "per_view.json")
        pv = _per_view_strict_counts(clean_per_view, method_per_view, clean_method, method_name)
        for frame_row in pv.get("frames", []):
            per_view_rows.append({"scene": scene, **frame_row})

        decision = decision_by_scene.get(scene, {})
        delta_clean = row.get("delta_vs_clean") or _metric_delta(row.get("method", {}), row.get("clean", {}))
        delta_source = row.get("delta_vs_source_ela") or _metric_delta(row.get("method", {}), row.get("source_ela", {}))
        rows.append(
            {
                "scene": scene,
                "method_name": method_name,
                "clean_baseline_method": clean_method,
                "selected_branch": decision.get("selected_method", ""),
                "stable_adaptive": decision.get("stable_adaptive"),
                "uses_test_gt_for_branch": decision.get("uses_test_gt"),
                "PSNR": row.get("method", {}).get("PSNR"),
                "SSIM": row.get("method", {}).get("SSIM"),
                "LPIPS": row.get("method", {}).get("LPIPS"),
                "clean_PSNR": row.get("clean", {}).get("PSNR"),
                "clean_SSIM": row.get("clean", {}).get("SSIM"),
                "clean_LPIPS": row.get("clean", {}).get("LPIPS"),
                "source_ela_PSNR": row.get("source_ela", {}).get("PSNR"),
                "source_ela_SSIM": row.get("source_ela", {}).get("SSIM"),
                "source_ela_LPIPS": row.get("source_ela", {}).get("LPIPS"),
                **delta_clean,
                "strict_rgb_win_vs_clean": _strict_rgb_win(delta_clean),
                "dPSNR_vs_source_ela": delta_source.get("dPSNR"),
                "dSSIM_vs_source_ela": delta_source.get("dSSIM"),
                "dLPIPS_vs_source_ela": delta_source.get("dLPIPS"),
                "strict_rgb_win_vs_source_ela": _strict_rgb_win(delta_source),
                "total_removed_fraction": row.get("total_removed_fraction"),
                "phasef_extra_removed_fraction": compact_topology.get("removed_fraction"),
                "pre_triangles": compact_topology.get("pre_triangles"),
                "post_triangles": compact_topology.get("post_triangles"),
                "pre_vertices": compact_topology.get("pre_vertices"),
                "post_vertices": compact_topology.get("post_vertices"),
                "degenerate_face_count": compact_topology.get("degenerate_face_count"),
                "invalid_index_count": compact_topology.get("invalid_index_count"),
                "per_view_available": pv.get("available", False),
                "per_view_frame_count": pv.get("frame_count"),
                "per_view_strict_rgb_wins": pv.get("strict_rgb_wins"),
                "per_view_strict_rgb_win_fraction": pv.get("strict_rgb_win_fraction"),
                "per_view_worst_dPSNR": pv.get("worst_dPSNR"),
                "per_view_worst_dSSIM": pv.get("worst_dSSIM"),
                "per_view_worst_dLPIPS": pv.get("worst_dLPIPS"),
                **geo,
                "clean_wandb_runs": ";".join(_wandb_runs(clean_scene_root)),
                "model_path": str(model),
            }
        )

    summary_stats = _summarize(rows, per_view_rows)
    return {
        "method": "Phase-J guarded adaptive edge policy closure audit",
        "inputs": {
            "summary_json": str(args.summary_json),
            "decisions_json": str(args.decisions_json),
            "phase_root": str(phase_root),
            "clean_root": str(clean_root),
            "source_root": str(source_root),
        },
        "summary": summary_stats,
        "rows": rows,
        "per_view_rows": per_view_rows,
    }


def _mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _summarize(rows: list[dict[str, Any]], per_view_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene_count": len(rows),
        "strict_rgb_wins_vs_clean": sum(1 for row in rows if row.get("strict_rgb_win_vs_clean")),
        "strict_rgb_wins_vs_source_ela": sum(1 for row in rows if row.get("strict_rgb_win_vs_source_ela")),
        "mean_dPSNR_vs_clean": _mean_field(rows, "dPSNR"),
        "mean_dSSIM_vs_clean": _mean_field(rows, "dSSIM"),
        "mean_dLPIPS_vs_clean": _mean_field(rows, "dLPIPS"),
        "mean_dPSNR_vs_source_ela": _mean_field(rows, "dPSNR_vs_source_ela"),
        "mean_dSSIM_vs_source_ela": _mean_field(rows, "dSSIM_vs_source_ela"),
        "mean_dLPIPS_vs_source_ela": _mean_field(rows, "dLPIPS_vs_source_ela"),
        "mean_total_removed_fraction": _mean_field(rows, "total_removed_fraction"),
        "mean_phasef_extra_removed_fraction": _mean_field(rows, "phasef_extra_removed_fraction"),
        "geometry_available_scenes": sum(1 for row in rows if row.get("available")),
        "strict_geometry_wins": sum(1 for row in rows if row.get("strict_geometry_win")),
        "geometry_safe_scenes": sum(1 for row in rows if row.get("geometry_safe")),
        "per_view_frame_count": len(per_view_rows),
        "per_view_strict_rgb_wins": sum(1 for row in per_view_rows if row.get("strict_rgb_win")),
        "per_view_strict_rgb_win_fraction": (
            sum(1 for row in per_view_rows if row.get("strict_rgb_win")) / max(len(per_view_rows), 1)
        ),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    val = _float(value)
    if val is None:
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "n/a"
        return str(value)
    return f"{val:.{digits}f}"


def _write_md(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    rows = report["rows"]
    lines = [
        "# Phase-J Closure Audit",
        "",
        "This audit is generated mechanically from existing Phase-J reports, clean MeshSplatting outputs, topology audits, per-view metrics, and sparse COLMAP geometry files. It does not select hyperparameters or views from held-out test metrics.",
        "",
        "## Summary",
        "",
        f"- Scenes: `{summary['scene_count']}`",
        f"- Strict RGB scene wins vs selected clean MeshSplatting: `{summary['strict_rgb_wins_vs_clean']} / {summary['scene_count']}`",
        f"- Strict RGB scene wins vs source Compact-ELA/SOR row: `{summary['strict_rgb_wins_vs_source_ela']} / {summary['scene_count']}`",
        f"- Mean delta vs clean: `{_fmt(summary['mean_dPSNR_vs_clean'])}` PSNR, `{_fmt(summary['mean_dSSIM_vs_clean'])}` SSIM, `{_fmt(summary['mean_dLPIPS_vs_clean'])}` LPIPS",
        f"- Mean delta vs source Compact-ELA/SOR row: `{_fmt(summary['mean_dPSNR_vs_source_ela'])}` PSNR, `{_fmt(summary['mean_dSSIM_vs_source_ela'])}` SSIM, `{_fmt(summary['mean_dLPIPS_vs_source_ela'])}` LPIPS",
        f"- Mean total triangle reduction: `{_fmt(100.0 * float(summary['mean_total_removed_fraction'] or 0.0), 4)}%`",
        f"- Sparse geometry strict wins: `{summary['strict_geometry_wins']} / {summary['geometry_available_scenes']}`",
        f"- Sparse geometry-safe scenes: `{summary['geometry_safe_scenes']} / {summary['geometry_available_scenes']}`",
        f"- Per-view strict RGB wins: `{summary['per_view_strict_rgb_wins']} / {summary['per_view_frame_count']}` (`{_fmt(100.0 * summary['per_view_strict_rgb_win_fraction'], 2)}%`)",
        "",
        "## Per-Scene Table",
        "",
        "| scene | branch | dPSNR clean | dSSIM clean | dLPIPS clean | tri red. | per-view strict | dAbsRel | dDepthMAE | dNormal | geom safe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        branch = "adaptive" if row.get("stable_adaptive") else "edge fallback"
        pv = "n/a"
        if row.get("per_view_frame_count"):
            pv = f"{row.get('per_view_strict_rgb_wins')} / {row.get('per_view_frame_count')}"
        lines.append(
            "| {scene} | {branch} | {dpsnr} | {dssim} | {dlpips} | {tri} | {pv} | {dabs} | {dmae} | {dnorm} | {safe} |".format(
                scene=row["scene"],
                branch=branch,
                dpsnr=_fmt(row.get("dPSNR")),
                dssim=_fmt(row.get("dSSIM")),
                dlpips=_fmt(row.get("dLPIPS")),
                tri=_fmt(100.0 * float(row.get("total_removed_fraction") or 0.0), 2) + "%",
                pv=pv,
                dabs=_fmt(row.get("d_abs_rel")),
                dmae=_fmt(row.get("d_depth_mae")),
                dnorm=_fmt(row.get("d_normal_mean_ang_deg")),
                safe="yes" if row.get("geometry_safe") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Closure Reading",
            "",
            "Phase-J is closed for the current selected Mip-NeRF360 RGB-plus-compactness claim: every scene improves PSNR, SSIM, and LPIPS over the selected clean baseline and over the source Compact-ELA/SOR row. The remaining non-closed part is the stricter representation-level claim: sparse geometry is safe but not strictly better on every scene, the strongest appearance recovery is still render-time ELA, and external cross-protocol evidence must remain a separate validation axis rather than a completed guarantee.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a paper-facing closure audit for the Phase-J ECSR method.")
    parser.add_argument(
        "--summary_json",
        default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json",
    )
    parser.add_argument(
        "--decisions_json",
        default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/ours_26000_phasej_guarded_adaptedge_ela_guarded_decisions.json",
    )
    parser.add_argument(
        "--phase_root",
        default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix",
    )
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument(
        "--source_root",
        default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k",
    )
    parser.add_argument("--source_policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit")
    parser.add_argument("--scenes", default="")
    args = parser.parse_args()
    args.summary_json = Path(args.summary_json)
    args.decisions_json = Path(args.decisions_json)
    args.scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    report = collect(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phasej_closure_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(report["rows"], out_dir / "phasej_closure_audit.csv")
    _write_csv(report["per_view_rows"], out_dir / "phasej_per_view_deltas.csv")
    _write_md(report, out_dir / "phasej_closure_audit.md")
    print(json.dumps(report["summary"], indent=2))
    print(f"[audit] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
