#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402


DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
GEOMETRY_TOL = 1e-5


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _score(rgb: dict[str, float]) -> float:
    psnr = _num(rgb.get("psnr"))
    ssim = _num(rgb.get("ssim"))
    lpips = _num(rgb.get("lpips"))
    if not all(math.isfinite(v) for v in (psnr, ssim, lpips)):
        return -math.inf
    return psnr + 20.0 * ssim - 20.0 * lpips


def _topology(model: Path, iteration: int) -> tuple[int | None, int | None]:
    try:
        import torch

        state = torch.load(checkpoint_path(model, iteration), map_location="cpu")
        return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])
    except Exception:
        return None, None


def _selector_summary(model: Path) -> dict[str, Any]:
    payload = _read_json(model.parent / "selector" / "compaction_candidates.json")
    summary = payload.get("summary") or {}
    policy = payload.get("adaptive_policy_decision") or {}
    return {
        "selected_fraction": _num(summary.get("selected_fraction")),
        "selected_count": int(summary.get("selected_count", 0) or 0),
        "policy_fraction": _num(policy.get("target_prune_fraction")),
        "policy_risk": _num((policy.get("risk") or {}).get("policy_risk")),
        "policy_reason": str(policy.get("reason", "")),
    }


def _ela_summary(model: Path, method_name: str) -> dict[str, Any]:
    payload = _read_json(model / "test" / method_name / "ela_report.json")
    policy = payload.get("policy") or {}
    calibration = payload.get("calibration") or {}
    return {
        "ela_alpha": _num(payload.get("alpha")),
        "ela_mode": str(policy.get("mode", payload.get("mode", ""))),
        "ela_k": int(policy.get("k", payload.get("k", 0)) or 0),
        "ela_depth_rel_tol": _num(policy.get("depth_rel_tol", payload.get("depth_rel_tol"))),
        "ela_residual_clip": _num(policy.get("residual_clip", payload.get("residual_clip"))),
        "ela_direction_weight": _num(policy.get("direction_weight", payload.get("direction_weight"))),
        "ela_mean_covered_fraction": _num(payload.get("mean_covered_fraction")),
        "ela_mean_confidence": _num(payload.get("mean_confidence")),
        "ela_calib_views": len(calibration.get("calibration_views") or []),
    }


def _baseline_candidates(clean_model: Path, iterations: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iteration in iterations:
        rgb = load_render_metrics(clean_model, iteration)
        geom = load_geometry_metrics(clean_model, iteration)
        tri, vertices = _topology(clean_model, iteration)
        rows.append(
            {
                "iteration": int(iteration),
                "method": f"ours_{int(iteration)}",
                "score": _score(rgb),
                "psnr": rgb["psnr"],
                "ssim": rgb["ssim"],
                "lpips": rgb["lpips"],
                "abs_rel": geom["abs_rel"],
                "depth_mae": geom["depth_mae"],
                "normal": geom["normal_mean_ang_deg"],
                "triangles": tri,
                "vertices": vertices,
            }
        )
    return rows


def _select_baseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    finite = [row for row in rows if math.isfinite(float(row.get("score", -math.inf)))]
    if not finite:
        return None
    return max(finite, key=lambda row: (float(row["score"]), float(row["psnr"])))


def _metric_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = {
        key: [row for row in rows if math.isfinite(_num(row.get(key)))]
        for key in ("psnr", "ssim", "lpips", "abs_rel", "depth_mae", "normal")
    }
    out: dict[str, Any] = {}
    if finite["psnr"]:
        out["env_psnr"] = max(finite["psnr"], key=lambda row: float(row["psnr"]))["psnr"]
    if finite["ssim"]:
        out["env_ssim"] = max(finite["ssim"], key=lambda row: float(row["ssim"]))["ssim"]
    if finite["lpips"]:
        out["env_lpips"] = min(finite["lpips"], key=lambda row: float(row["lpips"]))["lpips"]
    if finite["abs_rel"]:
        out["env_abs_rel"] = min(finite["abs_rel"], key=lambda row: float(row["abs_rel"]))["abs_rel"]
    if finite["depth_mae"]:
        out["env_depth_mae"] = min(finite["depth_mae"], key=lambda row: float(row["depth_mae"]))["depth_mae"]
    if finite["normal"]:
        out["env_normal"] = min(finite["normal"], key=lambda row: float(row["normal"]))["normal"]
    return out


def _status(row: dict[str, Any]) -> str:
    finite_rgb = all(math.isfinite(_num(row[key])) for key in ("method_psnr", "method_ssim", "method_lpips", "baseline_psnr", "baseline_ssim", "baseline_lpips"))
    if not finite_rgb:
        return "PENDING_OR_MISSING_EVAL"
    checks = {
        "rgb": row["d_psnr"] > 0.0 and row["d_ssim"] > 0.0 and row["d_lpips"] < 0.0,
        "geometry_win": row["d_abs_rel"] < -GEOMETRY_TOL and row["d_depth_mae"] < -GEOMETRY_TOL and row["d_normal"] < -GEOMETRY_TOL,
        "geometry_safe": row["d_abs_rel"] <= GEOMETRY_TOL and row["d_depth_mae"] <= GEOMETRY_TOL and row["d_normal"] <= GEOMETRY_TOL,
        "compact": row["triangle_reduction"] > 0.0 and row["vertex_reduction"] > 0.0,
    }
    if checks["rgb"] and checks["geometry_win"] and checks["compact"]:
        return "STRICT_ALL_AXIS_PASS"
    if checks["rgb"] and checks["geometry_safe"] and checks["compact"]:
        return "RGB_COMPACT_PASS_GEOMETRY_SAFE"
    if checks["rgb"] and checks["compact"]:
        return "RGB_COMPACT_PASS_GEOMETRY_MIXED"
    if checks["compact"]:
        return "COMPACT_ONLY_MIXED"
    return "FAIL"


def _row(scene: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    clean_model = ROOT / args.clean_root / scene
    compact_model = ROOT / args.method_root / scene / args.policy_tag / "compact_model"
    if not clean_model.is_dir() or not compact_model.is_dir():
        return None, []
    baseline_rows = _baseline_candidates(clean_model, args.baseline_iterations)
    baseline = _select_baseline(baseline_rows)
    if baseline is None:
        return None, baseline_rows

    method_rgb = load_render_metrics(compact_model, args.method_iteration)
    method_payload = _read_json(compact_model / "results.json").get(args.method_name, {})
    if method_payload:
        method_rgb = {
            "psnr": _num(method_payload.get("PSNR")),
            "ssim": _num(method_payload.get("SSIM")),
            "lpips": _num(method_payload.get("LPIPS")),
        }
    method_geom = load_geometry_metrics(compact_model, args.method_iteration)
    method_tri, method_vertices = _topology(compact_model, args.method_iteration)
    selector = _selector_summary(compact_model)
    ela = _ela_summary(compact_model, args.method_name)
    envelope = _metric_envelope(baseline_rows)

    triangle_reduction = 1.0 - float(method_tri) / float(baseline["triangles"]) if method_tri and baseline["triangles"] else math.nan
    vertex_reduction = 1.0 - float(method_vertices) / float(baseline["vertices"]) if method_vertices and baseline["vertices"] else math.nan
    row = {
        "scene": scene,
        "policy_tag": args.policy_tag,
        "method_name": args.method_name,
        "baseline_iteration": int(baseline["iteration"]),
        "method_iteration": int(args.method_iteration),
        "baseline_score": baseline["score"],
        "baseline_psnr": baseline["psnr"],
        "baseline_ssim": baseline["ssim"],
        "baseline_lpips": baseline["lpips"],
        "baseline_abs_rel": baseline["abs_rel"],
        "baseline_depth_mae": baseline["depth_mae"],
        "baseline_normal": baseline["normal"],
        "baseline_triangles": baseline["triangles"],
        "baseline_vertices": baseline["vertices"],
        "method_psnr": method_rgb["psnr"],
        "method_ssim": method_rgb["ssim"],
        "method_lpips": method_rgb["lpips"],
        "method_abs_rel": method_geom["abs_rel"],
        "method_depth_mae": method_geom["depth_mae"],
        "method_normal": method_geom["normal_mean_ang_deg"],
        "method_triangles": method_tri,
        "method_vertices": method_vertices,
        "triangle_reduction": triangle_reduction,
        "vertex_reduction": vertex_reduction,
        **selector,
        **ela,
        **envelope,
    }
    row.update(
        {
            "d_psnr": row["method_psnr"] - row["baseline_psnr"],
            "d_ssim": row["method_ssim"] - row["baseline_ssim"],
            "d_lpips": row["method_lpips"] - row["baseline_lpips"],
            "d_abs_rel": row["method_abs_rel"] - row["baseline_abs_rel"],
            "d_depth_mae": row["method_depth_mae"] - row["baseline_depth_mae"],
            "d_normal": row["method_normal"] - row["baseline_normal"],
            "d_psnr_vs_env": row["method_psnr"] - row.get("env_psnr", math.nan),
            "d_ssim_vs_env": row["method_ssim"] - row.get("env_ssim", math.nan),
            "d_lpips_vs_env": row["method_lpips"] - row.get("env_lpips", math.nan),
        }
    )
    row["status"] = _status(row)
    return row, baseline_rows


def _fmt(value: Any, precision: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{precision}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    strict = sum(1 for row in rows if row["status"] == "STRICT_ALL_AXIS_PASS")
    rgb_compact = sum(1 for row in rows if row["status"] in {"STRICT_ALL_AXIS_PASS", "RGB_COMPACT_PASS_GEOMETRY_SAFE", "RGB_COMPACT_PASS_GEOMETRY_MIXED"})
    geometry_safe = sum(1 for row in rows if row["status"] in {"STRICT_ALL_AXIS_PASS", "RGB_COMPACT_PASS_GEOMETRY_SAFE"})
    lines = [
        "# Paper Mip-NeRF360 Compact-ELA Same-Protocol Audit",
        "",
        "Baseline selection uses held-out test metrics only. For each scene, the clean MeshSplatting checkpoint is selected by the coherent score `PSNR + 20 * SSIM - 20 * LPIPS` over the configured clean iterations; train metrics are not used.",
        "",
        f"Method: `{args.method_name}` on compact checkpoints at iteration `{args.method_iteration}`.",
        f"Clean baseline candidates: `{','.join(str(x) for x in args.baseline_iterations)}`.",
        "",
        "| scene | clean iter | prune | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | tri red. | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['baseline_iteration']} | {_fmt(row['policy_fraction'])} | "
            f"{_fmt(row['method_psnr'])} | {_fmt(row['method_ssim'])} | {_fmt(row['method_lpips'])} | "
            f"{_fmt(row['method_abs_rel'])} | {_fmt(row['method_depth_mae'])} | {_fmt(row['method_normal'])} | "
            f"{float(row['d_psnr']):+.6f} | {float(row['d_ssim']):+.6f} | {float(row['d_lpips']):+.6f} | "
            f"{float(row['d_abs_rel']):+.6f} | {float(row['d_depth_mae']):+.6f} | {float(row['d_normal']):+.6f} | "
            f"{100.0 * float(row['triangle_reduction']):.2f}% | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- available scenes: `{len(rows)}`",
            f"- strict all-axis pass: `{strict}/{len(rows)}`",
            f"- RGB + compact + geometry-safe pass: `{geometry_safe}/{len(rows)}`",
            f"- RGB + compact pass: `{rgb_compact}/{len(rows)}`",
            f"- mean dPSNR: `{_fmt(sum(row['d_psnr'] for row in rows) / len(rows) if rows else math.nan)}`",
            f"- mean dSSIM: `{_fmt(sum(row['d_ssim'] for row in rows) / len(rows) if rows else math.nan)}`",
            f"- mean dLPIPS: `{_fmt(sum(row['d_lpips'] for row in rows) / len(rows) if rows else math.nan)}`",
            f"- mean triangle reduction: `{_fmt(sum(row['triangle_reduction'] for row in rows) / len(rows) if rows else math.nan)}`",
            "",
            "## Clean Baseline Candidate Diagnostics",
            "",
            "| scene | iter | score | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | triangles |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {row['scene']} | {row['iteration']} | {_fmt(row['score'])} | {_fmt(row['psnr'])} | {_fmt(row['ssim'])} | "
            f"{_fmt(row['lpips'])} | {_fmt(row['abs_rel'])} | {_fmt(row['depth_mae'])} | {_fmt(row['normal'])} | {row['triangles']} |"
        )
    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "The ELA component is calibrated only from train-split renders/GT/depth/cameras and then applied to held-out test views. It is therefore a fixed train-only reconstruction policy, not a test-metric selector. Geometry metrics and topology come from the compact checkpoint, not from the ELA render layer.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_wandb(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_name, config=vars(args))
    summary = {
        "scenes": len(rows),
        "strict_all_axis_pass": sum(1 for row in rows if row["status"] == "STRICT_ALL_AXIS_PASS"),
        "rgb_geometry_safe_compact_pass": sum(1 for row in rows if row["status"] in {"STRICT_ALL_AXIS_PASS", "RGB_COMPACT_PASS_GEOMETRY_SAFE"}),
        "rgb_compact_pass": sum(1 for row in rows if row["status"] in {"STRICT_ALL_AXIS_PASS", "RGB_COMPACT_PASS_GEOMETRY_SAFE", "RGB_COMPACT_PASS_GEOMETRY_MIXED"}),
        "mean_d_psnr": sum(row["d_psnr"] for row in rows) / len(rows) if rows else math.nan,
        "mean_d_ssim": sum(row["d_ssim"] for row in rows) / len(rows) if rows else math.nan,
        "mean_d_lpips": sum(row["d_lpips"] for row in rows) / len(rows) if rows else math.nan,
        "mean_triangle_reduction": sum(row["triangle_reduction"] for row in rows) / len(rows) if rows else math.nan,
    }
    wandb.log(summary)
    run.summary.update(summary)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect compact CSEF-ATR + train-only ELA vs clean MeshSplatting metrics.")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--method_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_csef_atr_26k")
    parser.add_argument("--policy_tag", default="csef_atr_compact_ela")
    parser.add_argument("--method_name", default="ours_26000_csef_atr_compact_ela")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--baseline_iterations", default="26000,30000")
    parser.add_argument("--method_iteration", type=int, default=26000)
    parser.add_argument("--out_dir", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_csef_atr_26k")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="paper_m360_compact_ela_csef_atr_26k")
    parser.add_argument("--wandb_name", default="paper_m360_compact_ela_policy_metrics")
    args = parser.parse_args()

    args.baseline_iterations = [int(item.strip()) for item in str(args.baseline_iterations).replace(" ", ",").split(",") if item.strip()]
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for scene in scenes:
        row, base = _row(scene, args)
        for item in base:
            item["scene"] = scene
        baseline_rows.extend(base)
        if row is not None:
            rows.append(row)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "compact_ela_vs_clean.csv", rows)
    _write_csv(out / "compact_ela_clean_baseline_candidates.csv", baseline_rows)
    (out / "compact_ela_vs_clean.json").write_text(
        json.dumps({"rows": rows, "baseline_candidates": baseline_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(out / "compact_ela_vs_clean_report.md", rows, baseline_rows, args)
    _log_wandb(rows, args)
    print(json.dumps({"rows": len(rows), "report": str(out / "compact_ela_vs_clean_report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
