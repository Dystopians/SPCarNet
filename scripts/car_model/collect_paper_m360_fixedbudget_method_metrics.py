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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _topology(model: Path, iteration: int) -> tuple[int | None, int | None]:
    try:
        import torch

        state = torch.load(checkpoint_path(model, iteration), map_location="cpu")
        return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])
    except Exception:
        return None, None


def _wandb_id(model: Path) -> str:
    wandb_dir = model / "wandb"
    if not wandb_dir.is_dir():
        return ""
    runs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return ""
    name = runs[0].name
    return name.rsplit("-", 1)[-1] if "-" in name else name


def _selector_summary(method_model: Path) -> dict[str, Any]:
    payload = _read_json(method_model.parent / "selector" / "compaction_candidates.json")
    summary = payload.get("summary") or {}
    policy = payload.get("adaptive_policy_decision") or {}
    return {
        "selected_fraction": _num(summary.get("selected_fraction")),
        "selected_count": int(summary.get("selected_count", 0) or 0),
        "policy_fraction": _num(policy.get("target_prune_fraction")),
        "policy_risk": _num((policy.get("risk") or {}).get("policy_risk")),
    }


def _topology_unchanged(method_model: Path) -> bool | None:
    payload = _read_json(method_model.parent / "recovery_contract" / "topology_audit.json")
    value = payload.get("topology_unchanged")
    return bool(value) if value is not None else None


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _status(row: dict[str, Any]) -> str:
    checks = {
        "psnr": row["d_psnr"] > 0.0,
        "ssim": row["d_ssim"] > 0.0,
        "lpips": row["d_lpips"] < 0.0,
        "abs_rel": row["d_abs_rel"] < 0.0,
        "depth_mae": row["d_depth_mae"] < 0.0,
        "normal": row["d_normal"] < 0.0,
        "triangles": row["triangle_reduction"] > 0.0,
    }
    finite_rgb = all(math.isfinite(row[key]) for key in ("method_psnr", "method_ssim", "method_lpips", "clean_psnr", "clean_ssim", "clean_lpips"))
    if not finite_rgb:
        return "PENDING_OR_MISSING_EVAL"
    if all(checks.values()):
        return "PASS_ALL_METRIC_AND_COMPACT"
    if checks["psnr"] and checks["ssim"] and checks["lpips"] and checks["triangles"]:
        return "RGB_AND_COMPACT_PASS_GEOMETRY_MIXED"
    if checks["triangles"]:
        return "COMPACT_ONLY_MIXED"
    return "FAIL"


def _row(scene: str, clean_root: Path, method_root: Path, policy_tag: str, clean_iteration: int, method_iteration: int) -> dict[str, Any] | None:
    clean_model = clean_root / scene
    method_model = method_root / scene / policy_tag / "recovery_model"
    if not method_model.is_dir() or not clean_model.is_dir():
        return None
    clean_rgb = load_render_metrics(clean_model, clean_iteration)
    method_rgb = load_render_metrics(method_model, method_iteration)
    clean_geom = load_geometry_metrics(clean_model, clean_iteration)
    method_geom = load_geometry_metrics(method_model, method_iteration)
    clean_tri, clean_vertices = _topology(clean_model, clean_iteration)
    method_tri, method_vertices = _topology(method_model, method_iteration)
    selector = _selector_summary(method_model)
    reduction = 1.0 - float(method_tri) / float(clean_tri) if clean_tri and method_tri else math.nan
    row = {
        "scene": scene,
        "policy_tag": policy_tag,
        "wandb": _wandb_id(method_model),
        "clean_iteration": clean_iteration,
        "method_iteration": method_iteration,
        "clean_triangles": clean_tri,
        "method_triangles": method_tri,
        "clean_vertices": clean_vertices,
        "method_vertices": method_vertices,
        "triangle_reduction": reduction,
        "selector_fraction": selector["selected_fraction"],
        "policy_fraction": selector["policy_fraction"],
        "policy_risk": selector["policy_risk"],
        "topology_unchanged": _topology_unchanged(method_model),
        "clean_psnr": clean_rgb["psnr"],
        "clean_ssim": clean_rgb["ssim"],
        "clean_lpips": clean_rgb["lpips"],
        "clean_abs_rel": clean_geom["abs_rel"],
        "clean_depth_mae": clean_geom["depth_mae"],
        "clean_normal": clean_geom["normal_mean_ang_deg"],
        "method_psnr": method_rgb["psnr"],
        "method_ssim": method_rgb["ssim"],
        "method_lpips": method_rgb["lpips"],
        "method_abs_rel": method_geom["abs_rel"],
        "method_depth_mae": method_geom["depth_mae"],
        "method_normal": method_geom["normal_mean_ang_deg"],
    }
    row.update(
        {
            "d_psnr": row["method_psnr"] - row["clean_psnr"],
            "d_ssim": row["method_ssim"] - row["clean_ssim"],
            "d_lpips": row["method_lpips"] - row["clean_lpips"],
            "d_abs_rel": row["method_abs_rel"] - row["clean_abs_rel"],
            "d_depth_mae": row["method_depth_mae"] - row["clean_depth_mae"],
            "d_normal": row["method_normal"] - row["clean_normal"],
        }
    )
    row["status"] = _status(row)
    return row


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Paper Mip-NeRF360 Fixed-Budget Method Audit",
        "",
        f"Clean and method are compared at iteration `{args.final_iteration}`. The method compacts the clean checkpoint at `{args.compact_iteration}` and uses only the remaining budget to recover to `{args.final_iteration}`.",
        "",
        "| scene | W&B | prune | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | tri reduction | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | `{row['wandb']}` | {_fmt(row['policy_fraction'])} | "
            f"{_fmt(row['method_psnr'])} | {_fmt(row['method_ssim'])} | {_fmt(row['method_lpips'])} | "
            f"{_fmt(row['method_abs_rel'])} | {_fmt(row['method_depth_mae'])} | {_fmt(row['method_normal'])} | "
            f"{row['d_psnr']:+.6f} | {row['d_ssim']:+.6f} | {row['d_lpips']:+.6f} | "
            f"{row['d_abs_rel']:+.6f} | {row['d_depth_mae']:+.6f} | {row['d_normal']:+.6f} | "
            f"{100.0 * row['triangle_reduction']:.2f}% | `{row['status']}` |"
        )
    pass_count = sum(row["status"] == "PASS_ALL_METRIC_AND_COMPACT" for row in rows)
    rgb_count = sum(row["status"] in {"PASS_ALL_METRIC_AND_COMPACT", "RGB_AND_COMPACT_PASS_GEOMETRY_MIXED"} for row in rows)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- available rows: `{len(rows)}`",
            f"- all-metric + compact pass: `{pass_count}` / `{len(rows)}`",
            f"- RGB + compact pass: `{rgb_count}` / `{len(rows)}`",
            "",
            "This is the reviewer-facing fixed-budget check. A 30k->34k recovery can still be useful as a diagnostic, but it must not be mixed with this table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_wandb(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_name, config=vars(args))
    pass_count = sum(row["status"] == "PASS_ALL_METRIC_AND_COMPACT" for row in rows)
    rgb_count = sum(row["status"] in {"PASS_ALL_METRIC_AND_COMPACT", "RGB_AND_COMPACT_PASS_GEOMETRY_MIXED"} for row in rows)
    summary = {
        "available_rows": len(rows),
        "pass_all_metric_compact": pass_count,
        "rgb_compact_pass": rgb_count,
        "mean_d_psnr": sum(row["d_psnr"] for row in rows) / len(rows) if rows else math.nan,
        "mean_d_ssim": sum(row["d_ssim"] for row in rows) / len(rows) if rows else math.nan,
        "mean_d_lpips": sum(row["d_lpips"] for row in rows) / len(rows) if rows else math.nan,
        "mean_triangle_reduction": sum(row["triangle_reduction"] for row in rows) / len(rows) if rows else math.nan,
    }
    wandb.log(summary)
    for row in rows:
        prefix = f"scene/{row['scene']}"
        wandb.log(
            {
                f"{prefix}/d_psnr": row["d_psnr"],
                f"{prefix}/d_ssim": row["d_ssim"],
                f"{prefix}/d_lpips": row["d_lpips"],
                f"{prefix}/triangle_reduction": row["triangle_reduction"],
                f"{prefix}/pass_all_metric_compact": int(row["status"] == "PASS_ALL_METRIC_AND_COMPACT"),
            }
        )
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect same-final-iteration fixed-budget method vs clean metrics.")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--method_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/fixedbudget_csef_atr_26kto30k")
    parser.add_argument("--policy_tag", default="csef_atr_fixedbudget")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--compact_iteration", type=int, default=26000)
    parser.add_argument("--final_iteration", type=int, default=30000)
    parser.add_argument("--out_dir", default="outputs/carnet/meshsplatopt/paper_m360_repro/fixedbudget_csef_atr_26kto30k")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="paper_m360_fixedbudget_csef_atr_26kto30k")
    parser.add_argument("--wandb_name", default="paper_m360_fixedbudget_method_metrics")
    args = parser.parse_args()

    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    rows = [
        row
        for scene in scenes
        if (row := _row(scene, ROOT / args.clean_root, ROOT / args.method_root, args.policy_tag, args.final_iteration, args.final_iteration)) is not None
    ]
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "fixedbudget_method_vs_clean.csv", rows)
    (out / "fixedbudget_method_vs_clean.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out / "fixedbudget_method_vs_clean_report.md", rows, args)
    _log_wandb(rows, args)
    print(json.dumps({"rows": len(rows), "report": str(out / "fixedbudget_method_vs_clean_report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
