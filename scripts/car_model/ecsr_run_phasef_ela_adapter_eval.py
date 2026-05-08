#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}


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


def _metric(payload: dict[str, Any], key: str) -> dict[str, float]:
    item = payload.get(key, {})
    return {"PSNR": _num(item.get("PSNR")), "SSIM": _num(item.get("SSIM")), "LPIPS": _num(item.get("LPIPS"))}


def _score(metrics: dict[str, Any]) -> float:
    vals = (_num(metrics.get("PSNR")), _num(metrics.get("SSIM")), _num(metrics.get("LPIPS")))
    if not all(math.isfinite(v) for v in vals):
        return -math.inf
    psnr, ssim, lpips = vals
    return psnr + 20.0 * ssim - 20.0 * lpips


def _select_clean(clean_results: dict[str, Any], iterations: list[int]) -> tuple[str, dict[str, float]]:
    rows = []
    for iteration in iterations:
        key = f"ours_{int(iteration)}"
        metrics = _metric(clean_results, key)
        rows.append((key, metrics, _score(metrics)))
    finite = [row for row in rows if math.isfinite(row[2])]
    if not finite:
        return "", {"PSNR": math.nan, "SSIM": math.nan, "LPIPS": math.nan}
    key, metrics, _ = max(finite, key=lambda row: (row[2], row[1]["PSNR"]))
    return key, metrics


def _image_set(scene: str, outdoor_images: str, indoor_images: str) -> str:
    return outdoor_images if scene in OUTDOOR_SCENES else indoor_images


def _run(cmd: list[str], *, gpu: int, log_path: Path) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _summary_stem(args: argparse.Namespace) -> str:
    suffix = str(args.summary_suffix or "").strip()
    if not suffix:
        return "phasef_ela_eval_summary"
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    return "phasef_ela_eval_summary" + suffix


def _ratio_tag(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 10000.0)):04d}"


def _selected_model(policy_root: Path, scene: str, force_ratio: float | None = None) -> Path:
    if force_ratio is not None:
        model = policy_root / scene / _ratio_tag(force_ratio) / "compact_model"
        if not model.is_dir():
            raise FileNotFoundError(model)
        return model
    summary = _read_json(policy_root / scene / "summary.json")
    selected = summary.get("selected") or {}
    model_path = selected.get("model_path")
    if not model_path:
        raise RuntimeError(f"missing selected model in {policy_root / scene / 'summary.json'}")
    model = ROOT / model_path
    if not model.is_dir():
        raise FileNotFoundError(model)
    return model


def _render_base(args: argparse.Namespace, scene: str, model: Path, log_path: Path) -> None:
    base_dir = model / "train" / args.base_method_name
    test_dir = model / "test" / args.base_method_name
    if (base_dir / "depths").is_dir() and (test_dir / "depths").is_dir() and (test_dir / "camera_index.json").is_file():
        return
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(ROOT / args.dataset_root / scene),
        "-m",
        str(model),
        "-i",
        _image_set(scene, args.outdoor_images, args.indoor_images),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--method_name",
        args.base_method_name,
        "--quiet",
    ]
    if args.skip_failed_views:
        cmd.append("--skip_failed_views")
    _run(cmd, gpu=args.gpu, log_path=log_path)


def _apply_ela(args: argparse.Namespace, scene: str, model: Path, log_path: Path) -> None:
    report = model / "test" / args.method_name / "ela_report.json"
    if report.is_file():
        return
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py",
        "--base_model_path",
        str(model),
        "--iteration",
        str(args.iteration),
        "--base_method_name",
        args.base_method_name,
        "--method_name",
        args.method_name,
        "--edge_gate",
        "--edge_gate_quantile",
        str(args.edge_gate_quantile),
        "--edge_gate_dilate",
        str(args.edge_gate_dilate),
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_name}_{scene}",
    ]
    if args.fixed_policy:
        cmd.extend(
            [
                "--mode",
                args.fixed_mode,
                "--k",
                str(args.fixed_k),
                "--residual_clip",
                str(args.fixed_residual_clip),
                "--depth_rel_tol",
                str(args.fixed_depth_rel_tol),
                "--direction_weight",
                str(args.fixed_direction_weight),
                "--alpha",
                str(args.fixed_alpha),
            ]
        )
    else:
        cmd.extend(
            [
                "--auto_policy",
                "--policy_modes",
                args.policy_modes,
                "--policy_k_values",
                args.policy_k_values,
                "--policy_depth_rel_values",
                args.policy_depth_rel_values,
                "--policy_residual_clip_values",
                args.policy_residual_clip_values,
                "--policy_direction_weight_values",
                args.policy_direction_weight_values,
                "--policy_objective",
                args.policy_objective,
                "--policy_holdout_fraction",
                str(args.policy_holdout_fraction),
                "--alpha_grid",
                args.alpha_grid,
                "--calib_sampler",
                args.calib_sampler,
            ]
        )
    if args.benefit_policy:
        cmd.append("--benefit_policy")
        cmd.extend(["--benefit_feature_mode", args.benefit_feature_mode])
    if args.calib_lpips:
        cmd.append("--calib_lpips")
    _run(cmd, gpu=args.gpu, log_path=log_path)


def _evaluate(args: argparse.Namespace, model: Path, log_path: Path) -> dict[str, float]:
    existing = _metric(_read_json(model / "results.json"), args.method_name)
    if all(math.isfinite(existing[k]) for k in ("PSNR", "SSIM", "LPIPS")):
        return existing
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model),
        "--split",
        "test",
        "--methods",
        args.method_name,
        "--merge_model_results",
    ]
    _run(cmd, gpu=args.gpu, log_path=log_path)
    return _metric(_read_json(model / "results.json"), args.method_name)


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {"dPSNR": a["PSNR"] - b["PSNR"], "dSSIM": a["SSIM"] - b["SSIM"], "dLPIPS": a["LPIPS"] - b["LPIPS"]}


def _scene_row(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    policy_root = ROOT / args.policy_root
    model = _selected_model(policy_root, scene, args.force_ratio)
    log_path = policy_root / scene / "phasef_ela_eval.log"
    _render_base(args, scene, model, log_path)
    _apply_ela(args, scene, model, log_path)
    method = _evaluate(args, model, log_path)

    clean_key, clean = _select_clean(_read_json(ROOT / args.clean_root / scene / "results.json"), args.clean_iterations)
    source_model = ROOT / args.source_root / scene / args.policy_tag / "compact_model"
    source = _metric(_read_json(source_model / "results.json"), args.source_ela_method)
    heldout = _read_json(policy_root / scene / "heldout_eval_summary.json")
    total_removed = _num(heldout.get("total_removed_fraction"))
    row = {
        "scene": scene,
        "model": str(model.relative_to(ROOT)),
        "method_name": args.method_name,
        "clean_baseline_method": clean_key,
        "total_removed_fraction": total_removed,
        "method": method,
        "clean": clean,
        "source_ela": source,
        "delta_vs_clean": _delta(method, clean),
        "delta_vs_source_ela": _delta(method, source),
        "force_ratio": args.force_ratio,
    }
    (policy_root / scene / f"{_summary_stem(args)}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return row


def _write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    root = ROOT / args.policy_root
    payload = {"rows": rows, "args": vars(args)}
    stem = _summary_stem(args)
    (root / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mean = lambda key, sub: sum(row[key][sub] for row in rows) / len(rows) if rows else math.nan
    lines = [
        "# ECSR Phase-F + Train-Only ELA Evaluation",
        "",
        "This applies the same train-only evidence-lumigraph adapter family on top of the fixed Phase-F selected compact checkpoints.",
        "",
        f"- scenes: `{len(rows)}`",
        f"- mean total triangle reduction: `{sum(_num(row['total_removed_fraction']) for row in rows) / len(rows) if rows else math.nan:.6f}`",
        f"- mean dPSNR vs clean: `{mean('delta_vs_clean', 'dPSNR'):.6f}`",
        f"- mean dSSIM vs clean: `{mean('delta_vs_clean', 'dSSIM'):.6f}`",
        f"- mean dLPIPS vs clean: `{mean('delta_vs_clean', 'dLPIPS'):.6f}`",
        f"- mean dPSNR vs source ELA: `{mean('delta_vs_source_ela', 'dPSNR'):.6f}`",
        f"- mean dSSIM vs source ELA: `{mean('delta_vs_source_ela', 'dSSIM'):.6f}`",
        f"- mean dLPIPS vs source ELA: `{mean('delta_vs_source_ela', 'dLPIPS'):.6f}`",
        "",
        "| scene | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR source ELA | dSSIM source ELA | dLPIPS source ELA | tri red. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["method"]
        dc = row["delta_vs_clean"]
        ds = row["delta_vs_source_ela"]
        lines.append(
            f"| {row['scene']} | {m['PSNR']:.6f} | {m['SSIM']:.6f} | {m['LPIPS']:.6f} | "
            f"{dc['dPSNR']:+.6f} | {dc['dSSIM']:+.6f} | {dc['dLPIPS']:+.6f} | "
            f"{ds['dPSNR']:+.6f} | {ds['dSSIM']:+.6f} | {ds['dLPIPS']:+.6f} | "
            f"{100.0 * _num(row['total_removed_fraction']):.2f}% |"
        )
    (root / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply train-only ELA on fixed Phase-F compact checkpoints and evaluate held-out metrics.")
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--source_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--scenes", default="garden")
    parser.add_argument("--clean_iterations", default="26000,30000")
    parser.add_argument("--source_ela_method", default="ours_26000_sor_adaptive_geo_compact_ela")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default="ours_26000_phasef_extra_compact_ela")
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--policy_modes", default="residual")
    parser.add_argument("--policy_k_values", default="4,8")
    parser.add_argument("--policy_depth_rel_values", default="0.06,0.12")
    parser.add_argument("--policy_residual_clip_values", default="0.2,0.25")
    parser.add_argument("--policy_direction_weight_values", default="0.2,0.35")
    parser.add_argument("--policy_objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--policy_holdout_fraction", type=float, default=0.0)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="stride_first")
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--fixed_policy", action="store_true")
    parser.add_argument("--fixed_mode", choices=("residual", "color"), default="residual")
    parser.add_argument("--fixed_k", type=int, default=8)
    parser.add_argument("--fixed_residual_clip", type=float, default=0.2)
    parser.add_argument("--fixed_depth_rel_tol", type=float, default=0.12)
    parser.add_argument("--fixed_direction_weight", type=float, default=0.2)
    parser.add_argument("--fixed_alpha", type=float, default=1.0)
    parser.add_argument("--edge_gate_quantile", type=float, default=0.7)
    parser.add_argument("--edge_gate_dilate", type=int, default=1)
    parser.add_argument("--benefit_policy", action="store_true")
    parser.add_argument("--benefit_feature_mode", choices=("confidence_magnitude", "confidence_magnitude_edge", "auto"), default="confidence_magnitude")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phase_f_plus_ela")
    parser.add_argument("--wandb_name", default="phase_f_plus_ela")
    parser.add_argument("--force_ratio", type=float, default=None, help="diagnostic override: use policy_root/<scene>/ratio_xxxx/compact_model instead of the selected policy model")
    parser.add_argument("--summary_suffix", default="", help="suffix for per-scene and aggregate summary files, so diagnostics do not overwrite the fixed-policy report")
    args = parser.parse_args()
    args.clean_iterations = [int(item) for item in str(args.clean_iterations).replace(" ", ",").split(",") if item.strip()]
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    rows = [_scene_row(args, scene) for scene in scenes]
    _write_summary(args, rows)
    print(json.dumps({"rows": len(rows), "report": str(ROOT / args.policy_root / f"{_summary_stem(args)}.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
