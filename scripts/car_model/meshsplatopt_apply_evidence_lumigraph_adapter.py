#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_lumigraph_adapter import (
    FrameLoader,
    adapt_frame,
    calibrate_alpha,
    load_split_frames,
    save_image_tensor,
)


def _parse_alpha_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _selected_calibration_row(calibration: dict, alpha: float) -> dict:
    rows = calibration.get("rows", [])
    for row in rows:
        if abs(float(row.get("alpha", -999.0)) - float(alpha)) < 1e-9:
            return row
    return {}


def _choose_policy(args: argparse.Namespace, train_frames, alpha_grid: list[float], device: torch.device) -> tuple[dict, dict, list[dict]]:
    if not args.auto_policy:
        calibration = calibrate_alpha(
            train_frames,
            alpha_grid=alpha_grid,
            k=args.k,
            mode=args.mode,
            calib_stride=args.calib_stride,
            calib_max_views=args.calib_max_views,
            residual_clip=args.residual_clip,
            depth_abs_tol=args.depth_abs_tol,
            depth_rel_tol=args.depth_rel_tol,
            direction_weight=args.direction_weight,
            device=device,
        )
        policy = {
            "mode": args.mode,
            "k": int(args.k),
            "residual_clip": float(args.residual_clip),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
        }
        return policy, calibration, []

    candidate_rows: list[dict] = []
    best: tuple[float, int, dict, dict] | None = None
    modes = [m.strip() for m in args.policy_modes.split(",") if m.strip()]
    k_values = _parse_int_grid(args.policy_k_values)
    depth_rel_values = _parse_float_grid(args.policy_depth_rel_values)
    clip_values = _parse_float_grid(args.policy_residual_clip_values)
    order = 0
    for mode in modes:
        for k in k_values:
            for depth_rel in depth_rel_values:
                for residual_clip in clip_values:
                    calibration = calibrate_alpha(
                        train_frames,
                        alpha_grid=alpha_grid,
                        k=int(k),
                        mode=mode,
                        calib_stride=args.calib_stride,
                        calib_max_views=args.calib_max_views,
                        residual_clip=float(residual_clip),
                        depth_abs_tol=args.depth_abs_tol,
                        depth_rel_tol=float(depth_rel),
                        direction_weight=args.direction_weight,
                        device=device,
                    )
                    alpha = float(calibration["alpha"])
                    row = _selected_calibration_row(calibration, alpha)
                    gain = float(row.get("psnr_gain", 0.0))
                    policy = {
                        "mode": mode,
                        "k": int(k),
                        "residual_clip": float(residual_clip),
                        "depth_abs_tol": float(args.depth_abs_tol),
                        "depth_rel_tol": float(depth_rel),
                    }
                    candidate_rows.append(
                        {
                            **policy,
                            "alpha": alpha,
                            "calib_psnr_gain": gain,
                            "calib_psnr": row.get("psnr"),
                            "calib_base_psnr": row.get("base_psnr"),
                        }
                    )
                    rank = (gain, -order)
                    if best is None or rank > (best[0], best[1]):
                        best = (gain, -order, policy, calibration)
                    order += 1
    assert best is not None
    return best[2], best[3], candidate_rows


def _copy_gt(target_frames, out_gt: Path) -> None:
    out_gt.mkdir(parents=True, exist_ok=True)
    for frame in target_frames:
        dst = out_gt / frame.render_path.name
        if dst.exists():
            continue
        shutil.copy2(frame.gt_path, dst)


def _maybe_wandb(args: argparse.Namespace, report: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[ELA] W&B unavailable, skipping log: {exc}")
        return
    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=mode,
        config={
            "base_model_path": args.base_model_path,
            "iteration": args.iteration,
            "k": args.k,
            "mode": args.mode,
            "residual_clip": args.residual_clip,
            "depth_abs_tol": args.depth_abs_tol,
            "depth_rel_tol": args.depth_rel_tol,
            "direction_weight": args.direction_weight,
            "target_split": args.target_split,
            "method_name": args.method_name,
        },
    )
    flat = {
        "ela/alpha": float(report.get("alpha", 0.0)),
        "ela/k": int(report.get("k", 0)),
        "ela/depth_rel_tol": float(report.get("depth_rel_tol", 0.0)),
        "ela/residual_clip": float(report.get("residual_clip", 0.0)),
        "ela/target_frames": int(report.get("target_frames", 0)),
        "ela/mean_covered_fraction": float(report.get("mean_covered_fraction", 0.0)),
        "ela/mean_confidence": float(report.get("mean_confidence", 0.0)),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model = Path(args.base_model_path)
    output_model = Path(args.output_model_path) if args.output_model_path else base_model
    base_method = f"ours_{args.iteration}"
    method_name = args.method_name or f"ours_{args.iteration}_ela_k{args.k}"

    train_frames = load_split_frames(base_model, "train", base_method)
    target_frames = load_split_frames(base_model, args.target_split, base_method)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    policy, calibration, policy_candidates = _choose_policy(args, train_frames, alpha_grid, device)
    alpha = float(args.alpha) if args.alpha >= 0.0 else float(calibration["alpha"])
    out_method = output_model / args.target_split / method_name
    out_render = out_method / "renders"
    out_gt = out_method / "gt"
    out_render.mkdir(parents=True, exist_ok=True)
    _copy_gt(target_frames, out_gt)

    loader = FrameLoader(device=device)
    infos = []
    for target in tqdm(target_frames, desc=f"ELA {args.target_split}"):
        adapted, info = adapt_frame(
            target,
            train_frames,
            k=int(policy["k"]),
            alpha=alpha,
            mode=str(policy["mode"]),
            residual_clip=float(policy["residual_clip"]),
            min_confidence=args.min_confidence,
            depth_abs_tol=float(policy["depth_abs_tol"]),
            depth_rel_tol=float(policy["depth_rel_tol"]),
            direction_weight=args.direction_weight,
            loader=loader,
            device=device,
        )
        save_image_tensor(adapted, out_render / target.render_path.name)
        infos.append({"frame": target.name, **info})

    report = {
        "method": "Evidence Lumigraph Adapter",
        "base_model_path": str(base_model),
        "output_model_path": str(output_model),
        "base_method": base_method,
        "method_name": method_name,
        "target_split": args.target_split,
        "target_frames": len(target_frames),
        "train_support_frames": len(train_frames),
        "alpha": alpha,
        "alpha_source": "cli" if args.alpha >= 0.0 else "train_calibration",
        "calibration": calibration,
        "auto_policy": bool(args.auto_policy),
        "policy": policy,
        "policy_candidates": policy_candidates,
        "mode": str(policy["mode"]),
        "k": int(policy["k"]),
        "residual_clip": float(policy["residual_clip"]),
        "depth_abs_tol": float(policy["depth_abs_tol"]),
        "depth_rel_tol": float(policy["depth_rel_tol"]),
        "direction_weight": float(args.direction_weight),
        "mean_covered_fraction": float(sum(float(x["covered_fraction"]) for x in infos) / max(len(infos), 1)),
        "mean_confidence": float(sum(float(x["mean_confidence"]) for x in infos) / max(len(infos), 1)),
        "frames": infos,
    }
    (out_method / "ela_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ELA] Saved adapted renders: {out_render}")
    print(f"[ELA] alpha={alpha:.4f} covered={report['mean_covered_fraction']:.4f} confidence={report['mean_confidence']:.6f}")
    _maybe_wandb(args, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply train-only geometry-aware residual evidence lumigraph adapter.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_model_path", default="")
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--method_name", default="")
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--mode", choices=("residual", "color"), default="residual")
    parser.add_argument("--auto_policy", action="store_true")
    parser.add_argument("--policy_modes", default="residual,color")
    parser.add_argument("--policy_k_values", default="4,8")
    parser.add_argument("--policy_depth_rel_values", default="0.06,0.12")
    parser.add_argument("--policy_residual_clip_values", default="0.25")
    parser.add_argument("--alpha", default=-1.0, type=float, help="Override alpha. Default <0 uses train-only calibration.")
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--calib_stride", default=16, type=int)
    parser.add_argument("--calib_max_views", default=16, type=int)
    parser.add_argument("--residual_clip", default=0.25, type=float)
    parser.add_argument("--min_confidence", default=1e-4, type=float)
    parser.add_argument("--depth_abs_tol", default=0.02, type=float)
    parser.add_argument("--depth_rel_tol", default=0.03, type=float)
    parser.add_argument("--direction_weight", default=0.35, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
