#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_lumigraph_adapter import save_image_tensor
from utils.loss_utils import ssim


def _parse_float_grid(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _method_dir(model_path: Path, split: str, method: str) -> Path:
    path = model_path / split / method
    if not (path / "renders").is_dir():
        raise FileNotFoundError(f"Missing renders: {path / 'renders'}")
    if not (path / "gt").is_dir():
        raise FileNotFoundError(f"Missing gt: {path / 'gt'}")
    return path


def _read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).to(device=device, dtype=torch.float32)


def _sample_paths(paths: list[Path], max_views: int, sampler: str) -> list[Path]:
    if not paths:
        return []
    if int(max_views) <= 0 or len(paths) <= int(max_views):
        return paths
    if sampler == "first":
        return paths[: int(max_views)]
    if sampler != "uniform":
        raise ValueError(f"Unsupported calibration sampler: {sampler}")
    if int(max_views) == 1:
        return [paths[len(paths) // 2]]
    raw = torch.linspace(0, len(paths) - 1, steps=int(max_views)).round().to(torch.int64).tolist()
    selected = []
    seen: set[int] = set()
    for idx in raw:
        if idx in seen:
            continue
        seen.add(int(idx))
        selected.append(paths[int(idx)])
    return selected


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _calibrate_weight(
    safe_dir: Path,
    broad_dir: Path,
    weights: list[float],
    *,
    objective: str,
    ssim_weight: float,
    lpips_weight: float,
    compute_lpips: bool,
    min_psnr_gain: float,
    min_ssim_gain: float,
    min_lpips_gain: float,
    calib_max_views: int,
    calib_sampler: str,
    device: torch.device,
) -> dict:
    render_paths = sorted((safe_dir / "renders").glob("*.png"))
    render_paths = _sample_paths(render_paths, calib_max_views, calib_sampler)
    lpips_model = None
    if compute_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)
    rows = []
    best_weight = 0.0
    best_score = -float("inf")
    for weight in weights:
        mse_total = 0.0
        safe_mse_total = 0.0
        ssim_total = 0.0
        safe_ssim_total = 0.0
        lpips_total = 0.0
        safe_lpips_total = 0.0
        count = 0
        for safe_path in render_paths:
            broad_path = broad_dir / "renders" / safe_path.name
            gt_path = safe_dir / "gt" / safe_path.name
            if not broad_path.is_file() or not gt_path.is_file():
                continue
            safe = _read_rgb(safe_path, device)
            broad = _read_rgb(broad_path, device)
            gt = _read_rgb(gt_path, device)
            pred = torch.clamp(safe * (1.0 - float(weight)) + broad * float(weight), 0.0, 1.0)
            mse_total += float(torch.mean((pred - gt) ** 2).detach().cpu().item())
            safe_mse_total += float(torch.mean((safe - gt) ** 2).detach().cpu().item())
            pred_ssim = float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
            safe_ssim = float(ssim(safe.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
            ssim_total += pred_ssim
            safe_ssim_total += safe_ssim
            if lpips_model is not None:
                with torch.no_grad():
                    lpips_total += float(lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
                    safe_lpips_total += float(lpips_model(safe.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
            count += 1
        count = max(count, 1)
        mse = mse_total / count
        safe_mse = safe_mse_total / count
        mean_ssim = ssim_total / count
        mean_safe_ssim = safe_ssim_total / count
        mean_lpips = lpips_total / count if lpips_model is not None else 0.0
        mean_safe_lpips = safe_lpips_total / count if lpips_model is not None else 0.0
        psnr_gain = _psnr(mse) - _psnr(safe_mse)
        ssim_gain = mean_ssim - mean_safe_ssim
        lpips_gain = mean_safe_lpips - mean_lpips if lpips_model is not None else 0.0
        score = psnr_gain
        if objective == "balanced":
            score += float(ssim_weight) * ssim_gain
            if lpips_model is not None:
                score += float(lpips_weight) * lpips_gain
        pareto_pass = (psnr_gain >= float(min_psnr_gain)) and (ssim_gain >= float(min_ssim_gain))
        if lpips_model is not None:
            pareto_pass = pareto_pass and (lpips_gain >= float(min_lpips_gain))
        if not pareto_pass:
            score = -1e30
        row = {
            "weight": float(weight),
            "views": count,
            "psnr": _psnr(mse),
            "safe_psnr": _psnr(safe_mse),
            "psnr_gain": psnr_gain,
            "ssim": mean_ssim,
            "safe_ssim": mean_safe_ssim,
            "ssim_gain": ssim_gain,
            "lpips": mean_lpips if lpips_model is not None else None,
            "safe_lpips": mean_safe_lpips if lpips_model is not None else None,
            "lpips_gain": lpips_gain if lpips_model is not None else None,
            "pareto_pass": bool(pareto_pass),
            "selection_score": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    zero_row = next((row for row in rows if abs(float(row["weight"])) < 1e-9), None)
    if zero_row is not None and best_score <= float(zero_row["selection_score"]):
        best_weight = 0.0
    return {
        "weight": best_weight,
        "rows": rows,
        "calibration_views": [path.stem for path in render_paths],
        "objective": objective,
        "ssim_weight": float(ssim_weight),
        "lpips_weight": float(lpips_weight),
        "min_psnr_gain": float(min_psnr_gain),
        "min_ssim_gain": float(min_ssim_gain),
        "min_lpips_gain": float(min_lpips_gain),
        "compute_lpips": bool(compute_lpips),
    }


def _copy_blend(safe_dir: Path, broad_dir: Path, out_dir: Path, weight: float, device: torch.device) -> int:
    out_renders = out_dir / "renders"
    out_gt = out_dir / "gt"
    out_renders.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    count = 0
    for safe_path in sorted((safe_dir / "renders").glob("*.png")):
        broad_path = broad_dir / "renders" / safe_path.name
        gt_path = safe_dir / "gt" / safe_path.name
        if not broad_path.is_file() or not gt_path.is_file():
            continue
        safe = _read_rgb(safe_path, device)
        broad = _read_rgb(broad_path, device)
        pred = torch.clamp(safe * (1.0 - float(weight)) + broad * float(weight), 0.0, 1.0)
        save_image_tensor(pred, out_renders / safe_path.name)
        shutil.copy2(gt_path, out_gt / safe_path.name)
        count += 1
    return count


def _maybe_wandb(args: argparse.Namespace, report: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[Portfolio] W&B unavailable, skipping log: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
        config=vars(args),
    )
    flat = {
        "portfolio/weight": float(report["weight"]),
        "portfolio/target_frames": int(report["target_frames"]),
        "portfolio/calibration_views": int(len(report.get("calibration", {}).get("calibration_views", []))),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-calibrated portfolio blend for two ELA render methods.")
    parser.add_argument("--safe_model_path", required=True)
    parser.add_argument("--safe_method_name", required=True)
    parser.add_argument("--broad_model_path", required=True)
    parser.add_argument("--broad_method_name", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--target_split", default="test")
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--weight_grid", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", default=40.0, type=float)
    parser.add_argument("--lpips_weight", default=40.0, type=float)
    parser.add_argument("--min_psnr_gain", default=-1e9, type=float)
    parser.add_argument("--min_ssim_gain", default=-1e9, type=float)
    parser.add_argument("--min_lpips_gain", default=-1e9, type=float)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--calib_max_views", default=16, type=int)
    parser.add_argument("--calib_sampler", choices=("first", "uniform"), default="uniform")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    safe_root = Path(args.safe_model_path)
    broad_root = Path(args.broad_model_path)
    safe_calib = _method_dir(safe_root, args.calib_split, args.safe_method_name)
    broad_calib = _method_dir(broad_root, args.calib_split, args.broad_method_name)
    safe_target = _method_dir(safe_root, args.target_split, args.safe_method_name)
    broad_target = _method_dir(broad_root, args.target_split, args.broad_method_name)
    calibration = _calibrate_weight(
        safe_calib,
        broad_calib,
        _parse_float_grid(args.weight_grid),
        objective=args.objective,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        compute_lpips=args.calib_lpips,
        min_psnr_gain=args.min_psnr_gain,
        min_ssim_gain=args.min_ssim_gain,
        min_lpips_gain=args.min_lpips_gain,
        calib_max_views=args.calib_max_views,
        calib_sampler=args.calib_sampler,
        device=device,
    )
    weight = float(calibration["weight"])
    out_method = Path(args.output_model_path) / args.target_split / args.method_name
    target_frames = _copy_blend(safe_target, broad_target, out_method, weight, device)
    report = {
        "method": "Evidence Portfolio Adapter",
        "safe_model_path": str(safe_root),
        "safe_method_name": args.safe_method_name,
        "broad_model_path": str(broad_root),
        "broad_method_name": args.broad_method_name,
        "output_model_path": args.output_model_path,
        "method_name": args.method_name,
        "target_split": args.target_split,
        "calib_split": args.calib_split,
        "weight": weight,
        "target_frames": target_frames,
        "calibration": calibration,
    }
    out_method.mkdir(parents=True, exist_ok=True)
    (out_method / "portfolio_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[Portfolio] saved {target_frames} blended renders at weight={weight:.4f}: {out_method / 'renders'}")
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
