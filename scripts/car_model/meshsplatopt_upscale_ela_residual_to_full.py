#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def _read_rgb(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).to(dtype=torch.float32)


def _save_rgb(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(torch.clamp(image.detach().cpu(), 0.0, 1.0)).save(path)


def _images(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing image directory: {path}")
    return {p.name: p for p in sorted(path.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}


def _parse_alpha_grid(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _uniform_subset(names: list[str], max_views: int) -> list[str]:
    if max_views <= 0 or len(names) <= max_views:
        return names
    if max_views == 1:
        return [names[len(names) // 2]]
    idxs = torch.linspace(0, len(names) - 1, steps=max_views).round().long().tolist()
    return [names[int(i)] for i in sorted(set(idxs))]


def _up_delta(low_base_path: Path, low_ela_path: Path, full_shape: tuple[int, int], mode: str) -> torch.Tensor:
    low_delta = _read_rgb(low_ela_path) - _read_rgb(low_base_path)
    return F.interpolate(
        low_delta.unsqueeze(0),
        size=full_shape,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    ).squeeze(0)


def _calibrate_alpha(args: argparse.Namespace, model: Path) -> tuple[float, dict]:
    split = args.calib_split
    low_base_dir = model / split / args.lowres_base_method
    low_ela_dir = model / split / args.lowres_ela_method
    full_base_dir = model / split / args.full_base_method
    low_base = _images(low_base_dir / "renders")
    low_ela = _images(low_ela_dir / "renders")
    full_base = _images(full_base_dir / "renders")
    full_gt = _images(full_base_dir / "gt")
    names = sorted(set(low_base) & set(low_ela) & set(full_base) & set(full_gt))
    names = _uniform_subset(names, int(args.calib_max_views))
    if not names:
        raise RuntimeError(f"No calibration frames for split={split}")

    alphas = _parse_alpha_grid(args.alpha_grid)
    accum = {
        alpha: {"ssim": [], "psnr": [], "lpips": []}
        for alpha in alphas
    }
    device = torch.device(args.device)
    with torch.no_grad():
        for name in names:
            full = _read_rgb(full_base[name]).to(device)
            gt = _read_rgb(full_gt[name]).to(device)
            delta = _up_delta(
                low_base[name],
                low_ela[name],
                tuple(full.shape[-2:]),
                args.resize_mode,
            ).to(device)
            gt_b = gt.unsqueeze(0)
            for alpha in alphas:
                pred = torch.clamp(full + float(alpha) * delta, 0.0, 1.0).unsqueeze(0)
                accum[alpha]["ssim"].append(float(ssim(pred, gt_b).detach().cpu().item()))
                accum[alpha]["psnr"].append(float(psnr(pred, gt_b).detach().cpu().item()))
                if args.calib_lpips:
                    accum[alpha]["lpips"].append(float(lpips(pred, gt_b, net_type="vgg").detach().cpu().item()))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = []
    for alpha in alphas:
        vals = accum[alpha]
        row = {
            "alpha": float(alpha),
            "views": len(names),
            "ssim": float(sum(vals["ssim"]) / max(len(vals["ssim"]), 1)),
            "psnr": float(sum(vals["psnr"]) / max(len(vals["psnr"]), 1)),
            "lpips": (
                float(sum(vals["lpips"]) / max(len(vals["lpips"]), 1))
                if args.calib_lpips
                else 0.0
            ),
        }
        row["selection_score"] = (
            row["psnr"]
            + float(args.policy_ssim_weight) * row["ssim"]
            - (float(args.policy_lpips_weight) * row["lpips"] if args.calib_lpips else 0.0)
        )
        rows.append(row)

    base = next((row for row in rows if abs(float(row["alpha"])) < 1e-12), rows[0])
    candidates = rows
    if args.strict_all_axis_alpha:
        min_psnr = base["psnr"] + float(args.strict_alpha_min_psnr_gain)
        min_ssim = base["ssim"] + float(args.strict_alpha_min_ssim_gain)
        max_lpips = base["lpips"] - float(args.strict_alpha_min_lpips_gain)
        candidates = [
            row
            for row in rows
            if row["psnr"] >= min_psnr - 1e-8
            and row["ssim"] >= min_ssim - 1e-8
            and (not args.calib_lpips or row["lpips"] <= max_lpips + 1e-8)
        ]
        if not candidates:
            candidates = [base]
    pre_peak_filter_count = len(candidates)
    if float(args.alpha_ssim_peak_tolerance) >= 0.0 and candidates:
        peak_ssim = max(float(row["ssim"]) for row in candidates)
        tolerance = float(args.alpha_ssim_peak_tolerance)
        peak_candidates = [
            row
            for row in candidates
            if float(row["ssim"]) >= peak_ssim - tolerance - 1e-8
        ]
        if peak_candidates:
            candidates = peak_candidates
    best = max(candidates, key=lambda row: (row["selection_score"], row["psnr"], -row["lpips"]))
    report = {
        "enabled": True,
        "calib_split": split,
        "calib_frames": names,
        "strict_all_axis_alpha": bool(args.strict_all_axis_alpha),
        "strict_alpha_min_psnr_gain": float(args.strict_alpha_min_psnr_gain),
        "strict_alpha_min_ssim_gain": float(args.strict_alpha_min_ssim_gain),
        "strict_alpha_min_lpips_gain": float(args.strict_alpha_min_lpips_gain),
        "alpha_ssim_peak_tolerance": float(args.alpha_ssim_peak_tolerance),
        "candidate_count_before_peak_filter": int(pre_peak_filter_count),
        "candidate_count_after_peak_filter": int(len(candidates)),
        "selected_alpha": float(best["alpha"]),
        "selected_score": float(best["selection_score"]),
        "base_score": float(base["selection_score"]),
        "rows": rows,
    }
    return float(best["alpha"]), report


def run(args: argparse.Namespace) -> dict:
    model = Path(args.model_path)
    split = args.split
    low_base_dir = model / split / args.lowres_base_method
    low_ela_dir = model / split / args.lowres_ela_method
    full_base_dir = model / split / args.full_base_method
    out_dir = model / split / args.output_method
    out_renders = out_dir / "renders"
    out_gt = out_dir / "gt"
    out_renders.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    low_base = _images(low_base_dir / "renders")
    low_ela = _images(low_ela_dir / "renders")
    full_base = _images(full_base_dir / "renders")
    full_gt = _images(full_base_dir / "gt")
    residual_names = sorted(set(low_base) & set(low_ela) & set(full_base))
    full_names = sorted(full_base)
    if not residual_names:
        raise RuntimeError(
            f"No common render names across {low_base_dir}, {low_ela_dir}, and {full_base_dir}"
        )

    alpha = float(args.alpha)
    calibration_report = {"enabled": False}
    if args.auto_alpha:
        alpha, calibration_report = _calibrate_alpha(args, model)

    residual_norms: list[float] = []
    fallback = 0
    for name in full_names:
        full = _read_rgb(full_base[name])
        if name in low_base and name in low_ela:
            low_delta = _read_rgb(low_ela[name]) - _read_rgb(low_base[name])
            up_delta = F.interpolate(
                low_delta.unsqueeze(0),
                size=tuple(full.shape[-2:]),
                mode=args.resize_mode,
                align_corners=False if args.resize_mode in {"bilinear", "bicubic"} else None,
            ).squeeze(0)
            out = torch.clamp(full + alpha * up_delta, 0.0, 1.0)
            residual_norms.append(float(torch.mean(torch.abs(up_delta)).item()))
        else:
            out = full
            fallback += 1
        _save_rgb(out, out_renders / name)
        if name in full_gt:
            shutil.copy2(full_gt[name], out_gt / name)

    report = {
        "method": "Low-resolution ELA residual upsample",
        "model_path": str(model),
        "split": split,
        "lowres_base_method": args.lowres_base_method,
        "lowres_ela_method": args.lowres_ela_method,
        "full_base_method": args.full_base_method,
        "output_method": args.output_method,
        "frames": len(full_names),
        "residual_frames": len(residual_names),
        "fallback_frames": int(fallback),
        "alpha": float(alpha),
        "requested_alpha": float(args.alpha),
        "auto_alpha": calibration_report,
        "resize_mode": str(args.resize_mode),
        "mean_abs_upscaled_residual": float(sum(residual_norms) / max(len(residual_norms), 1)),
    }
    (out_dir / "ela_upscale_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale low-resolution ELA residuals onto full-resolution renders.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--lowres_base_method", required=True)
    parser.add_argument("--lowres_ela_method", required=True)
    parser.add_argument("--full_base_method", required=True)
    parser.add_argument("--output_method", required=True)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--auto_alpha", action="store_true")
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--calib_max_views", default=16, type=int)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--strict_all_axis_alpha", action="store_true")
    parser.add_argument("--strict_alpha_min_psnr_gain", default=0.0, type=float)
    parser.add_argument("--strict_alpha_min_ssim_gain", default=0.0, type=float)
    parser.add_argument("--strict_alpha_min_lpips_gain", default=0.0, type=float)
    parser.add_argument(
        "--alpha_ssim_peak_tolerance",
        default=-1.0,
        type=float,
        help=(
            "When non-negative, keep only train-calibrated alpha candidates whose SSIM is within "
            "this tolerance of the best candidate SSIM before applying the scalar policy score."
        ),
    )
    parser.add_argument("--policy_ssim_weight", default=20.0, type=float)
    parser.add_argument("--policy_lpips_weight", default=20.0, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize_mode", default="bilinear", choices=("nearest", "bilinear", "bicubic"))
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
