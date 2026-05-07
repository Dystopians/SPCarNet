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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.loss_utils import ssim


def _parse_float_grid(text: str) -> list[float]:
    return sorted({float(x.strip()) for x in text.split(",") if x.strip()})


def _parse_int_grid(text: str) -> list[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = max(int(item), 1)
        if value % 2 == 0:
            value += 1
        out.append(value)
    return sorted(set(out)) or [1]


def _method_dir(model: Path, split: str, method: str) -> Path:
    path = model / split / method
    if not (path / "renders").is_dir():
        raise FileNotFoundError(f"Missing renders: {path / 'renders'}")
    if not (path / "gt").is_dir():
        raise FileNotFoundError(f"Missing gt: {path / 'gt'}")
    return path


def _common_names(*dirs: Path) -> list[str]:
    sets = []
    for directory in dirs:
        sets.append({p.name for p in (directory / "renders").glob("*.png")})
    return sorted(set.intersection(*sets)) if sets else []


def _sample_names(names: list[str], max_views: int, sampler: str) -> list[str]:
    if int(max_views) <= 0 or len(names) <= int(max_views):
        return list(names)
    if sampler == "first":
        return names[: int(max_views)]
    if sampler != "uniform":
        raise ValueError(f"Unsupported sampler: {sampler}")
    if int(max_views) == 1:
        return [names[len(names) // 2]]
    raw = torch.linspace(0, len(names) - 1, steps=int(max_views)).round().to(torch.int64).tolist()
    out = []
    seen = set()
    for idx in raw:
        if int(idx) in seen:
            continue
        seen.add(int(idx))
        out.append(names[int(idx)])
    return out


def _read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).to(device=device, dtype=torch.float32)


def _save_rgb(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(torch.clamp(image.detach().cpu(), 0.0, 1.0)).save(path)


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _structure_score(safe: torch.Tensor, parent: torch.Tensor, kernels: list[int]) -> torch.Tensor:
    diff = torch.mean(torch.abs(safe - parent), dim=0, keepdim=True)
    scores = [diff]
    x = diff.unsqueeze(0)
    for kernel in kernels:
        if int(kernel) <= 1:
            continue
        pooled = F.avg_pool2d(x, kernel_size=int(kernel), stride=1, padding=int(kernel) // 2).squeeze(0)
        scores.append(pooled)
    return torch.max(torch.stack(scores, dim=0), dim=0).values


def _make_mask(score: torch.Tensor, threshold: float, softness: float, dilate: int, max_blend: float) -> torch.Tensor:
    if float(softness) <= 0.0:
        mask = (score >= float(threshold)).to(dtype=torch.float32)
    else:
        mask = torch.sigmoid((score - float(threshold)) / max(float(softness), 1e-8))
    radius = max(int(dilate), 0)
    if radius > 0:
        kernel = 2 * radius + 1
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
    return torch.clamp(mask * float(max_blend), 0.0, 1.0)


def _blend(safe: torch.Tensor, candidate: torch.Tensor, parent: torch.Tensor, policy: dict) -> tuple[torch.Tensor, dict[str, float]]:
    frame_distance = float(torch.mean(torch.abs(safe - parent)).detach().cpu().item())
    score = _structure_score(safe, parent, [int(x) for x in policy["kernels"]])
    if frame_distance < float(policy.get("frame_threshold", 0.0)):
        mask = torch.zeros_like(score)
    else:
        mask = _make_mask(
            score,
            threshold=float(policy["threshold"]),
            softness=float(policy["softness"]),
            dilate=int(policy["dilate"]),
            max_blend=float(policy["max_blend"]),
        )
    out = torch.clamp(safe * (1.0 - mask) + candidate * mask, 0.0, 1.0)
    return out, {
        "frame_distance": frame_distance,
        "mask_mean": float(mask.mean().detach().cpu().item()),
        "mask_p95": float(torch.quantile(mask.reshape(-1).detach().float(), 0.95).cpu().item()),
        "score_mean": float(score.mean().detach().cpu().item()),
        "score_p95": float(torch.quantile(score.reshape(-1).detach().float(), 0.95).cpu().item()),
    }


def _metrics(pred: torch.Tensor, gt: torch.Tensor, lpips_model=None) -> dict[str, float]:
    mse = float(torch.mean((pred - gt) ** 2).detach().cpu().item())
    out = {
        "PSNR": _psnr_from_mse(mse),
        "SSIM": float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item()),
    }
    if lpips_model is not None:
        with torch.no_grad():
            out["LPIPS"] = float(lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
    return out


def _score_gain(pred: dict[str, float], safe: dict[str, float], objective: str, ssim_weight: float, lpips_weight: float) -> float:
    gain = float(pred["PSNR"] - safe["PSNR"])
    if objective == "balanced":
        gain += float(ssim_weight) * float(pred["SSIM"] - safe["SSIM"])
        if "LPIPS" in pred:
            gain += float(lpips_weight) * float(safe["LPIPS"] - pred["LPIPS"])
    return gain


def _calibrate(args: argparse.Namespace, safe_dir: Path, candidate_dir: Path, parent_dir: Path, names: list[str], device: torch.device) -> dict:
    if not names:
        raise RuntimeError("No common calibration renders across safe, candidate, and parent methods")
    lpips_model = None
    if args.calib_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)

    frames = []
    kernels = _parse_int_grid(args.local_kernels)
    for name in names:
        safe = _read_rgb(safe_dir / "renders" / name, device)
        candidate = _read_rgb(candidate_dir / "renders" / name, device)
        parent = _read_rgb(parent_dir / "renders" / name, device)
        gt = _read_rgb(safe_dir / "gt" / name, device)
        safe_row = _metrics(safe, gt, lpips_model)
        frames.append((name, safe, candidate, parent, gt, safe_row))

    rows = []
    best: tuple[tuple[bool, float, float, float], dict] | None = None
    frame_thresholds = _parse_float_grid(args.frame_threshold_grid) if args.frame_threshold_grid.strip() else []
    if float(args.frame_threshold_quantile) >= 0.0:
        distances = sorted(
            float(torch.mean(torch.abs(safe - parent)).detach().cpu().item())
            for _, safe, _, parent, _, _ in frames
        )
        q = min(max(float(args.frame_threshold_quantile), 0.0), 1.0)
        index = min(max(int(round(q * (len(distances) - 1))), 0), len(distances) - 1)
        frame_thresholds.append(float(distances[index]))
    frame_thresholds = sorted(set(frame_thresholds or [0.0]))
    for frame_threshold in frame_thresholds:
        for threshold in _parse_float_grid(args.threshold_grid):
            for softness in _parse_float_grid(args.softness_grid):
                for max_blend in _parse_float_grid(args.max_blend_grid):
                    policy = {
                        "frame_threshold": float(frame_threshold),
                        "threshold": float(threshold),
                        "softness": float(softness),
                        "max_blend": float(max_blend),
                        "dilate": int(args.mask_dilate),
                        "kernels": kernels,
                    }
                    pred_rows = []
                    safe_rows = []
                    frame_gains = []
                    mask_means = []
                    skipped = 0
                    for _, safe, candidate, parent, gt, safe_row in frames:
                        pred, info = _blend(safe, candidate, parent, policy)
                        pred_row = _metrics(pred, gt, lpips_model)
                        pred_rows.append(pred_row)
                        safe_rows.append(safe_row)
                        frame_gains.append(
                            _score_gain(pred_row, safe_row, args.objective, args.ssim_weight, args.lpips_weight)
                        )
                        mask_means.append(float(info["mask_mean"]))
                        skipped += int(float(info["mask_mean"]) <= 1e-8)
                    count = max(len(pred_rows), 1)
                    mean = {key: sum(row[key] for row in pred_rows) / count for key in pred_rows[0].keys()}
                    safe_mean = {key: sum(row[key] for row in safe_rows) / count for key in safe_rows[0].keys()}
                    gains = {
                        "d_psnr": mean["PSNR"] - safe_mean["PSNR"],
                        "d_ssim": mean["SSIM"] - safe_mean["SSIM"],
                        "d_lpips": safe_mean.get("LPIPS", 0.0) - mean.get("LPIPS", 0.0),
                    }
                    score = gains["d_psnr"]
                    if args.objective == "balanced":
                        score += float(args.ssim_weight) * gains["d_ssim"]
                        if "LPIPS" in mean:
                            score += float(args.lpips_weight) * gains["d_lpips"]
                    sorted_gains = sorted(frame_gains)
                    p05_index = min(max(int(math.floor(0.05 * (len(sorted_gains) - 1))), 0), len(sorted_gains) - 1)
                    p05 = float(sorted_gains[p05_index])
                    mask_mean = sum(mask_means) / max(len(mask_means), 1)
                    pass_gate = (
                        mask_mean >= float(args.min_mask_mean)
                        and gains["d_psnr"] >= float(args.min_mean_psnr_gain)
                        and gains["d_ssim"] >= float(args.min_mean_ssim_gain)
                        and gains["d_lpips"] >= float(args.min_mean_lpips_gain)
                        and p05 >= float(args.min_p05_score_gain)
                    )
                    row = {
                        **policy,
                        "calibration_views": len(frames),
                        "skipped_views": int(skipped),
                        "score": float(score),
                        "p05_frame_score_gain": p05,
                        "mean_mask": float(mask_mean),
                        "pass_gate": bool(pass_gate),
                        **mean,
                        **{f"safe_{key}": value for key, value in safe_mean.items()},
                        **gains,
                    }
                    rows.append(row)
                    rank = (
                        bool(pass_gate),
                        float(score),
                        -abs(float(mask_mean) - float(args.target_mask_mean)),
                        -float(frame_threshold),
                        -float(threshold),
                    )
                    if best is None or rank > best[0]:
                        best = (rank, row)
    selected = dict(best[1] if best is not None else rows[0])
    if not bool(selected.get("pass_gate", False)):
        zero = next((row for row in rows if abs(float(row["threshold"])) < 1e-12 and abs(float(row["max_blend"]) - 1.0) < 1e-12), None)
        if zero is not None:
            selected = dict(zero)
    return {
        "selected_policy": {
            "frame_threshold": float(selected["frame_threshold"]),
            "threshold": float(selected["threshold"]),
            "softness": float(selected["softness"]),
            "max_blend": float(selected["max_blend"]),
            "dilate": int(selected["dilate"]),
            "kernels": [int(x) for x in selected["kernels"]],
        },
        "selected_row": selected,
        "rows": rows,
    }


def _apply(args: argparse.Namespace, safe_dir: Path, candidate_dir: Path, parent_dir: Path, out_method: Path, policy: dict, device: torch.device) -> dict:
    names = _common_names(safe_dir, candidate_dir, parent_dir)
    if not names:
        raise RuntimeError("No common target renders across safe, candidate, and parent methods")
    out_render = out_method / "renders"
    out_gt = out_method / "gt"
    out_render.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    infos = []
    for name in names:
        safe = _read_rgb(safe_dir / "renders" / name, device)
        candidate = _read_rgb(candidate_dir / "renders" / name, device)
        parent = _read_rgb(parent_dir / "renders" / name, device)
        pred, info = _blend(safe, candidate, parent, policy)
        _save_rgb(pred, out_render / name)
        shutil.copy2(safe_dir / "gt" / name, out_gt / name)
        infos.append({"image": name, **info})
    return {
        "target_views": len(names),
        "mean_mask": float(sum(float(x["mask_mean"]) for x in infos) / max(len(infos), 1)),
        "mean_score": float(sum(float(x["score_mean"]) for x in infos) / max(len(infos), 1)),
        "frames": infos,
    }


def _maybe_wandb(args: argparse.Namespace, report: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[LocalParentGate] W&B unavailable: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
        config=vars(args),
    )
    selected = report["calibration"]["selected_row"]
    flat = {
        "local_parent_gate/threshold": float(selected["threshold"]),
        "local_parent_gate/frame_threshold": float(selected["frame_threshold"]),
        "local_parent_gate/softness": float(selected["softness"]),
        "local_parent_gate/max_blend": float(selected["max_blend"]),
        "local_parent_gate/calib_score": float(selected["score"]),
        "local_parent_gate/calib_mean_mask": float(selected["mean_mask"]),
        "local_parent_gate/target_mean_mask": float(report["application"]["mean_mask"]),
        "local_parent_gate/target_views": int(report["application"]["target_views"]),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-calibrated local parent-consistency gate for ELA render repair.")
    parser.add_argument("--safe_model_path", required=True)
    parser.add_argument("--safe_method_name", required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument("--candidate_method_name", required=True)
    parser.add_argument("--parent_model_path", required=True)
    parser.add_argument("--parent_method_name", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--target_split", default="test")
    parser.add_argument("--threshold_grid", default="0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.020")
    parser.add_argument("--frame_threshold_grid", default="0")
    parser.add_argument("--frame_threshold_quantile", default=-1.0, type=float)
    parser.add_argument("--softness_grid", default="0,0.002,0.004")
    parser.add_argument("--max_blend_grid", default="0.50,0.75,1.00")
    parser.add_argument("--local_kernels", default="1,9,25")
    parser.add_argument("--mask_dilate", type=int, default=0)
    parser.add_argument("--calib_max_views", type=int, default=64)
    parser.add_argument("--calib_sampler", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--min_mask_mean", type=float, default=0.01)
    parser.add_argument("--target_mask_mean", type=float, default=0.50)
    parser.add_argument("--min_mean_psnr_gain", type=float, default=-1e9)
    parser.add_argument("--min_mean_ssim_gain", type=float, default=-1e9)
    parser.add_argument("--min_mean_lpips_gain", type=float, default=-1e9)
    parser.add_argument("--min_p05_score_gain", type=float, default=-1e9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    safe_root = Path(args.safe_model_path)
    candidate_root = Path(args.candidate_model_path)
    parent_root = Path(args.parent_model_path)
    safe_calib = _method_dir(safe_root, args.calib_split, args.safe_method_name)
    candidate_calib = _method_dir(candidate_root, args.calib_split, args.candidate_method_name)
    parent_calib = _method_dir(parent_root, args.calib_split, args.parent_method_name)
    safe_target = _method_dir(safe_root, args.target_split, args.safe_method_name)
    candidate_target = _method_dir(candidate_root, args.target_split, args.candidate_method_name)
    parent_target = _method_dir(parent_root, args.target_split, args.parent_method_name)

    calib_names = _sample_names(
        _common_names(safe_calib, candidate_calib, parent_calib),
        args.calib_max_views,
        args.calib_sampler,
    )
    calibration = _calibrate(args, safe_calib, candidate_calib, parent_calib, calib_names, device)
    out_method = Path(args.output_model_path) / args.target_split / args.method_name
    application = _apply(args, safe_target, candidate_target, parent_target, out_method, calibration["selected_policy"], device)
    report = {
        "method": "Local Parent-Consistency Evidence Gate",
        "safe_model_path": args.safe_model_path,
        "safe_method_name": args.safe_method_name,
        "candidate_model_path": args.candidate_model_path,
        "candidate_method_name": args.candidate_method_name,
        "parent_model_path": args.parent_model_path,
        "parent_method_name": args.parent_method_name,
        "output_model_path": args.output_model_path,
        "method_name": args.method_name,
        "calib_split": args.calib_split,
        "target_split": args.target_split,
        "calibration": calibration,
        "application": application,
    }
    out_method.mkdir(parents=True, exist_ok=True)
    (out_method / "local_parent_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_policy": calibration["selected_policy"],
                "target_views": application["target_views"],
                "target_mean_mask": application["mean_mask"],
                "output": str(out_method),
            },
            indent=2,
        )
    )
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
