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

from utils.loss_utils import ssim


def _parse_float_grid(text: str) -> list[float]:
    return sorted({float(x.strip()) for x in text.split(",") if x.strip()})


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


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _parent_distance(safe_dir: Path, parent_dir: Path, name: str, device: torch.device) -> float:
    safe = _read_rgb(safe_dir / "renders" / name, device)
    parent = _read_rgb(parent_dir / "renders" / name, device)
    return float(torch.mean(torch.abs(safe - parent)).detach().cpu().item())


def _metrics(pred: torch.Tensor, gt: torch.Tensor, lpips_model=None) -> dict[str, float]:
    mse = float(torch.mean((pred - gt) ** 2).detach().cpu().item())
    row = {
        "PSNR": _psnr_from_mse(mse),
        "SSIM": float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item()),
    }
    if lpips_model is not None:
        with torch.no_grad():
            row["LPIPS"] = float(lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
    return row


def _calibrate(
    safe_dir: Path,
    candidate_dir: Path,
    parent_dir: Path,
    thresholds: list[float],
    names: list[str],
    *,
    objective: str,
    ssim_weight: float,
    lpips_weight: float,
    min_used_fraction: float,
    min_mean_psnr_gain: float,
    min_mean_ssim_gain: float,
    min_mean_lpips_gain: float,
    min_p05_score_gain: float,
    lpips_model,
    device: torch.device,
) -> dict:
    rows = []
    best = None
    parent_distances = {name: _parent_distance(safe_dir, parent_dir, name, device) for name in names}
    for threshold in thresholds:
        selected_metrics = []
        safe_metrics = []
        frame_score_gains = []
        used = 0
        for name in names:
            use_candidate = parent_distances[name] >= float(threshold)
            pred_dir = candidate_dir if use_candidate else safe_dir
            if use_candidate:
                used += 1
            pred = _read_rgb(pred_dir / "renders" / name, device)
            safe = _read_rgb(safe_dir / "renders" / name, device)
            gt = _read_rgb(safe_dir / "gt" / name, device)
            pred_row = _metrics(pred, gt, lpips_model)
            safe_row = _metrics(safe, gt, lpips_model)
            selected_metrics.append(pred_row)
            safe_metrics.append(safe_row)
            score_gain = pred_row["PSNR"] - safe_row["PSNR"]
            if objective == "balanced":
                score_gain += float(ssim_weight) * (pred_row["SSIM"] - safe_row["SSIM"])
                if "LPIPS" in pred_row:
                    score_gain += float(lpips_weight) * (safe_row["LPIPS"] - pred_row["LPIPS"])
            frame_score_gains.append(score_gain)
        count = max(len(selected_metrics), 1)
        mean = {
            key: sum(row[key] for row in selected_metrics) / count
            for key in selected_metrics[0].keys()
        }
        safe_mean = {
            key: sum(row[key] for row in safe_metrics) / count
            for key in safe_metrics[0].keys()
        }
        gains = {
            "d_psnr": mean["PSNR"] - safe_mean["PSNR"],
            "d_ssim": mean["SSIM"] - safe_mean["SSIM"],
            "d_lpips": safe_mean.get("LPIPS", 0.0) - mean.get("LPIPS", 0.0),
        }
        score = gains["d_psnr"]
        if objective == "balanced":
            score += float(ssim_weight) * gains["d_ssim"]
            if "LPIPS" in mean:
                score += float(lpips_weight) * gains["d_lpips"]
        sorted_scores = sorted(frame_score_gains)
        p05_index = min(max(int(math.floor(0.05 * (len(sorted_scores) - 1))), 0), len(sorted_scores) - 1)
        p05 = float(sorted_scores[p05_index]) if sorted_scores else 0.0
        used_fraction = float(used) / max(len(names), 1)
        pass_gate = (
            used_fraction >= float(min_used_fraction)
            and gains["d_psnr"] >= float(min_mean_psnr_gain)
            and gains["d_ssim"] >= float(min_mean_ssim_gain)
            and gains["d_lpips"] >= float(min_mean_lpips_gain)
            and p05 >= float(min_p05_score_gain)
        )
        row = {
            "threshold": float(threshold),
            "calibration_views": len(names),
            "used_candidate_views": int(used),
            "used_candidate_fraction": used_fraction,
            "score": float(score),
            "p05_frame_score_gain": p05,
            "pass_gate": bool(pass_gate),
            **mean,
            **{f"safe_{key}": value for key, value in safe_mean.items()},
            **gains,
        }
        rows.append(row)
        rank = (bool(pass_gate), float(score), -abs(float(threshold)))
        if best is None or rank > best[0]:
            best = (rank, row)
    selected = dict(best[1] if best is not None else rows[0])
    if not bool(selected.get("pass_gate", False)):
        zero = next((row for row in rows if abs(float(row["threshold"])) < 1e-12), None)
        if zero is not None:
            selected = dict(zero)
    return {
        "selected_threshold": float(selected["threshold"]),
        "selected_row": selected,
        "rows": rows,
        "parent_distance_summary": {
            "min": min(parent_distances.values()) if parent_distances else 0.0,
            "mean": sum(parent_distances.values()) / max(len(parent_distances), 1),
            "max": max(parent_distances.values()) if parent_distances else 0.0,
        },
    }


def _apply_gate(
    safe_dir: Path,
    candidate_dir: Path,
    parent_dir: Path,
    out_dir: Path,
    threshold: float,
    device: torch.device,
) -> dict:
    names = _common_names(safe_dir, candidate_dir, parent_dir)
    out_renders = out_dir / "renders"
    out_gt = out_dir / "gt"
    out_renders.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    used = []
    skipped = []
    for name in names:
        distance = _parent_distance(safe_dir, parent_dir, name, device)
        if distance >= float(threshold):
            src = candidate_dir / "renders" / name
            used.append({"image": name, "parent_distance": distance})
        else:
            src = safe_dir / "renders" / name
            skipped.append({"image": name, "parent_distance": distance})
        shutil.copy2(src, out_renders / name)
        shutil.copy2(safe_dir / "gt" / name, out_gt / name)
    return {
        "target_views": len(names),
        "used_candidate_views": len(used),
        "skipped_safe_views": len(skipped),
        "used_candidate": used,
        "skipped_safe": skipped,
    }


def _maybe_wandb(args: argparse.Namespace, report: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[ParentGate] W&B unavailable: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
        config=vars(args),
    )
    flat = {
        "parent_gate/threshold": float(report["calibration"]["selected_threshold"]),
        "parent_gate/target_views": int(report["application"]["target_views"]),
        "parent_gate/used_candidate_views": int(report["application"]["used_candidate_views"]),
        "parent_gate/used_candidate_fraction": float(report["application"]["used_candidate_views"])
        / max(int(report["application"]["target_views"]), 1),
        "parent_gate/calib_score": float(report["calibration"]["selected_row"].get("score", 0.0)),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-calibrated parent-consistency gate for ELA render repair.")
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
    parser.add_argument("--threshold_grid", default="0,0.012,0.014,0.016,0.017,0.018,0.020,0.024")
    parser.add_argument("--calib_max_views", type=int, default=0)
    parser.add_argument("--calib_sampler", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--min_used_fraction", type=float, default=0.0)
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
    calib_names = _common_names(safe_calib, candidate_calib, parent_calib)
    calib_names = _sample_names(calib_names, args.calib_max_views, args.calib_sampler)
    if not calib_names:
        raise RuntimeError("No common calibration renders across safe, candidate, and parent methods")
    lpips_model = None
    if args.calib_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)
    calibration = _calibrate(
        safe_calib,
        candidate_calib,
        parent_calib,
        _parse_float_grid(args.threshold_grid),
        calib_names,
        objective=args.objective,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        min_used_fraction=args.min_used_fraction,
        min_mean_psnr_gain=args.min_mean_psnr_gain,
        min_mean_ssim_gain=args.min_mean_ssim_gain,
        min_mean_lpips_gain=args.min_mean_lpips_gain,
        min_p05_score_gain=args.min_p05_score_gain,
        lpips_model=lpips_model,
        device=device,
    )
    out_method = Path(args.output_model_path) / args.target_split / args.method_name
    application = _apply_gate(
        safe_target,
        candidate_target,
        parent_target,
        out_method,
        float(calibration["selected_threshold"]),
        device,
    )
    if int(application["target_views"]) <= 0:
        raise RuntimeError("No common target renders across safe, candidate, and parent methods")
    report = {
        "method": "Parent-Consistency Evidence Gate",
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
    (out_method / "parent_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_threshold": calibration["selected_threshold"],
                "target_views": application["target_views"],
                "used_candidate_views": application["used_candidate_views"],
                "output": str(out_method),
            },
            indent=2,
        )
    )
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
