#!/usr/bin/env python3
"""Train-calibrated render-realized parent gate for SPCarNet candidates.

The gate learns a fixed local blending policy on a calibration split with GT.
At target/test time it only reads parent and candidate renders; target GT is
copied for later evaluation but is not used for policy selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.loss_utils import ssim  # noqa: E402


def _parse_float_grid(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def _parse_int_grid(text: str) -> list[int]:
    out: list[int] = []
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
    sets = [{path.name for path in (directory / "renders").glob("*.png")} for directory in dirs]
    return sorted(set.intersection(*sets)) if sets else []


def _numeric_png_stem(name: str) -> int | None:
    try:
        return int(Path(name).stem)
    except ValueError:
        return None


def _filter_names_by_view_subset(names: list[str], subset: str) -> list[str]:
    if subset == "all":
        return list(names)
    if subset not in {"even", "odd"}:
        raise ValueError(f"Unsupported calibration view subset: {subset}")
    want_even = subset == "even"
    out: list[str] = []
    for name in names:
        value = _numeric_png_stem(name)
        if value is None:
            continue
        if (value % 2 == 0) == want_even:
            out.append(name)
    return out


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
    out: list[str] = []
    seen: set[int] = set()
    for idx in raw:
        value = int(idx)
        if value in seen:
            continue
        seen.add(value)
        out.append(names[value])
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(value) for value in values)
    index = min(max(int(round(min(max(float(q), 0.0), 1.0) * (len(values) - 1))), 0), len(values) - 1)
    return float(values[index])


def _read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB")).to(device=device, dtype=torch.float32)


def _save_rgb(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(torch.clamp(image.detach().cpu(), 0.0, 1.0)).save(path)


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def _candidate_parent_score(candidate: torch.Tensor, parent: torch.Tensor, kernels: list[int]) -> torch.Tensor:
    diff = torch.mean(torch.abs(candidate - parent), dim=0, keepdim=True)
    scores = [diff]
    batched = diff.unsqueeze(0)
    for kernel in kernels:
        if int(kernel) <= 1:
            continue
        pooled = F.avg_pool2d(batched, kernel_size=int(kernel), stride=1, padding=int(kernel) // 2).squeeze(0)
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


def _blend(candidate: torch.Tensor, parent: torch.Tensor, policy: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    score = _candidate_parent_score(candidate, parent, [int(value) for value in policy["kernels"]])
    frame_distance = float(torch.mean(torch.abs(candidate - parent)).detach().cpu().item())
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
    out = torch.clamp(parent * (1.0 - mask) + candidate * mask, 0.0, 1.0)
    flat_score = score.reshape(-1).detach().float()
    flat_mask = mask.reshape(-1).detach().float()
    return out, {
        "frame_distance": frame_distance,
        "mask_mean": float(flat_mask.mean().cpu().item()),
        "mask_p95": float(torch.quantile(flat_mask, 0.95).cpu().item()),
        "score_mean": float(flat_score.mean().cpu().item()),
        "score_p95": float(torch.quantile(flat_score, 0.95).cpu().item()),
    }


def _fallback_policy(kernels: list[int], calibration_views: int = 0) -> dict[str, Any]:
    return {
        "frame_threshold": 1e9,
        "threshold": 1e9,
        "softness": 0.0,
        "max_blend": 0.0,
        "dilate": 0,
        "kernels": [int(value) for value in kernels],
        "calibration_views": int(calibration_views),
        "candidate_used_views": 0,
        "score": 0.0,
        "p05_frame_score_gain": 0.0,
        "p05_d_psnr": 0.0,
        "p05_d_ssim": 0.0,
        "p05_d_lpips": 0.0,
        "p95_delta_mse": 0.0,
        "mean_mask": 0.0,
        "pass_gate": True,
        "fallback_to_parent": True,
    }


def _metrics(pred: torch.Tensor, gt: torch.Tensor, lpips_model=None) -> dict[str, float]:
    mse = float(torch.mean((pred - gt) ** 2).detach().cpu().item())
    out = {
        "MSE": mse,
        "PSNR": _psnr_from_mse(mse),
        "SSIM": float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item()),
    }
    if lpips_model is not None:
        with torch.no_grad():
            out["LPIPS"] = float(lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
    return out


def _score_gain(pred: dict[str, float], parent: dict[str, float], objective: str, ssim_weight: float, lpips_weight: float) -> float:
    gain = float(pred["PSNR"] - parent["PSNR"])
    if objective == "balanced":
        gain += float(ssim_weight) * float(pred["SSIM"] - parent["SSIM"])
        if "LPIPS" in pred:
            gain += float(lpips_weight) * float(parent["LPIPS"] - pred["LPIPS"])
    return gain


def _load_calib_frames(
    parent_dir: Path,
    candidate_dir: Path,
    names: list[str],
    device: torch.device,
    lpips_model,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for name in names:
        parent = _read_rgb(parent_dir / "renders" / name, device)
        candidate = _read_rgb(candidate_dir / "renders" / name, device)
        gt = _read_rgb(parent_dir / "gt" / name, device)
        if tuple(parent.shape) != tuple(candidate.shape) or tuple(parent.shape) != tuple(gt.shape):
            raise RuntimeError(f"shape mismatch for {name}")
        frames.append(
            {
                "name": name,
                "parent": parent,
                "candidate": candidate,
                "gt": gt,
                "parent_metrics": _metrics(parent, gt, lpips_model),
            }
        )
    return frames


def _calibrate(args: argparse.Namespace, parent_dir: Path, candidate_dir: Path, names: list[str], device: torch.device) -> dict[str, Any]:
    if not names:
        raise RuntimeError("No common calibration renders")
    lpips_model = None
    if args.calib_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)
    frames = _load_calib_frames(parent_dir, candidate_dir, names, device, lpips_model)
    kernels = _parse_int_grid(args.local_kernels)
    frame_distances = sorted(float(torch.mean(torch.abs(frame["candidate"] - frame["parent"])).detach().cpu().item()) for frame in frames)
    frame_thresholds = _parse_float_grid(args.frame_threshold_grid) if args.frame_threshold_grid.strip() else [0.0]
    if float(args.frame_threshold_quantile) >= 0.0:
        q = min(max(float(args.frame_threshold_quantile), 0.0), 1.0)
        index = min(max(int(round(q * (len(frame_distances) - 1))), 0), len(frame_distances) - 1)
        frame_thresholds.append(float(frame_distances[index]))
    frame_thresholds = sorted(set(frame_thresholds or [0.0]))

    rows: list[dict[str, Any]] = []
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
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
                    parent_rows = []
                    frame_gains = []
                    frame_d_psnr = []
                    frame_d_ssim = []
                    frame_d_lpips = []
                    delta_mses = []
                    mask_means = []
                    candidate_used = 0
                    for frame in frames:
                        pred, info = _blend(frame["candidate"], frame["parent"], policy)
                        pred_row = _metrics(pred, frame["gt"], lpips_model)
                        parent_row = frame["parent_metrics"]
                        pred_rows.append(pred_row)
                        parent_rows.append(parent_row)
                        frame_gains.append(
                            _score_gain(pred_row, parent_row, args.objective, args.ssim_weight, args.lpips_weight)
                        )
                        frame_d_psnr.append(float(pred_row["PSNR"] - parent_row["PSNR"]))
                        frame_d_ssim.append(float(pred_row["SSIM"] - parent_row["SSIM"]))
                        frame_d_lpips.append(float(parent_row.get("LPIPS", 0.0) - pred_row.get("LPIPS", 0.0)))
                        delta_mses.append(float(pred_row["MSE"] - parent_row["MSE"]))
                        mask_mean = float(info["mask_mean"])
                        mask_means.append(mask_mean)
                        candidate_used += int(mask_mean > 1e-8)
                    count = max(len(pred_rows), 1)
                    mean = {key: sum(row[key] for row in pred_rows) / count for key in pred_rows[0].keys()}
                    parent_mean = {key: sum(row[key] for row in parent_rows) / count for key in parent_rows[0].keys()}
                    gains = {
                        "d_mse": mean["MSE"] - parent_mean["MSE"],
                        "d_psnr": mean["PSNR"] - parent_mean["PSNR"],
                        "d_ssim": mean["SSIM"] - parent_mean["SSIM"],
                        "d_lpips": parent_mean.get("LPIPS", 0.0) - mean.get("LPIPS", 0.0),
                    }
                    score = gains["d_psnr"]
                    if args.objective == "balanced":
                        score += float(args.ssim_weight) * gains["d_ssim"]
                        if "LPIPS" in mean:
                            score += float(args.lpips_weight) * gains["d_lpips"]
                    sorted_gains = sorted(frame_gains)
                    sorted_d_psnr = sorted(frame_d_psnr)
                    sorted_d_ssim = sorted(frame_d_ssim)
                    sorted_d_lpips = sorted(frame_d_lpips)
                    sorted_delta_mses = sorted(delta_mses)
                    p05_index = min(max(int(math.floor(0.05 * (len(sorted_gains) - 1))), 0), len(sorted_gains) - 1)
                    p95_mse_index = min(max(int(math.floor(0.95 * (len(sorted_delta_mses) - 1))), 0), len(sorted_delta_mses) - 1)
                    p05_gain = float(sorted_gains[p05_index])
                    p05_d_psnr = float(sorted_d_psnr[p05_index])
                    p05_d_ssim = float(sorted_d_ssim[p05_index])
                    p05_d_lpips = float(sorted_d_lpips[p05_index])
                    p95_delta_mse = float(sorted_delta_mses[p95_mse_index])
                    mean_mask = float(sum(mask_means) / max(len(mask_means), 1))
                    pass_gate = (
                        mean_mask >= float(args.min_mask_mean)
                        and gains["d_mse"] <= float(args.max_mean_mse_increase)
                        and p95_delta_mse <= float(args.max_p95_mse_increase)
                        and gains["d_psnr"] >= float(args.min_mean_psnr_gain)
                        and gains["d_ssim"] >= float(args.min_mean_ssim_gain)
                        and gains["d_lpips"] >= float(args.min_mean_lpips_gain)
                        and p05_gain >= float(args.min_p05_score_gain)
                        and p05_d_psnr >= float(args.min_p05_psnr_gain)
                        and p05_d_ssim >= float(args.min_p05_ssim_gain)
                        and p05_d_lpips >= float(args.min_p05_lpips_gain)
                    )
                    row: dict[str, Any] = {
                        **policy,
                        "calibration_views": len(frames),
                        "candidate_used_views": int(candidate_used),
                        "score": float(score),
                        "p05_frame_score_gain": p05_gain,
                        "p05_d_psnr": p05_d_psnr,
                        "p05_d_ssim": p05_d_ssim,
                        "p05_d_lpips": p05_d_lpips,
                        "p95_delta_mse": p95_delta_mse,
                        "mean_mask": mean_mask,
                        "pass_gate": bool(pass_gate),
                        **mean,
                        **{f"parent_{key}": value for key, value in parent_mean.items()},
                        **gains,
                    }
                    rows.append(row)
                    rank = (
                        bool(pass_gate),
                        float(score),
                        -max(float(gains["d_mse"]), 0.0),
                        -abs(mean_mask - float(args.target_mask_mean)),
                        -float(threshold),
                    )
                    if best is None or rank > best[0]:
                        best = (rank, row)
    selected = dict(best[1] if best is not None else rows[0])
    if not bool(selected.get("pass_gate", False)) and not args.allow_failed_policy:
        selected = _fallback_policy(kernels, calibration_views=len(frames))
    policy = {
        "frame_threshold": float(selected["frame_threshold"]),
        "threshold": float(selected["threshold"]),
        "softness": float(selected["softness"]),
        "max_blend": float(selected["max_blend"]),
        "dilate": int(selected["dilate"]),
        "kernels": [int(value) for value in selected["kernels"]],
    }
    return {
        "selected_policy": policy,
        "selected_row": selected,
        "rows": rows,
        "frame_distance_summary": {
            "min": min(frame_distances) if frame_distances else 0.0,
            "mean": sum(frame_distances) / max(len(frame_distances), 1),
            "max": max(frame_distances) if frame_distances else 0.0,
        },
    }


def _read_camera_index(method_dir: Path) -> dict[str, dict[str, Any]]:
    path = method_dir / "camera_index.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        if idx is None:
            continue
        try:
            out[f"{int(idx):05d}.png"] = item
        except (TypeError, ValueError):
            continue
    return out


def _camera_center(item: dict[str, Any]) -> tuple[float, float, float] | None:
    value = item.get("camera_center", item.get("position"))
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError):
        return None


def _center_distance(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs)))


def _nearest_center_distances(
    names: list[str],
    camera_index: dict[str, dict[str, Any]],
    support_centers: list[tuple[float, float, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not support_centers:
        return out
    for name in names:
        center = _camera_center(camera_index.get(name, {}))
        if center is None:
            continue
        out[name] = min(_center_distance(center, source) for source in support_centers)
    return out


def _source_names_from_manifest(path: str, fallback_names: list[str]) -> list[str]:
    manifest_path = Path(path).expanduser() if path else Path()
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        keys = payload.get("selected_frame_keys") if isinstance(payload, dict) else None
        if isinstance(keys, list):
            out = []
            for key in keys:
                text = str(key)
                if not text.endswith(".png"):
                    text = f"{text}.png"
                out.append(text)
            if out:
                return sorted(set(out))
        indices = payload.get("selected_frame_indices") if isinstance(payload, dict) else None
        if isinstance(indices, list):
            out = []
            for value in indices:
                try:
                    out.append(f"{int(value):05d}.png")
                except (TypeError, ValueError):
                    continue
            if out:
                return sorted(set(out))
    return list(fallback_names)


def _summarize_distances(values: list[float], q: float) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "p95": _quantile(values, q),
        "max": float(max(values)),
    }


def _evaluate_out_of_trajectory_gate(
    args: argparse.Namespace,
    parent_calib: Path,
    parent_target: Path,
    calib_names: list[str],
    target_names: list[str],
    application: dict[str, Any],
) -> dict[str, Any]:
    mode = str(args.oot_gate_mode)
    report: dict[str, Any] = {
        "enabled": mode != "off",
        "mode": mode,
        "pass": True,
        "fallback_reason": "",
    }
    if mode == "off":
        return report

    calib_index = _read_camera_index(parent_calib)
    target_index = _read_camera_index(parent_target)
    source_fallback = _filter_names_by_view_subset(list(calib_index), str(args.oot_source_view_subset))
    source_names = _source_names_from_manifest(str(args.oot_source_manifest), source_fallback)
    source_centers = [
        center
        for name in source_names
        if (center := _camera_center(calib_index.get(name, {}))) is not None
    ]
    if not calib_index or not target_index or not source_centers:
        report.update(
            {
                "pass": True,
                "fallback_reason": "camera_index_or_source_support_unavailable",
                "source_fit_view_count": len(source_centers),
            }
        )
        return report

    calib_dist_map = _nearest_center_distances(calib_names, calib_index, source_centers)
    target_dist_map = _nearest_center_distances(target_names, target_index, source_centers)
    calib_distances = list(calib_dist_map.values())
    target_distances = list(target_dist_map.values())
    calib_threshold = (
        _quantile(calib_distances, float(args.oot_center_quantile)) * (1.0 + float(args.oot_center_rel_margin))
        + float(args.oot_center_abs_margin)
    )
    app_frames = {str(row.get("image")): row for row in application.get("frames", []) if isinstance(row, dict)}
    target_ood = [name for name, distance in target_dist_map.items() if distance > calib_threshold]
    target_frame_fraction = float(len(target_ood) / max(len(target_dist_map), 1))
    weighted_num = 0.0
    weighted_den = 0.0
    frame_reports: list[dict[str, Any]] = []
    for name, distance in sorted(target_dist_map.items()):
        mask_mean = float(app_frames.get(name, {}).get("mask_mean", 0.0))
        is_ood = bool(distance > calib_threshold)
        weighted_den += mask_mean
        weighted_num += mask_mean if is_ood else 0.0
        frame_reports.append(
            {
                "image": name,
                "nearest_source_center_dist": float(distance),
                "mask_mean_before_oot": mask_mean,
                "oot_center_ood": is_ood,
            }
        )
    mask_weighted_fraction = float(weighted_num / max(weighted_den, 1e-12))
    target_mean_mask = float(application.get("mean_mask", 0.0))
    pass_gate = True
    reasons: list[str] = []
    if target_mean_mask >= float(args.oot_min_mask_mean_for_scene_check):
        if target_frame_fraction > float(args.oot_max_frame_fraction):
            pass_gate = False
            reasons.append("target_frame_fraction_exceeds_support")
        if mask_weighted_fraction > float(args.oot_max_mask_weighted_fraction):
            pass_gate = False
            reasons.append("mask_weighted_fraction_exceeds_support")

    report.update(
        {
            "pass": bool(pass_gate),
            "fallback_reason": ",".join(reasons),
            "source_manifest": str(args.oot_source_manifest),
            "source_fit_view_count": len(source_centers),
            "calib_trajectory_summary": _summarize_distances(calib_distances, float(args.oot_center_quantile)),
            "target_trajectory_summary": _summarize_distances(target_distances, float(args.oot_center_quantile)),
            "thresholds": {
                "center_dist": float(calib_threshold),
                "center_quantile": float(args.oot_center_quantile),
                "center_rel_margin": float(args.oot_center_rel_margin),
                "center_abs_margin": float(args.oot_center_abs_margin),
                "max_frame_fraction": float(args.oot_max_frame_fraction),
                "max_mask_weighted_fraction": float(args.oot_max_mask_weighted_fraction),
                "min_mask_mean_for_scene_check": float(args.oot_min_mask_mean_for_scene_check),
            },
            "target_frame_fraction": target_frame_fraction,
            "mask_weighted_ood_fraction": mask_weighted_fraction,
            "frames": frame_reports,
        }
    )
    return report


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _parent_fallback_info(candidate: torch.Tensor, parent: torch.Tensor, kernels: list[int]) -> dict[str, float]:
    score = _candidate_parent_score(candidate, parent, kernels)
    frame_distance = float(torch.mean(torch.abs(candidate - parent)).detach().cpu().item())
    flat_score = score.reshape(-1).detach().float()
    return {
        "frame_distance": frame_distance,
        "mask_mean": 0.0,
        "mask_p95": 0.0,
        "score_mean": float(flat_score.mean().cpu().item()),
        "score_p95": float(torch.quantile(flat_score, 0.95).cpu().item()),
    }


def _apply(
    args: argparse.Namespace,
    parent_dir: Path,
    candidate_dir: Path,
    out_method: Path,
    policy: dict[str, Any],
    device: torch.device,
    disabled_names: set[str] | None = None,
    disabled_reason: str = "",
) -> dict[str, Any]:
    names = _common_names(parent_dir, candidate_dir)
    if not names:
        raise RuntimeError("No common target renders")
    disabled_names = set(disabled_names or set())
    out_renders = out_method / "renders"
    out_gt = out_method / "gt"
    out_renders.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for name in names:
        parent = _read_rgb(parent_dir / "renders" / name, device)
        candidate = _read_rgb(candidate_dir / "renders" / name, device)
        if name in disabled_names:
            pred = parent
            info = _parent_fallback_info(candidate, parent, [int(value) for value in policy["kernels"]])
            info["frame_fallback"] = True
            info["frame_fallback_reason"] = disabled_reason
        else:
            pred, info = _blend(candidate, parent, policy)
            info["frame_fallback"] = False
            info["frame_fallback_reason"] = ""
        _save_rgb(pred, out_renders / name)
        shutil.copy2(parent_dir / "gt" / name, out_gt / name)
        frames.append({"image": name, **info})
    mean_mask = float(sum(float(row["mask_mean"]) for row in frames) / max(len(frames), 1))
    mean_distance = float(sum(float(row["frame_distance"]) for row in frames) / max(len(frames), 1))
    return {
        "target_views": len(names),
        "mean_mask": mean_mask,
        "mean_candidate_parent_distance": mean_distance,
        "frame_fallback_count": int(sum(1 for row in frames if bool(row.get("frame_fallback", False)))),
        "frame_fallback_reason": disabled_reason if disabled_names else "",
        "frames": frames,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["calibration"]["selected_row"]
    application = report["application"]
    oot = report.get("out_of_trajectory_gate", {})
    lines = [
        "# v109 Render-Realized Parent Gate Report",
        "",
        f"- parent_method: `{report['parent_method_name']}`",
        f"- candidate_method: `{report['candidate_method_name']}`",
        f"- method_name: `{report['method_name']}`",
        f"- calib_split: `{report['calib_split']}`",
        f"- target_split: `{report['target_split']}`",
        f"- calib_view_subset: `{report['calib_view_subset']}`",
        f"- calib_candidate_count: `{int(report['calib_candidate_count'])}`",
        f"- calib_selected_count: `{int(report['calib_selected_count'])}`",
        f"- no_target_gt_used_for_policy: `{report['no_target_gt_used_for_policy']}`",
        f"- selected_policy: `{json.dumps(report['calibration']['selected_policy'], sort_keys=True)}`",
        f"- fallback_to_parent: `{bool(selected.get('fallback_to_parent', False))}`",
        f"- calib_score: `{float(selected.get('score', 0.0)):.8f}`",
        f"- calib_mean_mask: `{float(selected.get('mean_mask', 0.0)):.8f}`",
        f"- target_mean_mask: `{float(application['mean_mask']):.8f}`",
        f"- target_views: `{int(application['target_views'])}`",
        f"- oot_gate_mode: `{oot.get('mode', 'off')}`",
        f"- oot_gate_pass: `{oot.get('pass', True)}`",
        f"- oot_fallback_reason: `{oot.get('fallback_reason', '')}`",
        "",
        "## Selected Calibration Row",
        "",
        "| dMSE | dPSNR | dSSIM | dLPIPS | p05 score gain | p05 dPSNR | p05 dSSIM | p05 dLPIPS | p95 delta MSE | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {d_mse:.8e} | {d_psnr:.8f} | {d_ssim:.8f} | {d_lpips:.8f} | {p05:.8f} | {p05_psnr:.8f} | {p05_ssim:.8f} | {p05_lpips:.8f} | {p95:.8e} | {passed} |".format(
            d_mse=float(selected.get("d_mse", 0.0)),
            d_psnr=float(selected.get("d_psnr", 0.0)),
            d_ssim=float(selected.get("d_ssim", 0.0)),
            d_lpips=float(selected.get("d_lpips", 0.0)),
            p05=float(selected.get("p05_frame_score_gain", 0.0)),
            p05_psnr=float(selected.get("p05_d_psnr", 0.0)),
            p05_ssim=float(selected.get("p05_d_ssim", 0.0)),
            p05_lpips=float(selected.get("p05_d_lpips", 0.0)),
            p95=float(selected.get("p95_delta_mse", 0.0)),
            passed="yes" if selected.get("pass_gate") else "no",
        ),
    ]
    if oot.get("enabled"):
        target_summary = oot.get("target_trajectory_summary", {})
        calib_summary = oot.get("calib_trajectory_summary", {})
        thresholds = oot.get("thresholds", {})
        lines.extend(
            [
                "",
                "## Out-of-Trajectory Gate",
                "",
                f"- source_fit_view_count: `{int(oot.get('source_fit_view_count', 0))}`",
                f"- calib_p95_center_dist: `{float(calib_summary.get('p95', 0.0)):.8f}`",
                f"- target_p95_center_dist: `{float(target_summary.get('p95', 0.0)):.8f}`",
                f"- center_dist_threshold: `{float(thresholds.get('center_dist', 0.0)):.8f}`",
                f"- target_frame_fraction: `{float(oot.get('target_frame_fraction', 0.0)):.8f}`",
                f"- mask_weighted_ood_fraction: `{float(oot.get('mask_weighted_ood_fraction', 0.0)):.8f}`",
                f"- applied_scene_fallback: `{bool(oot.get('applied_scene_fallback', False))}`",
                f"- applied_frame_fallback: `{bool(oot.get('applied_frame_fallback', False))}`",
                f"- frame_fallback_count: `{int(oot.get('frame_fallback_count', 0))}`",
                f"- mask_weighted_fraction_after_frame_fallback: `{float(oot.get('mask_weighted_fraction_after_frame_fallback', 0.0)):.8f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_wandb(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[v109ParentGate] W&B unavailable: {exc}")
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
        "v109_parent_gate/no_target_gt_used_for_policy": 1.0,
        "v109_parent_gate/calib_candidate_count": int(report["calib_candidate_count"]),
        "v109_parent_gate/calib_selected_count": int(report["calib_selected_count"]),
        "v109_parent_gate/calib_score": float(selected.get("score", 0.0)),
        "v109_parent_gate/calib_d_psnr": float(selected.get("d_psnr", 0.0)),
        "v109_parent_gate/calib_mean_mask": float(selected.get("mean_mask", 0.0)),
        "v109_parent_gate/target_mean_mask": float(report["application"]["mean_mask"]),
        "v109_parent_gate/target_views": int(report["application"]["target_views"]),
        "v109_parent_gate/fallback_to_parent": float(bool(selected.get("fallback_to_parent", False))),
    }
    oot = report.get("out_of_trajectory_gate", {})
    if oot.get("enabled"):
        flat.update(
            {
                "v109_parent_gate/oot_pass": float(bool(oot.get("pass", True))),
                "v109_parent_gate/oot_target_frame_fraction": float(oot.get("target_frame_fraction", 0.0)),
                "v109_parent_gate/oot_mask_weighted_fraction": float(oot.get("mask_weighted_ood_fraction", 0.0)),
            }
        )
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_model_path", required=True)
    parser.add_argument("--parent_method_name", required=True)
    parser.add_argument("--candidate_model_path", required=True)
    parser.add_argument("--candidate_method_name", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--target_split", default="test")
    parser.add_argument(
        "--calib_view_subset",
        choices=("all", "even", "odd"),
        default="all",
        help="Filter calibration common PNGs by numeric filename stem parity before calib_max_views sampling.",
    )
    parser.add_argument("--threshold_grid", default="0.0005,0.001,0.002,0.004,0.006,0.008,0.010,0.014,0.020")
    parser.add_argument("--frame_threshold_grid", default="0")
    parser.add_argument("--frame_threshold_quantile", default=-1.0, type=float)
    parser.add_argument("--softness_grid", default="0,0.0005,0.001,0.002")
    parser.add_argument("--max_blend_grid", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--local_kernels", default="1,9,25")
    parser.add_argument("--mask_dilate", type=int, default=0)
    parser.add_argument("--calib_max_views", type=int, default=64)
    parser.add_argument("--calib_sampler", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--min_mask_mean", type=float, default=0.0)
    parser.add_argument("--target_mask_mean", type=float, default=0.25)
    parser.add_argument("--max_mean_mse_increase", type=float, default=0.0)
    parser.add_argument("--max_p95_mse_increase", type=float, default=0.0)
    parser.add_argument("--min_mean_psnr_gain", type=float, default=0.0)
    parser.add_argument("--min_mean_ssim_gain", type=float, default=-1e-6)
    parser.add_argument("--min_mean_lpips_gain", type=float, default=-1e9)
    parser.add_argument("--min_p05_score_gain", type=float, default=-1e-4)
    parser.add_argument(
        "--min_p05_psnr_gain",
        type=float,
        default=-1e9,
        help="Require the 5th-percentile per-frame PSNR gain on calibration views to exceed this value.",
    )
    parser.add_argument(
        "--min_p05_ssim_gain",
        type=float,
        default=-1e9,
        help="Require the 5th-percentile per-frame SSIM gain on calibration views to exceed this value.",
    )
    parser.add_argument(
        "--min_p05_lpips_gain",
        type=float,
        default=-1e9,
        help="Require the 5th-percentile per-frame LPIPS improvement on calibration views to exceed this value.",
    )
    parser.add_argument("--allow_failed_policy", action="store_true")
    parser.add_argument("--oot_gate_mode", choices=("off", "report", "scene_fallback", "frame_fallback"), default="off")
    parser.add_argument("--oot_source_manifest", default="")
    parser.add_argument(
        "--oot_source_view_subset",
        choices=("all", "even", "odd"),
        default="even",
        help="Fallback source-support subset used when oot_source_manifest is absent.",
    )
    parser.add_argument("--oot_center_quantile", type=float, default=0.95)
    parser.add_argument("--oot_center_rel_margin", type=float, default=0.0)
    parser.add_argument("--oot_center_abs_margin", type=float, default=0.0)
    parser.add_argument("--oot_max_frame_fraction", type=float, default=0.10)
    parser.add_argument("--oot_max_mask_weighted_fraction", type=float, default=0.05)
    parser.add_argument("--oot_min_mask_mean_for_scene_check", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    parent_root = Path(args.parent_model_path)
    candidate_root = Path(args.candidate_model_path)
    parent_calib = _method_dir(parent_root, args.calib_split, args.parent_method_name)
    candidate_calib = _method_dir(candidate_root, args.calib_split, args.candidate_method_name)
    parent_target = _method_dir(parent_root, args.target_split, args.parent_method_name)
    candidate_target = _method_dir(candidate_root, args.target_split, args.candidate_method_name)
    calib_candidate_names = _common_names(parent_calib, candidate_calib)
    calib_subset_names = _filter_names_by_view_subset(calib_candidate_names, args.calib_view_subset)
    calib_names = _sample_names(calib_subset_names, args.calib_max_views, args.calib_sampler)
    calibration = _calibrate(args, parent_calib, candidate_calib, calib_names, device)
    out_method = Path(args.output_model_path) / args.target_split / args.method_name
    application = _apply(args, parent_target, candidate_target, out_method, calibration["selected_policy"], device)
    target_names = [str(row.get("image")) for row in application.get("frames", []) if isinstance(row, dict)]
    oot_report = _evaluate_out_of_trajectory_gate(
        args,
        parent_calib,
        parent_target,
        calib_names,
        target_names,
        application,
    )
    if (
        str(args.oot_gate_mode) == "scene_fallback"
        and not bool(oot_report.get("pass", True))
        and float(application.get("mean_mask", 0.0)) > 0.0
    ):
        before_oot_application = application
        fallback = _fallback_policy(calibration["selected_policy"]["kernels"], calibration_views=len(calib_names))
        fallback["fallback_to_parent"] = True
        fallback["fallback_reason"] = f"out_of_trajectory:{oot_report.get('fallback_reason', '')}"
        calibration["selected_policy"] = {
            "frame_threshold": float(fallback["frame_threshold"]),
            "threshold": float(fallback["threshold"]),
            "softness": float(fallback["softness"]),
            "max_blend": float(fallback["max_blend"]),
            "dilate": int(fallback["dilate"]),
            "kernels": [int(value) for value in fallback["kernels"]],
        }
        calibration["selected_row"] = fallback
        application = _apply(args, parent_target, candidate_target, out_method, calibration["selected_policy"], device)
        oot_report["applied_scene_fallback"] = True
        oot_report["application_before_oot"] = before_oot_application
        oot_report["applied_frame_fallback"] = False
    elif (
        str(args.oot_gate_mode) == "frame_fallback"
        and not bool(oot_report.get("pass", True))
        and float(application.get("mean_mask", 0.0)) > 0.0
    ):
        before_oot_application = application
        disabled_names = {
            str(row.get("image"))
            for row in oot_report.get("frames", [])
            if isinstance(row, dict) and bool(row.get("oot_center_ood", False))
        }
        disabled_names.discard("")
        if disabled_names:
            application = _apply(
                args,
                parent_target,
                candidate_target,
                out_method,
                calibration["selected_policy"],
                device,
                disabled_names=disabled_names,
                disabled_reason=f"out_of_trajectory:{oot_report.get('fallback_reason', '')}",
            )
        oot_after_frame_fallback = _evaluate_out_of_trajectory_gate(
            args,
            parent_calib,
            parent_target,
            calib_names,
            target_names,
            application,
        )
        oot_report["applied_scene_fallback"] = False
        oot_report["applied_frame_fallback"] = bool(disabled_names)
        oot_report["frame_fallback_count"] = int(len(disabled_names))
        oot_report["frame_fallback_images"] = sorted(disabled_names)
        oot_report["mask_weighted_fraction_after_frame_fallback"] = float(
            oot_after_frame_fallback.get("mask_weighted_ood_fraction", 0.0)
        )
        oot_report["application_before_oot"] = before_oot_application
    else:
        oot_report["applied_scene_fallback"] = False
        oot_report["applied_frame_fallback"] = False
    report = {
        "method": "v109 Render-Realized Parent-Preserving Gate",
        "schema_version": 1,
        "parent_model_path": args.parent_model_path,
        "parent_method_name": args.parent_method_name,
        "candidate_model_path": args.candidate_model_path,
        "candidate_method_name": args.candidate_method_name,
        "output_model_path": args.output_model_path,
        "method_name": args.method_name,
        "calib_split": args.calib_split,
        "target_split": args.target_split,
        "calib_view_subset": args.calib_view_subset,
        "calib_candidate_count": len(calib_candidate_names),
        "calib_subset_count": len(calib_subset_names),
        "calib_selected_count": len(calib_names),
        "no_target_gt_used_for_policy": True,
        "test_gt_usage": "copied for evaluator only after fixed policy application",
        "calibration": calibration,
        "application": application,
        "out_of_trajectory_gate": oot_report,
    }
    out_method.mkdir(parents=True, exist_ok=True)
    report_path = out_method / "v109_render_realized_parent_gate_report.json"
    md_path = out_method / "v109_render_realized_parent_gate_report.md"
    rows_path = out_method / "v109_render_realized_parent_gate_calibration_rows.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows_csv(rows_path, calibration["rows"])
    _write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "markdown": str(md_path),
                "rows": str(rows_path),
                "selected_policy": calibration["selected_policy"],
                "calib_view_subset": args.calib_view_subset,
                "calib_candidate_count": len(calib_candidate_names),
                "calib_selected_count": len(calib_names),
                "target_views": application["target_views"],
                "target_mean_mask": application["mean_mask"],
                "no_target_gt_used_for_policy": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
