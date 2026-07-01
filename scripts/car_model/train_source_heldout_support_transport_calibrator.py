#!/usr/bin/env python3
"""Train a source-heldout support-transport calibrator.

This is the first method step after the v298 diagnostic.  v298 showed that the
online Phase-J/ELA support-view residual signal has source-heldout headroom,
while the baked face/UV carrier loses too much target-conditioned information.

This script keeps the ELA information path and trains a small image-space
calibrator on train-only source-heldout views.  The learned module sees only
target-view render/evidence features and support-warp statistics; GT is used
only as train-heldout supervision and validation.  Target/test GT is not read.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_lumigraph_adapter import (  # noqa: E402
    FrameLoader,
    compute_evidence_signal,
    load_split_frames,
    mse_to_psnr,
    save_image_tensor,
)
from utils.loss_utils import ssim as torch_ssim  # noqa: E402


FEATURE_NAMES = [
    "signal_r",
    "signal_g",
    "signal_b",
    "abs_signal_r",
    "abs_signal_g",
    "abs_signal_b",
    "base_r",
    "base_g",
    "base_b",
    "log_confidence",
    "valid",
    "support_count_norm",
    "residual_std",
    "base_edge",
    "x_coord",
    "y_coord",
]


@dataclass
class TransportExample:
    name: str
    base: torch.Tensor
    gt: torch.Tensor
    signal: torch.Tensor
    valid: torch.Tensor
    features: torch.Tensor
    support_names: list[str]
    covered_fraction: float

    @property
    def true_residual(self) -> torch.Tensor:
        return self.gt - self.base


class SupportTransportCalibrator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 32,
        layers: int = 3,
        max_gain: float = 0.75,
        direct_scale: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        blocks: list[nn.Module] = []
        prev = int(in_channels)
        for _ in range(max(1, int(layers))):
            blocks.append(nn.Conv2d(prev, hidden_channels, kernel_size=3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            prev = hidden_channels
        self.body = nn.Sequential(*blocks)
        self.gain_head = nn.Conv2d(prev, 3, kernel_size=1)
        self.gate_head = nn.Conv2d(prev, 1, kernel_size=1)
        self.direct_head = nn.Conv2d(prev, 3, kernel_size=1) if float(direct_scale) > 0.0 else None
        self.max_gain = float(max_gain)
        self.direct_scale = float(direct_scale)

    def forward(self, features: torch.Tensor, signal: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = self.body(features)
        gain = self.max_gain * torch.sigmoid(self.gain_head(hidden))
        gate = torch.sigmoid(self.gate_head(hidden))
        delta = signal * gain * gate
        if self.direct_head is not None:
            delta = delta + self.direct_scale * torch.tanh(self.direct_head(hidden)) * gate
        return delta * valid.to(dtype=delta.dtype)


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if math.isfinite(value):
            values.append(value)
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _split_source_heldout(frames: list[Any], stride: int, offset: int) -> tuple[list[Any], list[Any]]:
    stride = max(int(stride), 2)
    offset = int(offset) % stride
    source, heldout = [], []
    for idx, frame in enumerate(frames):
        if idx % stride == offset:
            heldout.append(frame)
        else:
            source.append(frame)
    if not source or not heldout:
        raise ValueError(
            f"invalid source-heldout split: source={len(source)} heldout={len(heldout)} "
            f"stride={stride} offset={offset}"
        )
    return source, heldout


def _split_calibrator_train_val(
    frames: list[Any],
    *,
    val_stride: int,
    val_offset: int,
) -> tuple[list[Any], list[Any]]:
    if len(frames) < 2:
        raise ValueError("at least two heldout frames are required for calibrator train/val")
    val_stride = max(int(val_stride), 2)
    val_offset = int(val_offset) % val_stride
    train, val = [], []
    for idx, frame in enumerate(frames):
        if idx % val_stride == val_offset:
            val.append(frame)
        else:
            train.append(frame)
    if not train:
        train.append(val.pop())
    if not val:
        val.append(train[-1])
    return train, val


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tail(values: list[float], fraction: float = 0.20) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "cvar": 0.0}
    arr = sorted(float(v) for v in values)
    count = max(1, int(math.ceil(float(fraction) * len(arr))))
    return {"min": float(arr[0]), "cvar": float(sum(arr[:count]) / count)}


def _downscale_for_metric(image: torch.Tensor, max_side: int) -> torch.Tensor:
    max_side = int(max_side)
    if max_side <= 0:
        return image
    h, w = int(image.shape[-2]), int(image.shape[-1])
    current = max(h, w)
    if current <= max_side:
        return image
    scale = float(max_side) / float(current)
    out_h = max(8, int(round(h * scale)))
    out_w = max(8, int(round(w * scale)))
    return F.interpolate(image.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)


def _ssim_value(a: torch.Tensor, b: torch.Tensor, max_side: int) -> float:
    a_small = _downscale_for_metric(a, max_side).unsqueeze(0)
    b_small = _downscale_for_metric(b, max_side).unsqueeze(0)
    return float(torch_ssim(a_small, b_small).detach().cpu().item())


def _edge_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = torch.mean(image, dim=0, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, 1:] = gray[:, :, 1:] - gray[:, :, :-1]
    gy[:, 1:, :] = gray[:, 1:, :] - gray[:, :-1, :]
    return torch.sqrt(gx * gx + gy * gy + 1.0e-12)


def _edge_magnitude_nchw(image: torch.Tensor) -> torch.Tensor:
    gray = torch.mean(image, dim=1, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, :, 1:] = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy[:, :, 1:, :] = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    return torch.sqrt(gx * gx + gy * gy + 1.0e-12)


def _coordinate_features(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)


def _build_features(ev: Any, *, k: int) -> torch.Tensor:
    base = ev.base.to(dtype=torch.float32)
    signal = ev.signal.to(dtype=torch.float32)
    confidence = ev.confidence.to(dtype=torch.float32)
    valid = ev.valid.to(dtype=torch.float32)
    support_count = ev.support_count.to(dtype=torch.float32) if ev.support_count is not None else valid
    residual_std = ev.residual_std.to(dtype=torch.float32) if ev.residual_std is not None else torch.zeros_like(valid)
    height, width = int(base.shape[-2]), int(base.shape[-1])
    coords = _coordinate_features(height, width, base.device)
    features = torch.cat(
        [
            signal,
            torch.abs(signal),
            base,
            torch.log1p(torch.clamp(confidence, min=0.0)),
            valid,
            torch.clamp(support_count / max(float(k), 1.0), 0.0, 1.0),
            residual_std,
            _edge_magnitude(base),
            coords,
        ],
        dim=0,
    )
    if features.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(f"feature dim mismatch: got {features.shape[0]}, expected {len(FEATURE_NAMES)}")
    return features


def _compute_feature_stats(examples: list[TransportExample]) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot compute feature stats without examples")
    channels = int(examples[0].features.shape[0])
    total = 0
    sum_ch = torch.zeros(channels, dtype=torch.float64)
    sumsq_ch = torch.zeros(channels, dtype=torch.float64)
    for ex in examples:
        flat = ex.features.reshape(channels, -1).to(dtype=torch.float64)
        sum_ch += flat.sum(dim=1).cpu()
        sumsq_ch += torch.square(flat).sum(dim=1).cpu()
        total += int(flat.shape[1])
    mean = sum_ch / max(total, 1)
    var = torch.clamp(sumsq_ch / max(total, 1) - mean * mean, min=1.0e-8)
    std = torch.sqrt(var)
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)


def _normalize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=features.device, dtype=features.dtype).view(1, -1, 1, 1)
    std = std.to(device=features.device, dtype=features.dtype).view(1, -1, 1, 1)
    return (features - mean) / torch.clamp(std, min=1.0e-6)


def _random_crop(
    ex: TransportExample,
    *,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = int(ex.base.shape[-2]), int(ex.base.shape[-1])
    crop = int(crop_size)
    if crop <= 0 or crop >= min(height, width):
        y0, x0, y1, x1 = 0, 0, height, width
    else:
        y0 = random.randint(0, height - crop)
        x0 = random.randint(0, width - crop)
        y1, x1 = y0 + crop, x0 + crop
    features = ex.features[:, y0:y1, x0:x1].unsqueeze(0).to(device=device, dtype=torch.float32)
    signal = ex.signal[:, y0:y1, x0:x1].unsqueeze(0).to(device=device, dtype=torch.float32)
    valid = ex.valid[:, y0:y1, x0:x1].unsqueeze(0).to(device=device, dtype=torch.float32)
    base = ex.base[:, y0:y1, x0:x1].unsqueeze(0).to(device=device, dtype=torch.float32)
    gt = ex.gt[:, y0:y1, x0:x1].unsqueeze(0).to(device=device, dtype=torch.float32)
    return features, signal, valid, base, gt


def _residual_direction_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.expand_as(pred).to(dtype=pred.dtype)
    pred_vec = (pred * weight).reshape(pred.shape[0], -1)
    target_vec = (target * weight).reshape(target.shape[0], -1)
    denom = torch.clamp(torch.linalg.vector_norm(pred_vec, dim=1) * torch.linalg.vector_norm(target_vec, dim=1), min=1.0e-8)
    cosine = torch.sum(pred_vec * target_vec, dim=1) / denom
    return torch.mean(1.0 - cosine)


def _changed_fraction(delta: torch.Tensor) -> float:
    changed = torch.any(torch.abs(delta) > (0.5 / 255.0), dim=0)
    return float(torch.mean(changed.to(torch.float32)).detach().cpu().item())


def _image_metrics(
    base: torch.Tensor,
    gt: torch.Tensor,
    delta: torch.Tensor,
    *,
    compute_ssim: bool,
    ssim_max_side: int,
) -> dict[str, float]:
    candidate = torch.clamp(base + delta, 0.0, 1.0)
    base_mse = float(torch.mean(torch.square(base - gt)).detach().cpu().item())
    candidate_mse = float(torch.mean(torch.square(candidate - gt)).detach().cpu().item())
    row = {
        "base_mse": base_mse,
        "candidate_mse": candidate_mse,
        "mse_reduction": float(base_mse - candidate_mse),
        "base_psnr": mse_to_psnr(base_mse),
        "candidate_psnr": mse_to_psnr(candidate_mse),
        "psnr_gain": float(mse_to_psnr(candidate_mse) - mse_to_psnr(base_mse)),
        "changed_fraction": _changed_fraction(delta),
        "mean_abs_delta": float(torch.mean(torch.abs(delta)).detach().cpu().item()),
    }
    if compute_ssim:
        base_ssim = _ssim_value(base, gt, int(ssim_max_side))
        candidate_ssim = _ssim_value(candidate, gt, int(ssim_max_side))
        row.update({"base_ssim": base_ssim, "candidate_ssim": candidate_ssim, "ssim_gain": float(candidate_ssim - base_ssim)})
    return row


def _summarize_rows(rows: list[dict[str, float]], *, compute_ssim: bool) -> dict[str, float | dict[str, float]]:
    psnr_gain = [float(r["psnr_gain"]) for r in rows]
    summary: dict[str, float | dict[str, float]] = {
        "base_psnr": _mean([float(r["base_psnr"]) for r in rows]),
        "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
        "psnr_gain": _mean(psnr_gain),
        "psnr_gain_tail": _tail(psnr_gain),
        "positive_view_fraction": _mean([1.0 if v > 0.0 else 0.0 for v in psnr_gain]),
        "mean_changed_fraction": _mean([float(r["changed_fraction"]) for r in rows]),
        "mean_abs_delta": _mean([float(r["mean_abs_delta"]) for r in rows]),
    }
    if compute_ssim:
        ssim_gain = [float(r["ssim_gain"]) for r in rows]
        summary.update(
            {
                "base_ssim": _mean([float(r["base_ssim"]) for r in rows]),
                "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
                "ssim_gain": _mean(ssim_gain),
                "ssim_gain_tail": _tail(ssim_gain),
                "ssim_positive_view_fraction": _mean([1.0 if v > 0.0 else 0.0 for v in ssim_gain]),
            }
        )
    return summary


def _selection_objective(row: dict[str, Any], *, compute_ssim: bool) -> float:
    return float(row.get("psnr_gain", 0.0)) + (20.0 * float(row.get("ssim_gain", 0.0)) if compute_ssim else 0.0)


def _candidate_passes_fixed(
    row: dict[str, Any],
    fixed: dict[str, Any],
    *,
    compute_ssim: bool,
    min_changed_fraction: float,
    min_psnr_delta: float,
    min_ssim_delta: float,
) -> bool:
    if float(row.get("psnr_gain", 0.0)) <= 0.0:
        return False
    if float(row.get("mean_changed_fraction", 0.0)) < float(min_changed_fraction):
        return False
    if float(row.get("psnr_gain", 0.0)) - float(fixed.get("psnr_gain", 0.0)) < float(min_psnr_delta):
        return False
    if compute_ssim:
        if float(row.get("ssim_gain", 0.0)) <= 0.0:
            return False
        if float(row.get("ssim_gain", 0.0)) - float(fixed.get("ssim_gain", 0.0)) < float(min_ssim_delta):
            return False
    return True


def _select_vs_fixed(
    summaries: list[dict[str, Any]],
    fixed: dict[str, Any],
    *,
    compute_ssim: bool,
    min_changed_fraction: float,
    min_psnr_delta: float,
    min_ssim_delta: float,
) -> tuple[dict[str, Any], bool]:
    passing = [
        row
        for row in summaries
        if _candidate_passes_fixed(
            row,
            fixed,
            compute_ssim=compute_ssim,
            min_changed_fraction=min_changed_fraction,
            min_psnr_delta=min_psnr_delta,
            min_ssim_delta=min_ssim_delta,
        )
    ]
    pool = passing if passing else summaries
    return max(pool, key=lambda r: _selection_objective(r, compute_ssim=compute_ssim)), bool(passing)


def _evaluate_fixed_alpha(
    examples: list[TransportExample],
    alpha_grid: list[float],
    *,
    compute_ssim: bool,
    ssim_max_side: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha_summaries = []
    per_alpha: dict[float, list[dict[str, float]]] = {}
    for alpha in alpha_grid:
        rows = []
        for ex in examples:
            delta = float(alpha) * ex.signal
            row = _image_metrics(ex.base, ex.gt, delta, compute_ssim=compute_ssim, ssim_max_side=ssim_max_side)
            row["view"] = ex.name
            row["alpha"] = float(alpha)
            rows.append(row)
        per_alpha[float(alpha)] = rows
        summary = {"alpha": float(alpha), **_summarize_rows(rows, compute_ssim=compute_ssim)}
        alpha_summaries.append(summary)
    best = max(
        alpha_summaries,
        key=lambda r: _selection_objective(r, compute_ssim=compute_ssim),
    )
    return alpha_summaries, {"summary": best, "rows": per_alpha[float(best["alpha"])]}


def _evaluate_learned(
    model: SupportTransportCalibrator,
    examples: list[TransportExample],
    scale_grid: list[float],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    device: torch.device,
    compute_ssim: bool,
    ssim_max_side: int,
    visual_dir: Path | None = None,
    save_example_views: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    per_scale: dict[float, list[dict[str, float]]] = {float(scale): [] for scale in scale_grid}
    saved_payloads: list[tuple[TransportExample, torch.Tensor, float]] = []
    with torch.no_grad():
        for ex in tqdm(examples, desc="evaluate learned calibrator"):
            features = ex.features.unsqueeze(0).to(device=device, dtype=torch.float32)
            signal = ex.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = ex.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_delta = model(_normalize(features, mean, std), signal, valid).squeeze(0).detach().cpu()
            for scale in scale_grid:
                delta = float(scale) * pred_delta
                row = _image_metrics(ex.base, ex.gt, delta, compute_ssim=compute_ssim, ssim_max_side=ssim_max_side)
                row["view"] = ex.name
                row["scale"] = float(scale)
                per_scale[float(scale)].append(row)
            if visual_dir is not None and len(saved_payloads) < int(save_example_views):
                saved_payloads.append((ex, pred_delta, 1.0))
    summaries = []
    for scale, rows in per_scale.items():
        summaries.append({"scale": float(scale), **_summarize_rows(rows, compute_ssim=compute_ssim)})
    best = max(
        summaries,
        key=lambda r: _selection_objective(r, compute_ssim=compute_ssim),
    )
    best_scale = float(best["scale"])
    if visual_dir is not None and int(save_example_views) > 0:
        visual_dir.mkdir(parents=True, exist_ok=True)
        for ex, pred_delta, _ in saved_payloads:
            learned = torch.clamp(ex.base + best_scale * pred_delta, 0.0, 1.0)
            save_image_tensor(ex.base, visual_dir / f"{ex.name}_base.png")
            save_image_tensor(learned, visual_dir / f"{ex.name}_learned.png")
            save_image_tensor(ex.gt, visual_dir / f"{ex.name}_gt.png")
            save_image_tensor(torch.clamp(0.5 + 2.0 * pred_delta, 0.0, 1.0), visual_dir / f"{ex.name}_learned_delta_x2.png")
            save_image_tensor(torch.clamp(0.5 + 2.0 * ex.signal, 0.0, 1.0), visual_dir / f"{ex.name}_raw_signal_x2.png")
    return summaries, {"summary": best, "rows": per_scale[best_scale], "rows_by_scale": per_scale}


def _evaluate_hybrid(
    model: SupportTransportCalibrator,
    examples: list[TransportExample],
    *,
    anchor_alpha: float,
    scale_grid: list[float],
    blend_grid: list[float],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    compute_ssim: bool,
    ssim_max_side: int,
    visual_dir: Path | None = None,
    save_example_views: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    combo_rows: dict[tuple[float, float], list[dict[str, float]]] = {}
    saved_payloads: list[tuple[TransportExample, torch.Tensor]] = []
    with torch.no_grad():
        for ex in tqdm(examples, desc="evaluate hybrid anchor calibrator"):
            features = ex.features.unsqueeze(0).to(device=device, dtype=torch.float32)
            signal = ex.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
            valid = ex.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_delta = model(_normalize(features, mean, std), signal, valid).squeeze(0).detach().cpu()
            anchor_delta = float(anchor_alpha) * ex.signal
            for scale in scale_grid:
                learned_delta = float(scale) * pred_delta
                for blend in blend_grid:
                    blend = float(blend)
                    key = (float(scale), blend)
                    delta = (1.0 - blend) * anchor_delta + blend * learned_delta
                    row = _image_metrics(ex.base, ex.gt, delta, compute_ssim=compute_ssim, ssim_max_side=ssim_max_side)
                    row["view"] = ex.name
                    row["anchor_alpha"] = float(anchor_alpha)
                    row["scale"] = float(scale)
                    row["blend"] = blend
                    combo_rows.setdefault(key, []).append(row)
            if visual_dir is not None and len(saved_payloads) < int(save_example_views):
                saved_payloads.append((ex, pred_delta))
    summaries = []
    for (scale, blend), rows in combo_rows.items():
        summaries.append(
            {
                "anchor_alpha": float(anchor_alpha),
                "scale": float(scale),
                "blend": float(blend),
                **_summarize_rows(rows, compute_ssim=compute_ssim),
            }
        )
    best = max(
        summaries,
        key=lambda r: _selection_objective(r, compute_ssim=compute_ssim),
    )
    best_key = (float(best["scale"]), float(best["blend"]))
    if visual_dir is not None and int(save_example_views) > 0:
        visual_dir.mkdir(parents=True, exist_ok=True)
        best_scale = float(best["scale"])
        best_blend = float(best["blend"])
        for ex, pred_delta in saved_payloads:
            anchor_delta = float(anchor_alpha) * ex.signal
            learned_delta = best_scale * pred_delta
            hybrid_delta = (1.0 - best_blend) * anchor_delta + best_blend * learned_delta
            save_image_tensor(torch.clamp(ex.base + hybrid_delta, 0.0, 1.0), visual_dir / f"{ex.name}_hybrid.png")
            save_image_tensor(torch.clamp(0.5 + 2.0 * hybrid_delta, 0.0, 1.0), visual_dir / f"{ex.name}_hybrid_delta_x2.png")
    return summaries, {"summary": best, "rows": combo_rows[best_key], "rows_by_key": combo_rows}


def _build_examples(
    frames: list[Any],
    source_frames: list[Any],
    *,
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> list[TransportExample]:
    examples = []
    for target in tqdm(frames, desc="build ELA transport examples"):
        with torch.no_grad():
            ev = compute_evidence_signal(
                target,
                source_frames,
                k=int(args.k),
                mode="residual",
                residual_clip=float(args.residual_clip),
                min_confidence=float(args.min_confidence),
                depth_abs_tol=float(args.depth_abs_tol),
                depth_rel_tol=float(args.depth_rel_tol),
                direction_weight=float(args.direction_weight),
                evidence_max_side=int(args.evidence_max_side),
                loader=loader,
                device=device,
            )
            gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
            if ev.base.shape != gt.shape or ev.signal.shape != gt.shape or ev.valid.shape[-2:] != gt.shape[-2:]:
                raise RuntimeError(
                    f"shape mismatch for {target.name}: "
                    f"base={tuple(ev.base.shape)} signal={tuple(ev.signal.shape)} "
                    f"valid={tuple(ev.valid.shape)} gt={tuple(gt.shape)}"
                )
            features = _build_features(ev, k=int(args.k))
        examples.append(
            TransportExample(
                name=target.name,
                base=ev.base.detach().cpu(),
                gt=gt.detach().cpu(),
                signal=ev.signal.detach().cpu(),
                valid=ev.valid.detach().cpu().to(dtype=torch.float32),
                features=features.detach().cpu(),
                support_names=list(ev.support_names),
                covered_fraction=float(ev.valid.to(torch.float32).mean().detach().cpu().item()),
            )
        )
    return examples


def _train_model(
    train_examples: list[TransportExample],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[SupportTransportCalibrator, list[dict[str, float]]]:
    model = SupportTransportCalibrator(
        len(FEATURE_NAMES),
        hidden_channels=int(args.hidden_channels),
        layers=int(args.layers),
        max_gain=float(args.max_gain),
        direct_scale=float(args.direct_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    steps = max(1, int(args.train_steps))
    for step in tqdm(range(1, steps + 1), desc="train support calibrator"):
        model.train()
        ex = random.choice(train_examples)
        features, signal, valid, base, gt = _random_crop(ex, crop_size=int(args.crop_size), device=device)
        pred_delta = model(_normalize(features, mean, std), signal, valid)
        target_delta = gt - base
        candidate = torch.clamp(base + pred_delta, 0.0, 1.0)
        image_loss = torch.mean(torch.square(candidate - gt))
        valid3 = valid.expand_as(pred_delta)
        denom = torch.clamp(valid3.sum(), min=1.0)
        delta_loss = torch.sum(torch.square(pred_delta - target_delta) * valid3) / denom
        direction_loss = _residual_direction_loss(pred_delta, target_delta, valid)
        magnitude_loss = torch.abs(torch.mean(torch.abs(pred_delta) * valid3) - torch.mean(torch.abs(target_delta) * valid3))
        reg_loss = torch.mean(torch.abs(pred_delta))
        if float(args.ssim_loss_weight) > 0.0:
            ssim_loss = 1.0 - torch_ssim(candidate, gt)
        else:
            ssim_loss = torch.zeros((), device=device, dtype=torch.float32)
        if float(args.edge_loss_weight) > 0.0:
            edge_loss = torch.mean(torch.abs(_edge_magnitude_nchw(candidate) - _edge_magnitude_nchw(gt)))
        else:
            edge_loss = torch.zeros((), device=device, dtype=torch.float32)
        loss = (
            image_loss
            + float(args.delta_loss_weight) * delta_loss
            + float(args.direction_loss_weight) * direction_loss
            + float(args.magnitude_loss_weight) * magnitude_loss
            + float(args.ssim_loss_weight) * ssim_loss
            + float(args.edge_loss_weight) * edge_loss
            + float(args.delta_l1_reg) * reg_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        if step == 1 or step % max(1, int(args.log_interval)) == 0 or step == steps:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu().item()),
                    "image_loss": float(image_loss.detach().cpu().item()),
                    "delta_loss": float(delta_loss.detach().cpu().item()),
                    "direction_loss": float(direction_loss.detach().cpu().item()),
                    "magnitude_loss": float(magnitude_loss.detach().cpu().item()),
                    "ssim_loss": float(ssim_loss.detach().cpu().item()),
                    "edge_loss": float(edge_loss.detach().cpu().item()),
                    "reg_loss": float(reg_loss.detach().cpu().item()),
                }
            )
    return model, history


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "support_transport_calibrator_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    lines = [
        "# Source-Heldout Support-Transport Calibrator",
        "",
        "This is a train-only source-heldout method experiment. Target/test GT is not used.",
        "",
        "## Summary",
        "",
        f"- train source views: `{payload['split']['source_views']}`",
        f"- calibrator train / val views: `{payload['split']['calibrator_train_views']} / {payload['split']['calibrator_val_views']}`",
        f"- raw fixed best alpha: `{summary['fixed_best_alpha']}`",
        f"- raw fixed val PSNR gain: `{summary['fixed_val_psnr_gain']}`",
        f"- raw fixed val SSIM gain: `{summary.get('fixed_val_ssim_gain')}`",
        f"- learned best scale: `{summary['learned_best_scale']}`",
        f"- learned val PSNR gain: `{summary['learned_val_psnr_gain']}`",
        f"- learned val SSIM gain: `{summary.get('learned_val_ssim_gain')}`",
        f"- learned vs fixed PSNR delta: `{summary['learned_minus_fixed_psnr_gain']}`",
        f"- learned vs fixed SSIM delta: `{summary.get('learned_minus_fixed_ssim_gain')}`",
        f"- selected method: `{summary.get('selected_method')}`",
        f"- selected val PSNR gain: `{summary.get('selected_val_psnr_gain')}`",
        f"- selected val SSIM gain: `{summary.get('selected_val_ssim_gain')}`",
        f"- selected vs fixed PSNR delta: `{summary.get('selected_minus_fixed_psnr_gain')}`",
        f"- selected vs fixed SSIM delta: `{summary.get('selected_minus_fixed_ssim_gain')}`",
        f"- hybrid anchor alpha / scale / blend: `{summary.get('hybrid_anchor_alpha')}` / `{summary.get('hybrid_best_scale')}` / `{summary.get('hybrid_best_blend')}`",
        f"- all-axis val pass: `{summary['all_axis_val_pass']}`",
        "",
        "## Fixed Alpha Validation Sweep",
        "",
        "| alpha | PSNR gain | changed | pos views | min PSNR gain | CVaR20 PSNR gain | SSIM gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["validation"]["fixed_alpha_summaries"]:
        tail = row.get("psnr_gain_tail", {})
        lines.append(
            "| {alpha:.6g} | {psnr:+.9f} | {chg:.9f} | {pos:.6f} | {minv:+.9f} | {cvar:+.9f} | {ssim:+.9f} |".format(
                alpha=float(row.get("alpha", 0.0)),
                psnr=float(row.get("psnr_gain", 0.0)),
                chg=float(row.get("mean_changed_fraction", 0.0)),
                pos=float(row.get("positive_view_fraction", 0.0)),
                minv=float(tail.get("min", 0.0)),
                cvar=float(tail.get("cvar", 0.0)),
                ssim=float(row.get("ssim_gain", 0.0)),
            )
        )
    lines += [
        "",
        "## Learned Scale Validation Sweep",
        "",
        "| scale | PSNR gain | changed | pos views | min PSNR gain | CVaR20 PSNR gain | SSIM gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["validation"]["learned_scale_summaries"]:
        tail = row.get("psnr_gain_tail", {})
        lines.append(
            "| {scale:.6g} | {psnr:+.9f} | {chg:.9f} | {pos:.6f} | {minv:+.9f} | {cvar:+.9f} | {ssim:+.9f} |".format(
                scale=float(row.get("scale", 0.0)),
                psnr=float(row.get("psnr_gain", 0.0)),
                chg=float(row.get("mean_changed_fraction", 0.0)),
                pos=float(row.get("positive_view_fraction", 0.0)),
                minv=float(tail.get("min", 0.0)),
                cvar=float(tail.get("cvar", 0.0)),
                ssim=float(row.get("ssim_gain", 0.0)),
            )
        )
    if payload["validation"].get("hybrid_summaries"):
        lines += [
            "",
            "## Hybrid Anchor Validation Sweep",
            "",
            "| anchor alpha | scale | blend | PSNR gain | changed | pos views | min PSNR gain | CVaR20 PSNR gain | SSIM gain |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["validation"]["hybrid_summaries"]:
            tail = row.get("psnr_gain_tail", {})
            lines.append(
                "| {alpha:.6g} | {scale:.6g} | {blend:.6g} | {psnr:+.9f} | {chg:.9f} | {pos:.6f} | {minv:+.9f} | {cvar:+.9f} | {ssim:+.9f} |".format(
                    alpha=float(row.get("anchor_alpha", 0.0)),
                    scale=float(row.get("scale", 0.0)),
                    blend=float(row.get("blend", 0.0)),
                    psnr=float(row.get("psnr_gain", 0.0)),
                    chg=float(row.get("mean_changed_fraction", 0.0)),
                    pos=float(row.get("positive_view_fraction", 0.0)),
                    minv=float(tail.get("min", 0.0)),
                    cvar=float(tail.get("cvar", 0.0)),
                    ssim=float(row.get("ssim_gain", 0.0)),
                )
            )
    lines += ["", "## Verdict", "", str(payload.get("verdict", ""))]
    (output_dir / "support_transport_calibrator_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    base_model = Path(args.base_model_path)
    base_method = str(args.base_method_name)
    alpha_grid = _parse_float_grid(args.alpha_grid)
    learned_scale_grid = _parse_float_grid(args.learned_scale_grid)

    train_frames = load_split_frames(base_model, "train", base_method)
    source_frames, heldout_frames = _split_source_heldout(train_frames, int(args.heldout_stride), int(args.heldout_offset))
    if int(args.max_heldout_views) > 0:
        heldout_frames = heldout_frames[: int(args.max_heldout_views)]
    calibrator_train_frames, calibrator_val_frames = _split_calibrator_train_val(
        heldout_frames,
        val_stride=int(args.calibrator_val_stride),
        val_offset=int(args.calibrator_val_offset),
    )

    loader = FrameLoader(device=device)
    train_examples = _build_examples(calibrator_train_frames, source_frames, loader=loader, device=device, args=args)
    val_examples = _build_examples(calibrator_val_frames, source_frames, loader=loader, device=device, args=args)
    feature_mean, feature_std = _compute_feature_stats(train_examples)
    model, history = _train_model(train_examples, feature_mean, feature_std, device=device, args=args)

    fixed_val_summaries, fixed_val_best = _evaluate_fixed_alpha(
        val_examples,
        alpha_grid,
        compute_ssim=bool(args.compute_ssim),
        ssim_max_side=int(args.ssim_max_side),
    )
    learned_val_summaries, learned_val_best = _evaluate_learned(
        model,
        val_examples,
        learned_scale_grid,
        feature_mean,
        feature_std,
        device=device,
        compute_ssim=bool(args.compute_ssim),
        ssim_max_side=int(args.ssim_max_side),
        visual_dir=visual_dir,
        save_example_views=int(args.save_example_views),
    )
    if bool(args.enable_hybrid_eval):
        anchor_alpha = (
            float(args.hybrid_anchor_alpha)
            if math.isfinite(float(args.hybrid_anchor_alpha)) and float(args.hybrid_anchor_alpha) >= 0.0
            else float(fixed_val_best["summary"].get("alpha", 0.0))
        )
        hybrid_blend_grid = _parse_float_grid(args.hybrid_blend_grid)
        hybrid_val_summaries, hybrid_val_best = _evaluate_hybrid(
            model,
            val_examples,
            anchor_alpha=anchor_alpha,
            scale_grid=learned_scale_grid,
            blend_grid=hybrid_blend_grid,
            mean=feature_mean,
            std=feature_std,
            device=device,
            compute_ssim=bool(args.compute_ssim),
            ssim_max_side=int(args.ssim_max_side),
            visual_dir=visual_dir,
            save_example_views=int(args.save_example_views),
        )
    else:
        hybrid_val_summaries = []
        hybrid_val_best = {"summary": {}}
    if bool(args.skip_train_eval):
        learned_train_summaries = []
        learned_train_best = {"summary": {}}
    else:
        learned_train_summaries, learned_train_best = _evaluate_learned(
            model,
            train_examples,
            learned_scale_grid,
            feature_mean,
            feature_std,
            device=device,
            compute_ssim=bool(args.compute_ssim),
            ssim_max_side=int(args.ssim_max_side),
        )

    fixed_summary = fixed_val_best["summary"]
    learned_summary, learned_axis_pass = _select_vs_fixed(
        learned_val_summaries,
        fixed_summary,
        compute_ssim=bool(args.compute_ssim),
        min_changed_fraction=float(args.min_changed_fraction),
        min_psnr_delta=float(args.min_learned_vs_fixed_psnr_delta),
        min_ssim_delta=float(args.min_learned_vs_fixed_ssim_delta),
    )
    learned_val_best["summary"] = learned_summary
    learned_val_best["rows"] = learned_val_best.get("rows_by_scale", {}).get(float(learned_summary.get("scale", 0.0)), learned_val_best["rows"])
    if bool(args.enable_hybrid_eval):
        hybrid_summary, hybrid_axis_pass = _select_vs_fixed(
            hybrid_val_summaries,
            fixed_summary,
            compute_ssim=bool(args.compute_ssim),
            min_changed_fraction=float(args.min_changed_fraction),
            min_psnr_delta=float(args.min_learned_vs_fixed_psnr_delta),
            min_ssim_delta=float(args.min_learned_vs_fixed_ssim_delta),
        )
        hybrid_val_best["summary"] = hybrid_summary
        hybrid_val_best["rows"] = hybrid_val_best.get("rows_by_key", {}).get(
            (float(hybrid_summary.get("scale", 0.0)), float(hybrid_summary.get("blend", 0.0))),
            hybrid_val_best["rows"],
        )
    else:
        hybrid_axis_pass = False
    selected_summary = hybrid_val_best["summary"] if bool(args.enable_hybrid_eval) else learned_summary
    selected_method = "hybrid_anchor_calibrator" if bool(args.enable_hybrid_eval) else "learned_calibrator"
    learned_minus_fixed_psnr = float(learned_summary.get("psnr_gain", 0.0)) - float(fixed_summary.get("psnr_gain", 0.0))
    learned_minus_fixed_ssim = (
        float(learned_summary.get("ssim_gain", 0.0)) - float(fixed_summary.get("ssim_gain", 0.0))
        if bool(args.compute_ssim)
        else None
    )
    selected_minus_fixed_psnr = float(selected_summary.get("psnr_gain", 0.0)) - float(fixed_summary.get("psnr_gain", 0.0))
    selected_minus_fixed_ssim = (
        float(selected_summary.get("ssim_gain", 0.0)) - float(fixed_summary.get("ssim_gain", 0.0))
        if bool(args.compute_ssim)
        else None
    )
    all_axis_val_pass = bool(hybrid_axis_pass if bool(args.enable_hybrid_eval) else learned_axis_pass)
    verdict = (
        "The learned support-transport calibrator improves over the raw fixed-alpha ELA signal on source-heldout validation."
        if all_axis_val_pass
        else "The learned calibrator is implemented and evaluated, but it is not yet better than the raw fixed-alpha ELA signal under the configured validation gate."
    )
    payload: dict[str, Any] = {
        "method": "v299 source-heldout learned support-transport calibrator",
        "real_method_change": True,
        "target_gt_usage": "train source-heldout supervision only; target/test GT is not read",
        "base_model_path": str(base_model),
        "base_method_name": base_method,
        "device": str(device),
        "feature_names": FEATURE_NAMES,
        "split": {
            "train_views": int(len(train_frames)),
            "source_views": int(len(source_frames)),
            "heldout_views": int(len(heldout_frames)),
            "calibrator_train_views": int(len(calibrator_train_frames)),
            "calibrator_val_views": int(len(calibrator_val_frames)),
            "heldout_stride": int(args.heldout_stride),
            "heldout_offset": int(args.heldout_offset),
            "calibrator_val_stride": int(args.calibrator_val_stride),
            "calibrator_val_offset": int(args.calibrator_val_offset),
            "source_names": [frame.name for frame in source_frames],
            "calibrator_train_names": [frame.name for frame in calibrator_train_frames],
            "calibrator_val_names": [frame.name for frame in calibrator_val_frames],
        },
        "config": {
            "k": int(args.k),
            "alpha_grid": [float(x) for x in alpha_grid],
            "learned_scale_grid": [float(x) for x in learned_scale_grid],
            "residual_clip": float(args.residual_clip),
            "min_confidence": float(args.min_confidence),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
            "direction_weight": float(args.direction_weight),
            "evidence_max_side": int(args.evidence_max_side),
            "train_steps": int(args.train_steps),
            "crop_size": int(args.crop_size),
            "hidden_channels": int(args.hidden_channels),
            "layers": int(args.layers),
            "max_gain": float(args.max_gain),
            "direct_scale": float(args.direct_scale),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "delta_loss_weight": float(args.delta_loss_weight),
            "direction_loss_weight": float(args.direction_loss_weight),
            "magnitude_loss_weight": float(args.magnitude_loss_weight),
            "ssim_loss_weight": float(args.ssim_loss_weight),
            "edge_loss_weight": float(args.edge_loss_weight),
            "delta_l1_reg": float(args.delta_l1_reg),
            "compute_ssim": bool(args.compute_ssim),
            "ssim_max_side": int(args.ssim_max_side),
            "seed": int(args.seed),
        },
        "train_history": history,
        "training": {
            "learned_scale_summaries": learned_train_summaries,
            "learned_best": learned_train_best["summary"],
        },
        "validation": {
            "fixed_alpha_summaries": fixed_val_summaries,
            "fixed_best": fixed_summary,
            "fixed_best_rows": fixed_val_best["rows"],
            "learned_scale_summaries": learned_val_summaries,
            "learned_best": learned_summary,
            "learned_best_rows": learned_val_best["rows"],
            "hybrid_summaries": hybrid_val_summaries,
            "hybrid_best": hybrid_val_best["summary"],
            "hybrid_best_rows": hybrid_val_best.get("rows", []),
        },
        "coverage": {
            "train_mean_covered_fraction": _mean([float(ex.covered_fraction) for ex in train_examples]),
            "val_mean_covered_fraction": _mean([float(ex.covered_fraction) for ex in val_examples]),
        },
        "summary": {
            "fixed_best_alpha": float(fixed_summary.get("alpha", 0.0)),
            "fixed_val_psnr_gain": float(fixed_summary.get("psnr_gain", 0.0)),
            "fixed_val_ssim_gain": float(fixed_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "learned_best_scale": float(learned_summary.get("scale", 0.0)),
            "learned_val_psnr_gain": float(learned_summary.get("psnr_gain", 0.0)),
            "learned_val_ssim_gain": float(learned_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "learned_minus_fixed_psnr_gain": learned_minus_fixed_psnr,
            "learned_minus_fixed_ssim_gain": learned_minus_fixed_ssim,
            "selected_method": selected_method,
            "selected_val_psnr_gain": float(selected_summary.get("psnr_gain", 0.0)),
            "selected_val_ssim_gain": float(selected_summary.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "selected_minus_fixed_psnr_gain": selected_minus_fixed_psnr,
            "selected_minus_fixed_ssim_gain": selected_minus_fixed_ssim,
            "hybrid_anchor_alpha": float(hybrid_val_best["summary"].get("anchor_alpha", -1.0))
            if bool(args.enable_hybrid_eval)
            else None,
            "hybrid_best_scale": float(hybrid_val_best["summary"].get("scale", -1.0))
            if bool(args.enable_hybrid_eval)
            else None,
            "hybrid_best_blend": float(hybrid_val_best["summary"].get("blend", -1.0))
            if bool(args.enable_hybrid_eval)
            else None,
            "hybrid_val_psnr_gain": float(hybrid_val_best["summary"].get("psnr_gain", 0.0))
            if bool(args.enable_hybrid_eval)
            else None,
            "hybrid_val_ssim_gain": float(hybrid_val_best["summary"].get("ssim_gain", 0.0))
            if bool(args.enable_hybrid_eval) and bool(args.compute_ssim)
            else None,
            "learned_val_changed_fraction": float(learned_summary.get("mean_changed_fraction", 0.0)),
            "learned_val_positive_view_fraction": float(learned_summary.get("positive_view_fraction", 0.0)),
            "selected_val_changed_fraction": float(selected_summary.get("mean_changed_fraction", 0.0)),
            "selected_val_positive_view_fraction": float(selected_summary.get("positive_view_fraction", 0.0)),
            "all_axis_val_pass": bool(all_axis_val_pass),
        },
        "verdict": verdict,
        "final_status": "METHOD_EXPERIMENT_COMPLETE_NOT_PAPER_COMPLETE",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": FEATURE_NAMES,
            "config": payload["config"],
        },
        output_dir / "support_transport_calibrator.pt",
    )
    _write_report(output_dir, payload)

    if bool(args.enable_wandb):
        import wandb

        run = wandb.init(project=str(args.wandb_project), name=str(args.wandb_run_name), dir=str(output_dir))
        run.config.update(payload["config"])
        flat = {
            "v299/fixed_val_psnr_gain": float(payload["summary"]["fixed_val_psnr_gain"]),
            "v299/learned_val_psnr_gain": float(payload["summary"]["learned_val_psnr_gain"]),
            "v299/learned_minus_fixed_psnr_gain": float(payload["summary"]["learned_minus_fixed_psnr_gain"]),
            "v299/selected_val_psnr_gain": float(payload["summary"]["selected_val_psnr_gain"]),
            "v299/selected_minus_fixed_psnr_gain": float(payload["summary"]["selected_minus_fixed_psnr_gain"]),
            "v299/learned_val_changed_fraction": float(payload["summary"]["learned_val_changed_fraction"]),
            "v299/selected_val_changed_fraction": float(payload["summary"]["selected_val_changed_fraction"]),
            "v299/all_axis_val_pass": float(bool(payload["summary"]["all_axis_val_pass"])),
        }
        if payload["summary"]["learned_val_ssim_gain"] is not None:
            flat["v299/fixed_val_ssim_gain"] = float(payload["summary"]["fixed_val_ssim_gain"])
            flat["v299/learned_val_ssim_gain"] = float(payload["summary"]["learned_val_ssim_gain"])
            flat["v299/learned_minus_fixed_ssim_gain"] = float(payload["summary"]["learned_minus_fixed_ssim_gain"])
            flat["v299/selected_val_ssim_gain"] = float(payload["summary"]["selected_val_ssim_gain"])
            flat["v299/selected_minus_fixed_ssim_gain"] = float(payload["summary"]["selected_minus_fixed_ssim_gain"])
        for row in history:
            wandb.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=int(row["step"]))
        run.log(flat)
        run.summary.update(flat)
        run.finish()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--base_method_name", default="ours_26000")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout_stride", type=int, default=4)
    parser.add_argument("--heldout_offset", type=int, default=0)
    parser.add_argument("--calibrator_val_stride", type=int, default=3)
    parser.add_argument("--calibrator_val_offset", type=int, default=0)
    parser.add_argument("--max_heldout_views", type=int, default=0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--learned_scale_grid", default="0,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--enable_hybrid_eval", action="store_true")
    parser.add_argument("--hybrid_anchor_alpha", type=float, default=float("nan"))
    parser.add_argument("--hybrid_blend_grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--depth_abs_tol", type=float, default=0.02)
    parser.add_argument("--depth_rel_tol", type=float, default=0.03)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--evidence_max_side", type=int, default=512)
    parser.add_argument("--train_steps", type=int, default=400)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max_gain", type=float, default=0.75)
    parser.add_argument("--direct_scale", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--delta_loss_weight", type=float, default=0.50)
    parser.add_argument("--direction_loss_weight", type=float, default=0.02)
    parser.add_argument("--magnitude_loss_weight", type=float, default=0.05)
    parser.add_argument("--ssim_loss_weight", type=float, default=0.0)
    parser.add_argument("--edge_loss_weight", type=float, default=0.0)
    parser.add_argument("--delta_l1_reg", type=float, default=0.005)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--compute_ssim", action="store_true")
    parser.add_argument("--ssim_max_side", type=int, default=384)
    parser.add_argument("--min_changed_fraction", type=float, default=1.0e-4)
    parser.add_argument("--min_learned_vs_fixed_psnr_delta", type=float, default=0.0)
    parser.add_argument("--min_learned_vs_fixed_ssim_delta", type=float, default=0.0)
    parser.add_argument("--save_example_views", type=int, default=0)
    parser.add_argument("--skip_train_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=299)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="v299-source-heldout-support-transport-calibrator")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "report": str(Path(args.output_dir) / "support_transport_calibrator_report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
