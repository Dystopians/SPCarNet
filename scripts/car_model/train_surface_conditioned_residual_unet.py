#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    build_lpips_model,
    evidence_views,
    image_lpips_chw,
    image_ssim_chw,
    save_image_chw,
)
from utils.loss_utils import ssim  # noqa: E402


DEFAULT_FIT_EVIDENCE = "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence"


FORBIDDEN_TARGET_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_rgb_raw",
    "teacher_residual_l1",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
    return float("inf") if mse <= 1.0e-12 else float(-10.0 * math.log10(mse))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return sorted(set(values))


def _policy_split(paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    fit, val = [], []
    for idx, path in enumerate(paths):
        if int(stride) > 1 and idx % int(stride) == 0:
            val.append(path)
        else:
            fit.append(path)
    return fit, val


def _to_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4):
        return np.moveaxis(arr, -1, 0).astype(np.float32)
    raise ValueError(f"cannot convert array to CHW: shape={arr.shape}")


def _resize_chw_tensor(x: torch.Tensor, max_side: int) -> torch.Tensor:
    if int(max_side) <= 0:
        return x
    _, h, w = x.shape
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return x
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return F.interpolate(x.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _resize_bchw_tensor(x: torch.Tensor, max_side: int) -> torch.Tensor:
    if int(max_side) <= 0:
        return x
    _, _, h, w = x.shape
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return x
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _lpips_train_loss(
    lpips_model: torch.nn.Module | None,
    pred: torch.Tensor,
    target: torch.Tensor,
    max_side: int,
) -> torch.Tensor:
    if lpips_model is None:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    pred_in = _resize_bchw_tensor(pred, int(max_side))
    target_in = _resize_bchw_tensor(target, int(max_side))
    return lpips_model(pred_in, target_in).mean()


def _luma(x: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return torch.sum(x[:, :3] * weights, dim=1, keepdim=True)


def _resize_mask_bchw(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask.float(), size=size, mode="nearest").to(dtype=mask.dtype)


def _masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.mean(torch.abs(pred - target))
    mask = _resize_mask_bchw(mask.to(device=pred.device, dtype=pred.dtype), pred.shape[-2:])
    denom = torch.clamp(mask.sum() * pred.shape[1], min=1.0)
    return torch.sum(torch.abs(pred - target) * mask) / denom


def _fill_inactive_with_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return pred
    mask = _resize_mask_bchw(mask.to(device=pred.device, dtype=pred.dtype), pred.shape[-2:])
    return pred * mask + target * (1.0 - mask)


def _masked_ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return 1.0 - ssim(pred, target)
    masked_pred = _fill_inactive_with_target(pred, target, mask)
    return 1.0 - ssim(masked_pred, target)


def _luma_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_side: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_luma = _luma(_resize_bchw_tensor(pred, int(max_side)))
    target_luma = _luma(_resize_bchw_tensor(target, int(max_side)))
    pred_dx = pred_luma[:, :, :, 1:] - pred_luma[:, :, :, :-1]
    pred_dy = pred_luma[:, :, 1:, :] - pred_luma[:, :, :-1, :]
    target_dx = target_luma[:, :, :, 1:] - target_luma[:, :, :, :-1]
    target_dy = target_luma[:, :, 1:, :] - target_luma[:, :, :-1, :]
    diff_dx = torch.abs(pred_dx - target_dx)
    diff_dy = torch.abs(pred_dy - target_dy)
    if mask is None:
        return torch.mean(diff_dx) + torch.mean(diff_dy)
    mask_r = _resize_mask_bchw(mask.to(device=pred.device, dtype=pred.dtype), pred_luma.shape[-2:])
    mask_dx = torch.maximum(mask_r[:, :, :, 1:], mask_r[:, :, :, :-1])
    mask_dy = torch.maximum(mask_r[:, :, 1:, :], mask_r[:, :, :-1, :])
    denom_dx = torch.clamp(mask_dx.sum(), min=1.0)
    denom_dy = torch.clamp(mask_dy.sum(), min=1.0)
    return torch.sum(diff_dx * mask_dx) / denom_dx + torch.sum(diff_dy * mask_dy) / denom_dy


def _local_high_frequency(x: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    low = F.avg_pool2d(padded, kernel_size=3, stride=1)
    return x - low


def _multiscale_highfreq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_side: int,
    levels: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(levels) <= 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    pred_s = _resize_bchw_tensor(pred, int(max_side))
    target_s = _resize_bchw_tensor(target, int(max_side))
    mask_s = (
        _resize_mask_bchw(mask.to(device=pred.device, dtype=pred.dtype), pred_s.shape[-2:])
        if mask is not None
        else None
    )
    total = torch.zeros((), dtype=pred.dtype, device=pred.device)
    weight_sum = 0.0
    cur_pred = pred_s
    cur_target = target_s
    cur_mask = mask_s
    for level in range(int(levels)):
        if cur_pred.shape[-2] < 4 or cur_pred.shape[-1] < 4:
            break
        weight = 1.0 / float(2**level)
        total = total + float(weight) * _masked_l1_loss(
            _local_high_frequency(cur_pred),
            _local_high_frequency(cur_target),
            cur_mask,
        )
        weight_sum += float(weight)
        if level == int(levels) - 1:
            break
        cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
        cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2)
        if cur_mask is not None:
            cur_mask = (F.avg_pool2d(cur_mask, kernel_size=2, stride=2) > 0.25).to(dtype=cur_pred.dtype)
    if weight_sum <= 0.0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return total / float(weight_sum)


def _teacher_residual_projection_losses(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mask: torch.Tensor | None,
    min_l1: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = (torch.sum(torch.abs(target_delta), dim=1, keepdim=True) >= float(min_l1)).to(dtype=pred_delta.dtype)
    if mask is not None:
        active = active * _resize_mask_bchw(mask.to(device=pred_delta.device, dtype=pred_delta.dtype), pred_delta.shape[-2:])
    active_count = torch.sum(active)
    zero = torch.zeros((), dtype=pred_delta.dtype, device=pred_delta.device)
    if float(active_count.detach().cpu().item()) <= 0.0:
        return zero, zero, zero, zero

    pred = pred_delta * active
    target = target_delta * active
    pred_sq = torch.sum(pred * pred)
    target_sq = torch.sum(target * target)
    active_fraction = active_count / float(max(1, target_delta.shape[0] * target_delta.shape[2] * target_delta.shape[3]))
    if float(target_sq.detach().cpu().item()) <= 1.0e-12:
        return zero, zero, zero, active_fraction.detach()

    dot = torch.sum(pred * target)
    cosine = dot / torch.sqrt(torch.clamp(pred_sq * target_sq, min=1.0e-12))
    cosine_loss = 1.0 - torch.clamp(cosine, -1.0, 1.0)
    denom = torch.clamp(active_count * float(pred_delta.shape[1]), min=1.0)
    pred_rms = torch.sqrt(torch.clamp(pred_sq / denom, min=1.0e-12))
    target_rms = torch.sqrt(torch.clamp(target_sq / denom, min=1.0e-12))
    energy_loss = torch.abs(pred_rms - target_rms)
    return cosine_loss, energy_loss, cosine.detach(), active_fraction.detach()


def _build_residual_debt_mask(
    parent: torch.Tensor,
    teacher: torch.Tensor,
    gt: torch.Tensor | None,
    *,
    quantile: float,
    min_l1: float,
    dilate: int,
) -> torch.Tensor:
    """Train-fit-only mask for regions where the parent has real residual debt."""
    reference = gt if gt is not None else teacher
    residual_l1 = torch.mean(torch.abs(reference[:3] - parent[:3]), dim=0, keepdim=True)
    q = min(max(float(quantile), 0.0), 1.0)
    threshold = torch.quantile(residual_l1.reshape(-1), q)
    threshold = torch.maximum(
        threshold,
        torch.as_tensor(float(min_l1), dtype=residual_l1.dtype, device=residual_l1.device),
    )
    mask = (residual_l1 >= threshold).to(dtype=parent.dtype)
    radius = max(0, int(dilate))
    if radius > 0:
        kernel = 2 * radius + 1
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
    return mask


def _build_teacher_benefit_mask(
    z: np.lib.npyio.NpzFile,
    *,
    mode: str,
    min_gain_l1: float,
    dilate: int,
) -> torch.Tensor | None:
    """Train-fit-only mask where the Phase-J teacher is certified better than parent."""
    mode = str(mode)
    if mode == "off":
        return None
    better = None
    if "teacher_better_mask" in z:
        better = np.asarray(z["teacher_better_mask"], dtype=np.float32) > 0.0
    gain = None
    if "teacher_gain_l1" in z:
        gain = np.asarray(z["teacher_gain_l1"], dtype=np.float32)
    if mode == "teacher_better":
        mask_np = better
    elif mode == "positive_gain":
        mask_np = None if gain is None else gain > float(min_gain_l1)
    elif mode == "better_and_positive_gain":
        if better is None or gain is None:
            mask_np = better if gain is None else gain > float(min_gain_l1)
        else:
            mask_np = better & (gain > float(min_gain_l1))
    else:
        raise ValueError(f"unknown teacher_benefit_mask_mode: {mode}")
    if mask_np is None:
        return None
    mask = torch.from_numpy(mask_np.astype(np.float32)).reshape(1, *mask_np.shape[-2:])
    radius = max(0, int(dilate))
    if radius > 0:
        kernel = 2 * radius + 1
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
    return torch.clamp(mask, 0.0, 1.0)


def _blend_target_with_parent(parent: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return target
    mask = _resize_mask_bchw(mask.to(device=parent.device, dtype=parent.dtype), parent.shape[-2:])
    return target * mask + parent * (1.0 - mask)


def _camera_dir(z: np.lib.npyio.NpzFile, h: int, w: int) -> np.ndarray:
    cam = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
    cam = cam / max(float(np.linalg.norm(cam)), 1.0e-8)
    return np.broadcast_to(cam.reshape(3, 1, 1), (3, h, w)).astype(np.float32)


def _load_input_chw(z: np.lib.npyio.NpzFile) -> np.ndarray:
    parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0)
    _, h, w = parent.shape
    normal = np.clip(_to_chw(z["normal"])[:3], -1.0, 1.0)
    if normal.shape[1:] != (h, w):
        normal = np.zeros((3, h, w), dtype=np.float32)
    depth = np.asarray(z["depth"], dtype=np.float32)
    inv_depth = (1.0 / (1.0 + np.maximum(depth, 0.0))).reshape(1, h, w).astype(np.float32)
    alpha = np.clip(np.asarray(z["alpha"], dtype=np.float32).reshape(1, h, w), 0.0, 1.0)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    if bary.shape[0] != 3:
        bary = np.zeros((3, h, w), dtype=np.float32)
    bary = np.clip(np.nan_to_num(bary, nan=0.0, posinf=1.0, neginf=0.0), -0.05, 1.05)
    texture = _to_chw(z["texture"])
    if texture.shape[1:] != (h, w):
        texture = np.zeros((1, h, w), dtype=np.float32)
    texture = np.clip(texture[:1], 0.0, 1.0)
    cam = _camera_dir(z, h, w)
    valid = (np.asarray(z["face_id"], dtype=np.int64) >= 0).reshape(1, h, w).astype(np.float32)
    if "barycentric_valid" in z:
        valid *= np.asarray(z["barycentric_valid"]).reshape(1, h, w).astype(np.float32)
    return np.concatenate([parent, normal, inv_depth, alpha, bary, texture, cam, valid], axis=0).astype(np.float32)


def _append_alpha_channel_chw(features: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha_ch = torch.full(
        (1, features.shape[-2], features.shape[-1]),
        float(alpha),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, alpha_ch], dim=0)


def _load_example(
    path: Path,
    residual_key: str,
    max_side: int,
    include_gt: bool,
    face_lut: np.ndarray | None = None,
    residual_debt_mask: bool = False,
    residual_debt_quantile: float = 0.70,
    residual_debt_min_l1: float = 1.0 / 255.0,
    residual_debt_dilate: int = 1,
    teacher_benefit_mask_mode: str = "off",
    teacher_benefit_min_gain_l1: float = 0.0,
    teacher_benefit_dilate: int = 1,
) -> dict[str, Any]:
    z = np.load(path)
    parent = torch.from_numpy(np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32))
    residual = torch.from_numpy(np.asarray(z[residual_key], dtype=np.float32)[:3])
    teacher = torch.clamp(parent + residual, 0.0, 1.0)
    features = torch.from_numpy(_load_input_chw(z))
    resized_parent = _resize_chw_tensor(parent, max_side)
    resized_teacher = _resize_chw_tensor(teacher, max_side)
    out: dict[str, Any] = {
        "name": path.stem,
        "features": _resize_chw_tensor(features, max_side),
        "face_ids": _load_face_ids_tensor(z, face_lut, max_side),
        "parent": resized_parent,
        "teacher": resized_teacher,
    }
    resized_gt = None
    if include_gt and "rgb_gt" in z:
        gt = torch.from_numpy(np.clip(_to_chw(z["rgb_gt"])[:3], 0.0, 1.0).astype(np.float32))
        resized_gt = _resize_chw_tensor(gt, max_side)
        out["gt"] = resized_gt
    if bool(residual_debt_mask):
        out["residual_debt_mask"] = _build_residual_debt_mask(
            resized_parent,
            resized_teacher,
            resized_gt,
            quantile=float(residual_debt_quantile),
            min_l1=float(residual_debt_min_l1),
            dilate=int(residual_debt_dilate),
        )
    benefit_mask = _build_teacher_benefit_mask(
        z,
        mode=str(teacher_benefit_mask_mode),
        min_gain_l1=float(teacher_benefit_min_gain_l1),
        dilate=int(teacher_benefit_dilate),
    )
    if benefit_mask is not None:
        out["teacher_benefit_mask"] = _resize_chw_tensor(benefit_mask, max_side)
    return out


def verify_target_no_gt(evidence_dir: Path) -> dict[str, Any]:
    bad = []
    samples = []
    paths = evidence_views(evidence_dir)
    for path in paths:
        z = np.load(path)
        keys = set(z.files)
        present = sorted(keys & FORBIDDEN_TARGET_KEYS)
        if len(samples) < 4:
            samples.append({"path": str(path), "keys": sorted(keys)})
        if present:
            bad.append({"path": str(path), "forbidden_keys": present})
    return {
        "schema": "spcarnet_target_no_gt_verify_v184",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(FORBIDDEN_TARGET_KEYS),
        "sample_keys": samples,
        "bad_view_count": int(len(bad)),
        "bad_views": bad[:32],
        "target_gt_visible_to_apply": bool(any("rgb_gt" in row.get("forbidden_keys", []) for row in bad)),
        "target_residual_visible_to_apply": bool(
            any(set(row.get("forbidden_keys", [])) - {"rgb_gt"} for row in bad)
        ),
        "passed": len(bad) == 0,
    }


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch),
            torch.nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurfaceConditionedResidualUNet(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        max_delta: float,
        *,
        confidence_mode: str = "none",
        confidence_bias: float = 2.0,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
    ):
        super().__init__()
        confidence_mode = str(confidence_mode)
        if confidence_mode not in {"none", "sigmoid"}:
            raise ValueError(f"unknown confidence_mode: {confidence_mode}")
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.mid = ConvBlock(base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 6, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 3, base_ch)
        self.confidence_mode = confidence_mode
        self.confidence_bias = float(confidence_bias)
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        self.out = torch.nn.Conv2d(base_ch, 4 if confidence_mode == "sigmoid" else 3, 1)
        if confidence_mode == "sigmoid":
            torch.nn.init.constant_(self.out.bias[3], float(confidence_bias))
        self.max_delta = float(max_delta)

    def forward(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        del face_ids
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))
        m = self.mid(e3)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        raw = self.out(d1)
        delta = torch.tanh(raw[:, :3]) * self.max_delta
        if self.confidence_mode == "sigmoid":
            conf = torch.sigmoid(raw[:, 3:4])
            conf = self.confidence_min + (self.confidence_max - self.confidence_min) * conf
            return delta * conf
        return delta


@dataclass
class TrainBatch:
    features: torch.Tensor
    face_ids: torch.Tensor | None
    parent: torch.Tensor
    teacher: torch.Tensor
    gt: torch.Tensor | None
    residual_debt_mask: torch.Tensor | None
    teacher_benefit_mask: torch.Tensor | None


def _collect_train_face_lut(paths: list[Path], max_unique: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in tqdm(paths, desc="collect train face ids"):
        z = np.load(path)
        face_id = np.asarray(z["face_id"], dtype=np.int64).reshape(-1)
        face_id = face_id[face_id >= 0]
        if face_id.size:
            chunks.append(np.unique(face_id))
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    merged = np.unique(np.concatenate(chunks)).astype(np.int64)
    if int(max_unique) > 0 and merged.size > int(max_unique):
        # Deterministic subsampling keeps memory bounded while preserving broad surface coverage.
        keep = np.linspace(0, merged.size - 1, int(max_unique)).round().astype(np.int64)
        merged = merged[keep]
    return merged


def _load_residual_l1_npz(
    z: np.lib.npyio.NpzFile,
    *,
    residual_l1_key: str,
    residual_rgb_key: str,
) -> np.ndarray | None:
    if residual_l1_key in z:
        return np.asarray(z[residual_l1_key], dtype=np.float32)
    if residual_rgb_key in z:
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        if residual.ndim == 3:
            return np.mean(np.abs(residual[:3]), axis=0).astype(np.float32)
    return None


def _collect_residual_top_face_lut(
    paths: list[Path],
    *,
    residual_l1_key: str,
    residual_rgb_key: str,
    max_unique: int,
    min_alpha: float,
    min_residual_l1: float,
    priority_face_counts: dict[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    views_used = 0
    for path in tqdm(paths, desc="collect residual top face ids"):
        z = np.load(path)
        if "face_id" not in z:
            continue
        residual_l1 = _load_residual_l1_npz(
            z,
            residual_l1_key=residual_l1_key,
            residual_rgb_key=residual_rgb_key,
        )
        if residual_l1 is None:
            continue
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        mask = face_id >= 0
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        mask &= residual_l1 >= float(min_residual_l1)
        if not np.any(mask):
            continue
        views_used += 1
        faces, inverse = np.unique(face_id[mask], return_inverse=True)
        residual_values = residual_l1[mask].astype(np.float64)
        sums = np.bincount(inverse, weights=residual_values)
        nums = np.bincount(inverse)
        for face, score, count in zip(faces, sums, nums, strict=False):
            face_i = int(face)
            if face_i < 0:
                continue
            scores[face_i] = scores.get(face_i, 0.0) + float(score)
            counts[face_i] = counts.get(face_i, 0) + int(count)
    priority_face_counts = priority_face_counts or {}
    ranked = sorted(
        scores,
        key=lambda face: (
            1 if face in priority_face_counts else 0,
            priority_face_counts.get(face, 0),
            scores[face],
            counts.get(face, 0),
        ),
        reverse=True,
    )
    if int(max_unique) > 0:
        ranked = ranked[: int(max_unique)]
    face_lut = np.asarray(sorted(ranked), dtype=np.int64)
    summary = {
        "schema": "spcarnet_residual_top_face_lut_v1",
        "views_requested": int(len(paths)),
        "views_used": int(views_used),
        "residual_l1_key": str(residual_l1_key),
        "min_alpha": float(min_alpha),
        "min_residual_l1": float(min_residual_l1),
        "max_unique": int(max_unique),
        "selected_faces": int(face_lut.size),
        "total_scored_faces": int(len(scores)),
        "priority_face_count": int(len(priority_face_counts)),
        "selected_priority_faces": int(sum(1 for face in ranked if face in priority_face_counts)),
        "top_faces": [
            {
                "face_id": int(face),
                "score": float(scores[face]),
                "samples": int(counts.get(face, 0)),
                "priority_samples": int(priority_face_counts.get(face, 0)),
            }
            for face in ranked[:32]
        ],
        "target_or_test_gt_used": False,
    }
    return face_lut, summary


def _collect_visible_face_counts(
    evidence_dir: Path,
    *,
    min_alpha: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    counts: dict[int, int] = {}
    paths = evidence_views(evidence_dir)
    views_used = 0
    for path in tqdm(paths, desc="collect target-visible face ids"):
        z = np.load(path)
        if "face_id" not in z:
            continue
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        mask = face_id >= 0
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        if not np.any(mask):
            continue
        views_used += 1
        faces, nums = np.unique(face_id[mask], return_counts=True)
        for face, count in zip(faces, nums, strict=False):
            face_i = int(face)
            if face_i >= 0:
                counts[face_i] = counts.get(face_i, 0) + int(count)
    summary = {
        "schema": "spcarnet_target_visible_face_counts_v1",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "views_used": int(views_used),
        "min_alpha": float(min_alpha),
        "visible_faces": int(len(counts)),
        "top_faces": [
            {"face_id": int(face), "samples": int(count)}
            for face, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:32]
        ],
        "target_or_test_gt_used": False,
    }
    return counts, summary


def _texture_row_indices_np(
    face_id: np.ndarray,
    barycentric: np.ndarray,
    face_lut: np.ndarray,
    texture_size: int,
) -> np.ndarray:
    compact = _compact_face_ids(np.asarray(face_id, dtype=np.int64), face_lut)
    bary = np.asarray(barycentric, dtype=np.float32)
    if bary.shape[0] != 3:
        return np.zeros(compact.shape, dtype=np.int64)
    u = np.clip(bary[1], 0.0, 0.999999)
    v = np.clip(bary[2], 0.0, 0.999999)
    ubin = np.clip((u * float(texture_size)).astype(np.int64), 0, int(texture_size) - 1)
    vbin = np.clip((v * float(texture_size)).astype(np.int64), 0, int(texture_size) - 1)
    bin_id = vbin * int(texture_size) + ubin
    rows_per_face = int(texture_size) * int(texture_size)
    valid = compact > 0
    out = np.zeros(compact.shape, dtype=np.int64)
    out[valid] = (compact[valid] - 1) * rows_per_face + bin_id[valid] + 1
    return out


def _collect_surface_texture_support_stats(
    paths: list[Path],
    *,
    face_lut: np.ndarray,
    residual_l1_key: str,
    residual_rgb_key: str,
    texture_size: int,
    min_alpha: float,
    min_residual_l1: float,
    min_bin_support: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows_per_face = int(texture_size) * int(texture_size)
    rows = int(face_lut.size) * rows_per_face + 1
    support = np.zeros((rows,), dtype=np.int64)
    residual_sum = np.zeros((rows,), dtype=np.float64)
    views_used = 0
    for path in tqdm(paths, desc="collect low-rank surface support"):
        z = np.load(path)
        if "face_id" not in z or "barycentric" not in z:
            continue
        residual_l1 = _load_residual_l1_npz(
            z,
            residual_l1_key=residual_l1_key,
            residual_rgb_key=residual_rgb_key,
        )
        if residual_l1 is None:
            continue
        row_id = _texture_row_indices_np(z["face_id"], z["barycentric"], face_lut, int(texture_size))
        mask = row_id > 0
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        mask &= residual_l1 >= float(min_residual_l1)
        if not np.any(mask):
            continue
        views_used += 1
        ids = row_id[mask].reshape(-1)
        weights = residual_l1[mask].astype(np.float64).reshape(-1)
        support += np.bincount(ids, minlength=rows).astype(np.int64)
        residual_sum += np.bincount(ids, weights=weights, minlength=rows)
    mean_l1 = np.divide(
        residual_sum,
        np.maximum(support, 1),
        out=np.zeros_like(residual_sum, dtype=np.float64),
        where=support > 0,
    )
    max_support = int(max(1, int(support.max()) if support.size else 0))
    max_mean_l1 = float(max(1.0e-8, float(mean_l1.max()) if mean_l1.size else 0.0))
    log_support = np.log1p(support.astype(np.float64)) / math.log1p(float(max_support))
    mean_l1_norm = mean_l1 / max_mean_l1
    active = (support >= int(min_bin_support)).astype(np.float32)
    stats = np.stack([log_support, mean_l1_norm, active], axis=1).astype(np.float32)
    stats[0] = 0.0
    active_rows = int(np.sum(active))
    summary = {
        "schema": "spcarnet_lowrank_surface_support_stats_v1",
        "views_requested": int(len(paths)),
        "views_used": int(views_used),
        "residual_l1_key": str(residual_l1_key),
        "texture_size": int(texture_size),
        "rows": int(rows),
        "rows_per_face": int(rows_per_face),
        "selected_faces": int(face_lut.size),
        "min_alpha": float(min_alpha),
        "min_residual_l1": float(min_residual_l1),
        "min_bin_support": int(min_bin_support),
        "active_rows": int(active_rows),
        "active_row_fraction": float(active_rows / max(1, rows - 1)),
        "max_support": int(max_support),
        "mean_support_active": float(np.mean(support[active > 0])) if active_rows else 0.0,
        "mean_residual_l1_active": float(np.mean(mean_l1[active > 0])) if active_rows else 0.0,
        "target_or_test_gt_used": False,
    }
    return stats, summary


def _collect_surface_evidence_texture_stats(
    paths: list[Path],
    *,
    face_lut: np.ndarray,
    residual_rgb_key: str,
    residual_l1_key: str,
    texture_size: int,
    min_alpha: float,
    min_residual_l1: float,
    min_bin_support: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows_per_face = int(texture_size) * int(texture_size)
    rows = int(face_lut.size) * rows_per_face + 1
    support = np.zeros((rows,), dtype=np.int64)
    residual_sum = np.zeros((rows, 3), dtype=np.float64)
    residual_sq_sum = np.zeros((rows, 3), dtype=np.float64)
    residual_l1_sum = np.zeros((rows,), dtype=np.float64)
    cam_sum = np.zeros((rows, 3), dtype=np.float64)
    views_used = 0
    for path in tqdm(paths, desc="collect surface evidence texture"):
        z = np.load(path)
        if "face_id" not in z or "barycentric" not in z or residual_rgb_key not in z:
            continue
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
        residual_l1 = _load_residual_l1_npz(
            z,
            residual_l1_key=residual_l1_key,
            residual_rgb_key=residual_rgb_key,
        )
        if residual_l1 is None:
            residual_l1 = np.mean(np.abs(residual), axis=0).astype(np.float32)
        row_id = _texture_row_indices_np(z["face_id"], z["barycentric"], face_lut, int(texture_size))
        mask = row_id > 0
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        mask &= np.asarray(residual_l1, dtype=np.float32) >= float(min_residual_l1)
        if not np.any(mask):
            continue
        views_used += 1
        ids = row_id[mask].reshape(-1)
        counts = np.bincount(ids, minlength=rows)
        support += counts.astype(np.int64)
        residual_l1_sum += np.bincount(ids, weights=np.asarray(residual_l1, dtype=np.float32)[mask], minlength=rows)
        for ch in range(3):
            values = residual[ch][mask].astype(np.float64)
            residual_sum[:, ch] += np.bincount(ids, weights=values, minlength=rows)
            residual_sq_sum[:, ch] += np.bincount(ids, weights=values * values, minlength=rows)
        cam = np.asarray(z["camera_center"], dtype=np.float64).reshape(3)
        cam_norm = float(np.linalg.norm(cam))
        if cam_norm > 1.0e-8:
            cam = cam / cam_norm
            counts_f = counts.astype(np.float64)
            for ch in range(3):
                cam_sum[:, ch] += counts_f * float(cam[ch])
    denom = np.maximum(support.reshape(-1, 1), 1)
    mean_rgb = residual_sum / denom
    var_rgb = np.maximum(residual_sq_sum / denom - mean_rgb * mean_rgb, 0.0)
    std_rgb = np.sqrt(var_rgb)
    mean_l1 = residual_l1_sum / np.maximum(support, 1)
    max_support = int(max(1, int(support.max()) if support.size else 0))
    max_mean_l1 = float(max(1.0e-8, float(mean_l1.max()) if mean_l1.size else 0.0))
    log_support = np.log1p(support.astype(np.float64)) / math.log1p(float(max_support))
    mean_l1_norm = mean_l1 / max_mean_l1
    active = (support >= int(min_bin_support)).astype(np.float32)
    mean_cam = cam_sum / denom
    cam_norm = np.linalg.norm(mean_cam, axis=1, keepdims=True)
    mean_cam = np.divide(mean_cam, np.maximum(cam_norm, 1.0e-8), out=np.zeros_like(mean_cam), where=cam_norm > 0)
    stats = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            mean_l1_norm.reshape(-1, 1),
            log_support.reshape(-1, 1),
            active.reshape(-1, 1),
            mean_cam,
        ],
        axis=1,
    ).astype(np.float16)
    stats[0] = 0.0
    active_rows = int(np.sum(active))
    summary = {
        "schema": "spcarnet_surface_evidence_texture_stats_v1",
        "views_requested": int(len(paths)),
        "views_used": int(views_used),
        "residual_rgb_key": str(residual_rgb_key),
        "residual_l1_key": str(residual_l1_key),
        "texture_size": int(texture_size),
        "rows": int(rows),
        "rows_per_face": int(rows_per_face),
        "selected_faces": int(face_lut.size),
        "channels": [
            "mean_residual_r",
            "mean_residual_g",
            "mean_residual_b",
            "std_residual_r",
            "std_residual_g",
            "std_residual_b",
            "mean_residual_l1_norm",
            "log_support",
            "active_support",
            "mean_source_cam_x",
            "mean_source_cam_y",
            "mean_source_cam_z",
        ],
        "min_alpha": float(min_alpha),
        "min_residual_l1": float(min_residual_l1),
        "min_bin_support": int(min_bin_support),
        "active_rows": int(active_rows),
        "active_row_fraction": float(active_rows / max(1, rows - 1)),
        "max_support": int(max_support),
        "mean_support_active": float(np.mean(support[active > 0])) if active_rows else 0.0,
        "mean_residual_l1_active": float(np.mean(mean_l1[active > 0])) if active_rows else 0.0,
        "mean_abs_residual_rgb_active": float(np.mean(np.abs(mean_rgb[active > 0]))) if active_rows else 0.0,
        "target_or_test_gt_used": False,
    }
    return stats, summary


def _collect_surface_source_evidence_bank_stats(
    paths: list[Path],
    *,
    face_lut: np.ndarray,
    residual_rgb_key: str,
    residual_l1_key: str,
    texture_size: int,
    top_k: int,
    min_alpha: float,
    min_residual_l1: float,
    min_bin_support: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows_per_face = int(texture_size) * int(texture_size)
    rows = int(face_lut.size) * rows_per_face + 1
    k = max(1, int(top_k))
    channels_per_source = 8
    slot_score = np.full((rows, k), -np.inf, dtype=np.float32)
    slot_values = np.zeros((rows, k, channels_per_source), dtype=np.float16)
    row_support_total = np.zeros((rows,), dtype=np.int64)
    row_view_count = np.zeros((rows,), dtype=np.int32)
    views_used = 0
    source_observations = 0
    for path in tqdm(paths, desc="collect source evidence bank"):
        z = np.load(path)
        if "face_id" not in z or "barycentric" not in z or residual_rgb_key not in z:
            continue
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
        residual_l1 = _load_residual_l1_npz(
            z,
            residual_l1_key=residual_l1_key,
            residual_rgb_key=residual_rgb_key,
        )
        if residual_l1 is None:
            residual_l1 = np.mean(np.abs(residual), axis=0).astype(np.float32)
        row_id = _texture_row_indices_np(z["face_id"], z["barycentric"], face_lut, int(texture_size))
        mask = row_id > 0
        if "barycentric_valid" in z:
            mask &= np.asarray(z["barycentric_valid"]).astype(bool)
        if "alpha" in z:
            mask &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
        mask &= np.asarray(residual_l1, dtype=np.float32) >= float(min_residual_l1)
        if not np.any(mask):
            continue
        ids = row_id[mask].reshape(-1)
        counts = np.bincount(ids, minlength=rows).astype(np.int64)
        active_rows = np.nonzero(counts >= int(min_bin_support))[0]
        if active_rows.size == 0:
            continue
        views_used += 1
        row_support_total[active_rows] += counts[active_rows]
        row_view_count[active_rows] += 1
        residual_l1_sum = np.bincount(
            ids,
            weights=np.asarray(residual_l1, dtype=np.float32)[mask].reshape(-1).astype(np.float64),
            minlength=rows,
        )
        mean_l1 = residual_l1_sum[active_rows] / np.maximum(counts[active_rows], 1)
        mean_rgb = np.zeros((active_rows.size, 3), dtype=np.float32)
        for ch in range(3):
            channel_sum = np.bincount(
                ids,
                weights=residual[ch][mask].reshape(-1).astype(np.float64),
                minlength=rows,
            )
            mean_rgb[:, ch] = (channel_sum[active_rows] / np.maximum(counts[active_rows], 1)).astype(np.float32)
        cam = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
        cam_norm = float(np.linalg.norm(cam))
        if cam_norm > 1.0e-8:
            cam = cam / cam_norm
        else:
            cam = np.zeros((3,), dtype=np.float32)
        score = (mean_l1.astype(np.float32) * np.log1p(counts[active_rows]).astype(np.float32)).astype(np.float32)
        current = slot_score[active_rows]
        replace_slot = np.argmin(current, axis=1)
        replace_score = current[np.arange(active_rows.size), replace_slot]
        replace = score > replace_score
        if not np.any(replace):
            continue
        rows_replace = active_rows[replace]
        slots_replace = replace_slot[replace]
        slot_score[rows_replace, slots_replace] = score[replace]
        values = np.zeros((rows_replace.size, channels_per_source), dtype=np.float32)
        values[:, :3] = mean_rgb[replace]
        values[:, 3:6] = cam.reshape(1, 3)
        values[:, 6] = mean_l1[replace].astype(np.float32)
        values[:, 7] = counts[rows_replace].astype(np.float32)
        slot_values[rows_replace, slots_replace] = values.astype(np.float16)
        source_observations += int(rows_replace.size)
    valid_slot = np.isfinite(slot_score)
    raw_l1 = slot_values[..., 6].astype(np.float32)
    raw_support = slot_values[..., 7].astype(np.float32)
    max_l1 = float(max(1.0e-8, float(raw_l1[valid_slot].max()) if np.any(valid_slot) else 0.0))
    max_support = float(max(1.0, float(raw_support[valid_slot].max()) if np.any(valid_slot) else 1.0))
    slot_values[..., 6] = np.where(valid_slot, raw_l1 / max_l1, 0.0).astype(np.float16)
    slot_values[..., 7] = np.where(
        valid_slot,
        np.log1p(raw_support) / math.log1p(max_support),
        0.0,
    ).astype(np.float16)
    stats = slot_values.reshape(rows, k * channels_per_source).astype(np.float16)
    stats[0] = 0.0
    active_rows = int(np.sum(np.any(valid_slot, axis=1)))
    filled_slots = int(np.sum(valid_slot))
    summary = {
        "schema": "spcarnet_surface_source_evidence_bank_stats_v1",
        "views_requested": int(len(paths)),
        "views_used": int(views_used),
        "residual_rgb_key": str(residual_rgb_key),
        "residual_l1_key": str(residual_l1_key),
        "texture_size": int(texture_size),
        "rows": int(rows),
        "rows_per_face": int(rows_per_face),
        "selected_faces": int(face_lut.size),
        "top_k": int(k),
        "channels_per_source": int(channels_per_source),
        "channels": [
            "residual_r",
            "residual_g",
            "residual_b",
            "source_cam_x",
            "source_cam_y",
            "source_cam_z",
            "residual_l1_norm",
            "log_source_pixel_support",
        ],
        "min_alpha": float(min_alpha),
        "min_residual_l1": float(min_residual_l1),
        "min_bin_support": int(min_bin_support),
        "active_rows": int(active_rows),
        "active_row_fraction": float(active_rows / max(1, rows - 1)),
        "filled_slots": int(filled_slots),
        "mean_filled_slots_active": float(filled_slots / max(1, active_rows)),
        "source_observations_inserted": int(source_observations),
        "max_source_pixel_support": float(max_support),
        "max_source_residual_l1": float(max_l1),
        "mean_support_active_row": float(np.mean(row_support_total[row_support_total > 0]))
        if np.any(row_support_total > 0)
        else 0.0,
        "mean_source_views_active_row": float(np.mean(row_view_count[row_view_count > 0]))
        if np.any(row_view_count > 0)
        else 0.0,
        "target_or_test_gt_used": False,
    }
    return stats, summary


def _compact_face_ids(raw_face_id: np.ndarray, face_lut: np.ndarray) -> np.ndarray:
    face_id = np.asarray(raw_face_id, dtype=np.int64)
    out = np.zeros(face_id.shape, dtype=np.int64)
    valid = face_id >= 0
    if not valid.any() or face_lut.size == 0:
        return out
    flat = face_id[valid]
    idx = np.searchsorted(face_lut, flat)
    ok = (idx < face_lut.size) & (face_lut[idx.clip(max=max(face_lut.size - 1, 0))] == flat)
    compact = np.zeros(flat.shape, dtype=np.int64)
    compact[ok] = idx[ok] + 1
    out[valid] = compact
    return out


def _resize_face_ids(face_ids: torch.Tensor, max_side: int) -> torch.Tensor:
    if int(max_side) <= 0:
        return face_ids
    h, w = face_ids.shape
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return face_ids
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    resized = F.interpolate(face_ids.float().view(1, 1, h, w), size=size, mode="nearest")
    return resized.view(size).long()


def _load_face_ids_tensor(z: np.lib.npyio.NpzFile, face_lut: np.ndarray | None, max_side: int) -> torch.Tensor | None:
    if face_lut is None:
        return None
    compact = _compact_face_ids(np.asarray(z["face_id"], dtype=np.int64), face_lut)
    face_ids = torch.from_numpy(compact.astype(np.int64))
    return _resize_face_ids(face_ids, max_side)


class SurfaceConditionedFaceEmbeddingUNet(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        max_delta: float,
        num_faces: int,
        embedding_dim: int,
        *,
        confidence_mode: str = "none",
        confidence_bias: float = 2.0,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
    ):
        super().__init__()
        self.face_embedding = torch.nn.Embedding(int(num_faces), int(embedding_dim), padding_idx=0)
        torch.nn.init.zeros_(self.face_embedding.weight)
        self.unet = SurfaceConditionedResidualUNet(
            in_ch + int(embedding_dim),
            base_ch,
            max_delta,
            confidence_mode=confidence_mode,
            confidence_bias=confidence_bias,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )

    def forward(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SurfaceConditionedFaceEmbeddingUNet")
        emb = self.face_embedding(face_ids.long()).permute(0, 3, 1, 2)
        return self.unet(torch.cat([x, emb], dim=1))


class SurfaceTextureResidualMLP(torch.nn.Module):
    """Surface-attached neural texture plus a compact per-pixel decoder.

    Each train-fit face owns a small UV-bin feature texture.  At render time the
    model indexes that texture by compact face id and barycentric bin, then
    decodes residual RGB from the surface feature and local view/render buffers.
    Unknown target faces/bins map to a zero padding feature.
    """

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        max_delta: float,
        num_faces: int,
        texture_size: int,
        feature_dim: int,
        layers: int,
        *,
        confidence_mode: str = "none",
        confidence_bias: float = 2.0,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
    ):
        super().__init__()
        confidence_mode = str(confidence_mode)
        if confidence_mode not in {"none", "sigmoid"}:
            raise ValueError(f"unknown confidence_mode: {confidence_mode}")
        self.texture_size = int(texture_size)
        self.bins_per_face = int(texture_size) * int(texture_size)
        self.num_faces = int(num_faces)
        self.feature_dim = int(feature_dim)
        self.max_delta = float(max_delta)
        self.confidence_mode = confidence_mode
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        rows = int(num_faces) * self.bins_per_face + 1
        self.surface_texture = torch.nn.Embedding(rows, int(feature_dim), padding_idx=0)
        torch.nn.init.normal_(self.surface_texture.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.surface_texture.weight[0].zero_()
        decoder_layers: list[torch.nn.Module] = []
        ch = int(in_ch) + int(feature_dim)
        for _ in range(max(1, int(layers) - 1)):
            decoder_layers.extend(
                [
                    torch.nn.Conv2d(ch, int(hidden_ch), 1),
                    torch.nn.GroupNorm(max(1, min(8, int(hidden_ch) // 4)), int(hidden_ch)),
                    torch.nn.SiLU(inplace=True),
                ]
            )
            ch = int(hidden_ch)
        out_ch = 4 if confidence_mode == "sigmoid" else 3
        decoder_layers.append(torch.nn.Conv2d(ch, out_ch, 1))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        if confidence_mode == "sigmoid":
            last = self.decoder[-1]
            if isinstance(last, torch.nn.Conv2d):
                torch.nn.init.constant_(last.bias[3], float(confidence_bias))

    def _texture_indices(self, x: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 11:
            raise ValueError("surface texture model expects barycentric channels in input features")
        bary = x[:, 8:11]
        u = torch.clamp(bary[:, 1], 0.0, 0.999999)
        v = torch.clamp(bary[:, 2], 0.0, 0.999999)
        ubin = torch.clamp((u * float(self.texture_size)).long(), 0, self.texture_size - 1)
        vbin = torch.clamp((v * float(self.texture_size)).long(), 0, self.texture_size - 1)
        bin_id = vbin * int(self.texture_size) + ubin
        compact_face = face_ids.long()
        valid = compact_face > 0
        index = (compact_face - 1) * int(self.bins_per_face) + bin_id + 1
        return torch.where(valid, index, torch.zeros_like(index))

    def forward(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SurfaceTextureResidualMLP")
        tex = self.surface_texture(self._texture_indices(x, face_ids)).permute(0, 3, 1, 2)
        raw = self.decoder(torch.cat([x, tex], dim=1))
        delta = torch.tanh(raw[:, :3]) * self.max_delta
        if self.confidence_mode == "sigmoid":
            conf = torch.sigmoid(raw[:, 3:4])
            conf = self.confidence_min + (self.confidence_max - self.confidence_min) * conf
            return delta * conf
        return delta


class SurfaceTextureConditionedUNet(torch.nn.Module):
    """Surface neural texture rasterized into an image-context residual U-Net.

    Compared with SurfaceTextureResidualMLP, the per-face/UV feature texture is
    not decoded pointwise.  The feature map is concatenated with parent render,
    surface buffers, and camera channels, then decoded by a compact U-Net.  This
    gives the baked residual carrier local 2D context while keeping unknown
    target faces as deterministic zero-feature no-ops.
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        max_delta: float,
        num_faces: int,
        texture_size: int,
        feature_dim: int,
        support_stats: np.ndarray | torch.Tensor | None = None,
        *,
        confidence_mode: str = "none",
        confidence_bias: float = 2.0,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
        support_gate_floor: float = 0.0,
        support_unknown_gate_floor: float = 0.0,
        evidence_stats: np.ndarray | torch.Tensor | None = None,
        evidence_residual_prior_weight: float = 0.0,
        evidence_view_gate_power: float = 0.0,
        evidence_source_bank_top_k: int = 0,
        evidence_source_bank_channels_per_source: int = 8,
    ):
        super().__init__()
        self.texture_size = int(texture_size)
        self.bins_per_face = int(texture_size) * int(texture_size)
        self.num_faces = int(num_faces)
        self.feature_dim = int(feature_dim)
        self.support_gate_floor = float(support_gate_floor)
        self.support_unknown_gate_floor = float(support_unknown_gate_floor)
        self.evidence_residual_prior_weight = float(evidence_residual_prior_weight)
        self.evidence_view_gate_power = float(evidence_view_gate_power)
        self.evidence_source_bank_top_k = max(0, int(evidence_source_bank_top_k))
        self.evidence_source_bank_channels_per_source = max(1, int(evidence_source_bank_channels_per_source))
        rows = int(num_faces) * self.bins_per_face + 1
        if support_stats is not None:
            stats = torch.as_tensor(support_stats, dtype=torch.float32)
            if stats.ndim != 2 or stats.shape[0] != rows:
                raise ValueError(f"support_stats shape {tuple(stats.shape)} does not match rows={rows}")
        else:
            stats = None
        self.register_buffer("surface_support_stats", stats, persistent=True)
        if evidence_stats is not None:
            evidence = torch.as_tensor(evidence_stats)
            if evidence.ndim != 2 or evidence.shape[0] != rows:
                raise ValueError(f"evidence_stats shape {tuple(evidence.shape)} does not match rows={rows}")
            evidence = evidence.to(dtype=torch.float16 if evidence.shape[1] >= 8 else torch.float32)
        else:
            evidence = None
        self.register_buffer("surface_evidence_stats", evidence, persistent=True)
        evidence_dim = 0 if evidence is None else int(evidence.shape[1])
        self.surface_texture = torch.nn.Embedding(rows, int(feature_dim), padding_idx=0)
        torch.nn.init.normal_(self.surface_texture.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.surface_texture.weight[0].zero_()
        self.unet = SurfaceConditionedResidualUNet(
            int(in_ch) + int(feature_dim) + int(evidence_dim),
            int(base_ch),
            float(max_delta),
            confidence_mode=confidence_mode,
            confidence_bias=confidence_bias,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )

    def _texture_indices(self, x: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 11:
            raise ValueError("surface texture U-Net expects barycentric channels in input features")
        bary = x[:, 8:11]
        u = torch.clamp(bary[:, 1], 0.0, 0.999999)
        v = torch.clamp(bary[:, 2], 0.0, 0.999999)
        ubin = torch.clamp((u * float(self.texture_size)).long(), 0, self.texture_size - 1)
        vbin = torch.clamp((v * float(self.texture_size)).long(), 0, self.texture_size - 1)
        bin_id = vbin * int(self.texture_size) + ubin
        compact_face = face_ids.long()
        valid = compact_face > 0
        index = (compact_face - 1) * int(self.bins_per_face) + bin_id + 1
        return torch.where(valid, index, torch.zeros_like(index))

    def forward(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SurfaceTextureConditionedUNet")
        idx = self._texture_indices(x, face_ids)
        tex = self.surface_texture(idx)
        gate = None
        if self.surface_support_stats is not None:
            gate = self._support_gate_from_indices(x, idx)
            tex = tex * gate.permute(0, 2, 3, 1)
        evidence_chw = None
        evidence_prior = None
        if self.surface_evidence_stats is not None:
            evidence = self.surface_evidence_stats[idx].to(dtype=x.dtype, device=x.device)
            evidence = evidence.clone()
            if self.evidence_source_bank_top_k > 0:
                evidence, evidence_prior = self._surface_source_bank_condition(x, evidence)
            elif evidence.shape[-1] >= 3:
                view_gate = self._surface_evidence_view_gate(x, evidence)
                evidence[..., :3] = evidence[..., :3] * view_gate.unsqueeze(-1)
                evidence_prior = evidence[..., :3].permute(0, 3, 1, 2)
            if gate is not None:
                evidence = evidence * gate.permute(0, 2, 3, 1)
            evidence_chw = evidence.permute(0, 3, 1, 2)
        tex = tex.permute(0, 3, 1, 2)
        parts = [x, tex]
        if evidence_chw is not None:
            parts.append(evidence_chw)
        delta = self.unet(torch.cat(parts, dim=1))
        if evidence_prior is not None and float(self.evidence_residual_prior_weight) != 0.0:
            delta = delta + float(self.evidence_residual_prior_weight) * evidence_prior
        if gate is not None:
            delta = delta * gate
        return delta

    def _surface_source_bank_condition(
        self,
        x: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        k = int(self.evidence_source_bank_top_k)
        c = int(self.evidence_source_bank_channels_per_source)
        if k <= 0 or c < 8 or evidence.shape[-1] < k * c:
            return evidence, None
        bank = evidence[..., : k * c].reshape(*evidence.shape[:3], k, c)
        raw_residual = bank[..., :3]
        source_cam = F.normalize(bank[..., 3:6], dim=-1, eps=1.0e-6)
        residual_conf = torch.clamp(bank[..., 6], 0.0, 1.0)
        support_conf = torch.clamp(bank[..., 7], 0.0, 1.0)
        score = residual_conf * support_conf
        if x.shape[1] > 14 and float(self.evidence_view_gate_power) > 0.0:
            target_cam = F.normalize(x[:, 12:15], dim=1, eps=1.0e-6).permute(0, 2, 3, 1).unsqueeze(-2)
            cosine = torch.sum(target_cam * source_cam, dim=-1).clamp(-1.0, 1.0)
            view_gate = torch.pow(
                torch.clamp(0.5 * (cosine + 1.0), 0.0, 1.0),
                float(self.evidence_view_gate_power),
            )
            score = score * view_gate
        active = (torch.sum(torch.abs(raw_residual), dim=-1) > 1.0e-8).to(dtype=x.dtype)
        score = score * active
        denom = torch.sum(score, dim=-1, keepdim=True).clamp(min=1.0e-6)
        weights = score / denom
        prior = torch.sum(weights.unsqueeze(-1) * raw_residual, dim=-2)
        prior = prior * torch.clamp(torch.sum(score, dim=-1, keepdim=True), 0.0, 1.0)
        bank = bank.clone()
        bank[..., :3] = raw_residual * score.unsqueeze(-1)
        bank[..., 6] = score
        evidence[..., : k * c] = bank.reshape(*evidence.shape[:3], k * c)
        return evidence, prior.permute(0, 3, 1, 2)

    def _surface_evidence_view_gate(self, x: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.shape[-1] < 12 or x.shape[1] <= 14:
            return torch.ones((*evidence.shape[:3],), dtype=x.dtype, device=x.device)
        power = float(self.evidence_view_gate_power)
        if power <= 0.0:
            return torch.ones((*evidence.shape[:3],), dtype=x.dtype, device=x.device)
        cam = F.normalize(x[:, 12:15], dim=1, eps=1.0e-6).permute(0, 2, 3, 1)
        mean_cam = F.normalize(evidence[..., 9:12].to(dtype=x.dtype), dim=-1, eps=1.0e-6)
        cosine = torch.sum(cam * mean_cam, dim=-1).clamp(-1.0, 1.0)
        return torch.pow(torch.clamp(0.5 * (cosine + 1.0), 0.0, 1.0), power)

    def _support_gate_from_indices(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.surface_support_stats is None:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        active = self.surface_support_stats[idx][..., 2].unsqueeze(1).to(dtype=x.dtype, device=x.device)
        if float(self.support_gate_floor) > 0.0 or float(self.support_unknown_gate_floor) > 0.0:
            known = (idx > 0).unsqueeze(1).to(dtype=x.dtype, device=x.device)
            known_floor = torch.full_like(active, min(max(float(self.support_gate_floor), 0.0), 1.0))
            unknown_floor = torch.full_like(active, min(max(float(self.support_unknown_gate_floor), 0.0), 1.0))
            floor = known * known_floor + (1.0 - known) * unknown_floor
            active = torch.maximum(active, floor)
            if x.shape[1] > 15:
                active = active * torch.clamp(x[:, 15:16], 0.0, 1.0)
        return torch.clamp(active, 0.0, 1.0)

    def support_mask(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SurfaceTextureConditionedUNet")
        if self.surface_support_stats is None:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        idx = self._texture_indices(x, face_ids)
        return self._support_gate_from_indices(x, idx)


class SupportAwareLowRankSurfaceTexture(torch.nn.Module):
    """Low-rank teacher residual texture with hard support-aware no-op gating.

    The texture stores K signed residual bases per selected face/UV bin.  A
    tiny view-conditioned decoder predicts only mixture weights and confidence.
    Unknown or low-support rows are deterministically gated to zero.
    """

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        max_delta: float,
        num_faces: int,
        texture_size: int,
        rank: int,
        layers: int,
        support_stats: np.ndarray | torch.Tensor,
        *,
        basis_init_std: float = 0.01,
        confidence_bias: float = 0.0,
        confidence_min: float = 0.0,
        confidence_max: float = 1.0,
    ):
        super().__init__()
        self.texture_size = int(texture_size)
        self.bins_per_face = int(texture_size) * int(texture_size)
        self.num_faces = int(num_faces)
        self.rank = int(rank)
        self.max_delta = float(max_delta)
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        rows = int(num_faces) * self.bins_per_face + 1
        stats = torch.as_tensor(support_stats, dtype=torch.float32)
        if stats.ndim != 2 or stats.shape[0] != rows:
            raise ValueError(f"support_stats shape {tuple(stats.shape)} does not match rows={rows}")
        self.register_buffer("surface_support_stats", stats, persistent=True)
        self.surface_basis = torch.nn.Embedding(rows, int(rank) * 3, padding_idx=0)
        torch.nn.init.normal_(self.surface_basis.weight, mean=0.0, std=float(basis_init_std))
        with torch.no_grad():
            self.surface_basis.weight[0].zero_()
        decoder_layers: list[torch.nn.Module] = []
        ch = int(in_ch) + int(stats.shape[1])
        for _ in range(max(1, int(layers) - 1)):
            decoder_layers.extend(
                [
                    torch.nn.Conv2d(ch, int(hidden_ch), 1),
                    torch.nn.GroupNorm(max(1, min(8, int(hidden_ch) // 4)), int(hidden_ch)),
                    torch.nn.SiLU(inplace=True),
                ]
            )
            ch = int(hidden_ch)
        decoder_layers.append(torch.nn.Conv2d(ch, int(rank) + 1, 1))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        last = self.decoder[-1]
        if isinstance(last, torch.nn.Conv2d):
            torch.nn.init.constant_(last.bias[int(rank)], float(confidence_bias))

    def _texture_indices(self, x: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 11:
            raise ValueError("low-rank surface texture model expects barycentric channels in input features")
        bary = x[:, 8:11]
        u = torch.clamp(bary[:, 1], 0.0, 0.999999)
        v = torch.clamp(bary[:, 2], 0.0, 0.999999)
        ubin = torch.clamp((u * float(self.texture_size)).long(), 0, self.texture_size - 1)
        vbin = torch.clamp((v * float(self.texture_size)).long(), 0, self.texture_size - 1)
        bin_id = vbin * int(self.texture_size) + ubin
        compact_face = face_ids.long()
        valid = compact_face > 0
        index = (compact_face - 1) * int(self.bins_per_face) + bin_id + 1
        return torch.where(valid, index, torch.zeros_like(index))

    def support_mask(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SupportAwareLowRankSurfaceTexture")
        idx = self._texture_indices(x, face_ids)
        return self.surface_support_stats[idx][..., 2].unsqueeze(1)

    def forward(self, x: torch.Tensor, face_ids: torch.Tensor | None = None) -> torch.Tensor:
        if face_ids is None:
            raise ValueError("face_ids are required for SupportAwareLowRankSurfaceTexture")
        idx = self._texture_indices(x, face_ids)
        stats = self.surface_support_stats[idx].permute(0, 3, 1, 2)
        raw = self.decoder(torch.cat([x, stats], dim=1))
        weights = torch.softmax(raw[:, : self.rank], dim=1)
        conf = torch.sigmoid(raw[:, self.rank : self.rank + 1])
        conf = self.confidence_min + (self.confidence_max - self.confidence_min) * conf
        basis = torch.tanh(self.surface_basis(idx)).view(*idx.shape, self.rank, 3) * self.max_delta
        delta = torch.sum(weights.permute(0, 2, 3, 1).unsqueeze(-1) * basis, dim=3)
        delta = delta.permute(0, 3, 1, 2)
        gate = stats[:, 2:3]
        return delta * conf * gate


def _sample_patch(example: dict[str, Any], patch_size: int, rng: random.Random) -> TrainBatch:
    features = example["features"]
    face_ids = example.get("face_ids")
    parent = example["parent"]
    teacher = example["teacher"]
    gt = example.get("gt")
    debt_mask = example.get("residual_debt_mask")
    benefit_mask = example.get("teacher_benefit_mask")
    _, h, w = parent.shape
    if int(patch_size) <= 0 or h <= int(patch_size) or w <= int(patch_size):
        return TrainBatch(features, face_ids, parent, teacher, gt, debt_mask, benefit_mask)
    ph = pw = int(patch_size)
    y = rng.randint(0, h - ph)
    x = rng.randint(0, w - pw)
    gt_patch = None if gt is None else gt[:, y : y + ph, x : x + pw]
    face_patch = None if face_ids is None else face_ids[y : y + ph, x : x + pw]
    debt_patch = None if debt_mask is None else debt_mask[:, y : y + ph, x : x + pw]
    benefit_patch = None if benefit_mask is None else benefit_mask[:, y : y + ph, x : x + pw]
    return TrainBatch(
        features[:, y : y + ph, x : x + pw],
        face_patch,
        parent[:, y : y + ph, x : x + pw],
        teacher[:, y : y + ph, x : x + pw],
        gt_patch,
        debt_patch,
        benefit_patch,
    )


def _predict_delta_tiled(
    model: torch.nn.Module,
    features: torch.Tensor,
    face_ids: torch.Tensor | None = None,
    *,
    device: torch.device,
    tile: int,
    overlap: int,
) -> torch.Tensor:
    _, h, w = features.shape
    if int(tile) <= 0 or (h <= int(tile) and w <= int(tile)):
        face_batch = None if face_ids is None else face_ids.unsqueeze(0).to(device)
        return model(features.unsqueeze(0).to(device), face_batch).squeeze(0).detach().cpu()
    tile = int(tile)
    overlap = max(0, int(overlap))
    stride = max(1, tile - overlap)
    out = torch.zeros((3, h, w), dtype=torch.float32)
    weight = torch.zeros((1, h, w), dtype=torch.float32)
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    if xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = features[:, y : min(y + tile, h), x : min(x + tile, w)]
                face_patch = None if face_ids is None else face_ids[y : y + patch.shape[1], x : x + patch.shape[2]]
                face_batch = None if face_patch is None else face_patch.unsqueeze(0).to(device)
                pred = model(patch.unsqueeze(0).to(device), face_batch).squeeze(0).detach().cpu()
                out[:, y : y + pred.shape[1], x : x + pred.shape[2]] += pred
                weight[:, y : y + pred.shape[1], x : x + pred.shape[2]] += 1.0
    return out / torch.clamp(weight, min=1.0)


def evaluate_policy_val(
    model: torch.nn.Module,
    val_paths: list[Path],
    *,
    residual_key: str,
    face_lut: np.ndarray | None,
    alpha_grid: list[float],
    alpha_conditioned_residual: bool,
    policy_allow_noop_alpha: bool,
    policy_select_mode: str,
    policy_tail_fraction: float,
    policy_min_psnr_gain: float,
    policy_min_ssim_gain: float,
    policy_min_lpips_gain: float,
    policy_cvar_psnr_gain: float,
    policy_cvar_ssim_gain: float,
    policy_cvar_lpips_gain: float,
    device: torch.device,
    eval_tile: int,
    eval_overlap: int,
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    cache: dict[str, tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]] = {}
    model.eval()
    for path in tqdm(val_paths, desc="policy-val v184"):
        z = np.load(path)
        features = torch.from_numpy(_load_input_chw(z))
        face_ids = _load_face_ids_tensor(z, face_lut, max_side=-1)
        parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
        gt = np.clip(_to_chw(z["rgb_gt"])[:3], 0.0, 1.0).astype(np.float32)
        teacher_ref = None
        teacher_metrics: dict[str, float] = {}
        if str(residual_key) in z:
            teacher_ref = np.clip(parent + np.asarray(z[str(residual_key)], dtype=np.float32)[:3], 0.0, 1.0)
        shared_delta: np.ndarray | None = None
        if not bool(alpha_conditioned_residual):
            shared_delta = _predict_delta_tiled(
                model,
                features,
                face_ids=face_ids,
                device=device,
                tile=eval_tile,
                overlap=eval_overlap,
            ).numpy()
        delta_by_alpha: dict[float, np.ndarray] = {}
        p_psnr = _psnr(parent, gt)
        p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
        p_lpips = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
        if teacher_ref is not None:
            teacher_metrics["teacher_psnr"] = float(_psnr(teacher_ref, gt))
            teacher_metrics["teacher_ssim"] = float(image_ssim_chw(teacher_ref, gt, int(ssim_max_side)))
            teacher_metrics["teacher_psnr_gain"] = float(teacher_metrics["teacher_psnr"] - p_psnr)
            teacher_metrics["teacher_ssim_gain"] = float(teacher_metrics["teacher_ssim"] - p_ssim)
            if compute_lpips:
                t_lpips = image_lpips_chw(teacher_ref, gt, int(lpips_max_side), lpips_model)
                teacher_metrics["teacher_lpips"] = float(t_lpips)
                teacher_metrics["teacher_lpips_gain"] = float(p_lpips - t_lpips)
        for alpha in alpha_grid:
            alpha_f = float(alpha)
            if bool(alpha_conditioned_residual):
                if alpha_f == 0.0:
                    delta = np.zeros_like(parent, dtype=np.float32)
                else:
                    delta = _predict_delta_tiled(
                        model,
                        _append_alpha_channel_chw(features, alpha_f),
                        face_ids=face_ids,
                        device=device,
                        tile=eval_tile,
                        overlap=eval_overlap,
                    ).numpy()
                cand = np.clip(parent + delta, 0.0, 1.0)
            else:
                assert shared_delta is not None
                delta = shared_delta
                cand = np.clip(parent + alpha_f * delta, 0.0, 1.0)
            delta_by_alpha[alpha_f] = delta
            row = {
                "view": path.stem,
                "parent_psnr": float(p_psnr),
                "candidate_psnr": float(_psnr(cand, gt)),
                "parent_ssim": float(p_ssim),
                "candidate_ssim": float(image_ssim_chw(cand, gt, int(ssim_max_side))),
            }
            row.update(teacher_metrics)
            row["psnr_gain"] = row["candidate_psnr"] - row["parent_psnr"]
            row["ssim_gain"] = row["candidate_ssim"] - row["parent_ssim"]
            if "teacher_psnr_gain" in row:
                denom = max(float(row["teacher_psnr_gain"]), 1.0e-8)
                row["psnr_teacher_gap_recovery"] = float(row["psnr_gain"] / denom)
            if "teacher_ssim_gain" in row:
                denom = max(float(row["teacher_ssim_gain"]), 1.0e-8)
                row["ssim_teacher_gap_recovery"] = float(row["ssim_gain"] / denom)
            applied_delta = delta if bool(alpha_conditioned_residual) else alpha_f * delta
            row["changed_fraction"] = float(np.mean(np.any(np.abs(applied_delta) > (0.5 / 255.0), axis=0)))
            if compute_lpips:
                c_lpips = image_lpips_chw(cand, gt, int(lpips_max_side), lpips_model)
                row["parent_lpips"] = float(p_lpips)
                row["candidate_lpips"] = float(c_lpips)
                row["lpips_gain"] = float(p_lpips - c_lpips)
                if "teacher_lpips_gain" in row:
                    denom = max(float(row["teacher_lpips_gain"]), 1.0e-8)
                    row["lpips_teacher_gap_recovery"] = float(row["lpips_gain"] / denom)
            rows_by_alpha[alpha_f].append(row)
        cache[path.stem] = (delta_by_alpha, parent, gt)

    def _tail_mean(values: list[float], fraction: float) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        k = max(1, int(math.ceil(float(fraction) * int(arr.size))))
        k = min(k, int(arr.size))
        return float(np.mean(np.sort(arr)[:k]))

    def _mean_score(row: dict[str, Any]) -> float:
        return (
            float(row.get("psnr_gain", 0.0))
            + 20.0 * float(row.get("ssim_gain", 0.0))
            + 20.0 * float(row.get("lpips_gain", 0.0))
        )

    def _tail_score(row: dict[str, Any]) -> float:
        return (
            _mean_score(row)
            + 0.25 * float(row.get("psnr_cvar_gain", 0.0))
            + 10.0 * float(row.get("ssim_cvar_gain", 0.0))
            + 30.0 * float(row.get("lpips_cvar_gain", 0.0))
            + 0.10 * float(row.get("psnr_min_gain", 0.0))
            + 5.0 * float(row.get("ssim_min_gain", 0.0))
            + 15.0 * float(row.get("lpips_min_gain", 0.0))
        )

    select_mode = str(policy_select_mode)
    if select_mode not in {"mean", "tail_guard"}:
        raise ValueError(f"unknown policy_select_mode: {select_mode}")
    summaries: list[dict[str, Any]] = []
    for alpha, rows in rows_by_alpha.items():
        psnr_gain = [float(r["psnr_gain"]) for r in rows]
        ssim_gain = [float(r["ssim_gain"]) for r in rows]
        summary = {
            "alpha": float(alpha),
            "parent_psnr": float(np.mean([r["parent_psnr"] for r in rows])),
            "candidate_psnr": float(np.mean([r["candidate_psnr"] for r in rows])),
            "psnr_gain": float(np.mean(psnr_gain)),
            "parent_ssim": float(np.mean([r["parent_ssim"] for r in rows])),
            "candidate_ssim": float(np.mean([r["candidate_ssim"] for r in rows])),
            "ssim_gain": float(np.mean(ssim_gain)),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
            "mean_changed_fraction": float(np.mean([r["changed_fraction"] for r in rows])),
            "psnr_min_gain": float(np.min(psnr_gain)) if psnr_gain else 0.0,
            "ssim_min_gain": float(np.min(ssim_gain)) if ssim_gain else 0.0,
            "psnr_cvar_gain": _tail_mean(psnr_gain, float(policy_tail_fraction)),
            "ssim_cvar_gain": _tail_mean(ssim_gain, float(policy_tail_fraction)),
        }
        if compute_lpips:
            lpips_gain = [float(r["lpips_gain"]) for r in rows]
            summary["parent_lpips"] = float(np.mean([r["parent_lpips"] for r in rows]))
            summary["candidate_lpips"] = float(np.mean([r["candidate_lpips"] for r in rows]))
            summary["lpips_gain"] = float(np.mean(lpips_gain))
            summary["lpips_positive_view_fraction"] = float(np.mean(np.asarray(lpips_gain) > 0.0))
            summary["lpips_min_gain"] = float(np.min(lpips_gain)) if lpips_gain else 0.0
            summary["lpips_cvar_gain"] = _tail_mean(lpips_gain, float(policy_tail_fraction))
        teacher_rows = [r for r in rows if "teacher_psnr" in r]
        if teacher_rows:
            summary["teacher_psnr"] = float(np.mean([r["teacher_psnr"] for r in teacher_rows]))
            summary["teacher_ssim"] = float(np.mean([r["teacher_ssim"] for r in teacher_rows]))
            summary["teacher_psnr_gain"] = float(np.mean([r["teacher_psnr_gain"] for r in teacher_rows]))
            summary["teacher_ssim_gain"] = float(np.mean([r["teacher_ssim_gain"] for r in teacher_rows]))
            summary["psnr_teacher_gap_recovery"] = float(
                np.mean([r.get("psnr_teacher_gap_recovery", 0.0) for r in teacher_rows])
            )
            summary["ssim_teacher_gap_recovery"] = float(
                np.mean([r.get("ssim_teacher_gap_recovery", 0.0) for r in teacher_rows])
            )
            if compute_lpips and "teacher_lpips" in teacher_rows[0]:
                summary["teacher_lpips"] = float(np.mean([r["teacher_lpips"] for r in teacher_rows]))
                summary["teacher_lpips_gain"] = float(np.mean([r["teacher_lpips_gain"] for r in teacher_rows]))
                summary["lpips_teacher_gap_recovery"] = float(
                    np.mean([r.get("lpips_teacher_gap_recovery", 0.0) for r in teacher_rows])
                )
        summary["mean_score"] = _mean_score(summary)
        summary["tail_score"] = _tail_score(summary)
        summaries.append(summary)
    eligible_summaries = summaries
    zero_alpha_excluded = False
    if not bool(policy_allow_noop_alpha):
        nonzero = [
            row
            for row in summaries
            if abs(float(row.get("alpha", 0.0))) > 1.0e-12
            and float(row.get("mean_changed_fraction", 0.0)) > 0.0
        ]
        if nonzero:
            eligible_summaries = nonzero
            zero_alpha_excluded = True
    best_mean = max(eligible_summaries, key=_mean_score)
    best_tail = max(eligible_summaries, key=_tail_score)
    best = best_tail if select_mode == "tail_guard" else best_mean
    pass_rows = [
        row
        for row in eligible_summaries
        if float(row.get("psnr_gain", 0.0)) > 0.0
        and float(row.get("ssim_gain", 0.0)) > 0.0
        and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
    ]
    if select_mode == "tail_guard":
        pass_rows = [
            row
            for row in pass_rows
            if float(row.get("psnr_min_gain", 0.0)) >= float(policy_min_psnr_gain)
            and float(row.get("ssim_min_gain", 0.0)) >= float(policy_min_ssim_gain)
            and (not compute_lpips or float(row.get("lpips_min_gain", 0.0)) >= float(policy_min_lpips_gain))
            and float(row.get("psnr_cvar_gain", 0.0)) >= float(policy_cvar_psnr_gain)
            and float(row.get("ssim_cvar_gain", 0.0)) >= float(policy_cvar_ssim_gain)
            and (not compute_lpips or float(row.get("lpips_cvar_gain", 0.0)) >= float(policy_cvar_lpips_gain))
        ]
    best_all_axis = None
    if pass_rows:
        best_all_axis = max(
            pass_rows,
            key=_tail_score if select_mode == "tail_guard" else _mean_score,
        )
    if output_dir is not None:
        alpha = float((best_all_axis or best)["alpha"])
        (output_dir / "renders").mkdir(parents=True, exist_ok=True)
        (output_dir / "parent").mkdir(parents=True, exist_ok=True)
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)
        for stem, (delta_by_alpha, parent, gt) in cache.items():
            delta = delta_by_alpha[alpha]
            applied_delta = delta if bool(alpha_conditioned_residual) else alpha * delta
            save_image_chw(output_dir / "renders" / f"{stem}.png", np.clip(parent + applied_delta, 0.0, 1.0))
            save_image_chw(output_dir / "parent" / f"{stem}.png", parent)
            save_image_chw(output_dir / "gt" / f"{stem}.png", gt)
    return {
        "best": best,
        "best_mean": best_mean,
        "best_tail": best_tail,
        "best_all_axis": best_all_axis,
        "selection": {
            "mode": select_mode,
            "tail_fraction": float(policy_tail_fraction),
            "min_psnr_gain": float(policy_min_psnr_gain),
            "min_ssim_gain": float(policy_min_ssim_gain),
            "min_lpips_gain": float(policy_min_lpips_gain),
            "cvar_psnr_gain": float(policy_cvar_psnr_gain),
            "cvar_ssim_gain": float(policy_cvar_ssim_gain),
            "cvar_lpips_gain": float(policy_cvar_lpips_gain),
            "allow_noop_alpha": bool(policy_allow_noop_alpha),
            "zero_alpha_excluded_from_best": bool(zero_alpha_excluded),
            "eligible_alpha_count": int(len(eligible_summaries)),
        },
        "rows": summaries,
        "per_view_by_alpha": {str(k): v for k, v in rows_by_alpha.items()},
    }


def apply_target(
    model: torch.nn.Module,
    target_evidence_dir: Path,
    *,
    face_lut: np.ndarray | None,
    scene_name: str,
    method_name: str,
    alpha: float,
    alpha_conditioned_residual: bool,
    device: torch.device,
    eval_tile: int,
    eval_overlap: int,
    output_dir: Path,
) -> dict[str, Any]:
    no_gt = verify_target_no_gt(target_evidence_dir)
    if not bool(no_gt.get("passed")):
        raise RuntimeError(f"target no-GT verification failed: {no_gt}")
    target_paths = evidence_views(target_evidence_dir)
    safe_scene = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(scene_name).strip())
    if not safe_scene:
        safe_scene = "scene"
    out_root = output_dir / f"{safe_scene}_exact_target_apply"
    render_dir = out_root / "test" / method_name / "renders"
    parent_dir = out_root / "test" / method_name / "parent"
    render_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    model.eval()
    for path in tqdm(target_paths, desc="target v184 no-GT apply"):
        z = np.load(path)
        features = torch.from_numpy(_load_input_chw(z))
        face_ids = _load_face_ids_tensor(z, face_lut, max_side=-1)
        parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
        if bool(alpha_conditioned_residual) and float(alpha) == 0.0:
            delta = np.zeros_like(parent, dtype=np.float32)
        else:
            model_features = (
                _append_alpha_channel_chw(features, float(alpha)) if bool(alpha_conditioned_residual) else features
            )
            delta = _predict_delta_tiled(
                model,
                model_features,
                face_ids=face_ids,
                device=device,
                tile=eval_tile,
                overlap=eval_overlap,
            ).numpy()
        applied_delta = delta if bool(alpha_conditioned_residual) else float(alpha) * delta
        cand = np.clip(parent + applied_delta, 0.0, 1.0)
        save_image_chw(render_dir / f"{path.stem}.png", cand)
        save_image_chw(parent_dir / f"{path.stem}.png", parent)
        changed = np.any(np.abs(applied_delta) > (0.5 / 255.0), axis=0)
        row: dict[str, Any] = {
            "view": path.stem,
            "changed_fraction": float(np.mean(changed)),
        }
        valid_mask = features[15].numpy() > 0.5 if features.shape[0] > 15 else np.ones(changed.shape, dtype=bool)
        valid_count = max(1, int(np.sum(valid_mask)))
        if face_ids is not None:
            known = face_ids.numpy() > 0
            row["known_face_fraction"] = float(np.sum(known & valid_mask) / valid_count)
        if face_ids is not None and hasattr(model, "support_mask"):
            with torch.no_grad():
                support = (
                    model.support_mask(features.unsqueeze(0).to(device), face_ids.unsqueeze(0).to(device))
                    .squeeze(0)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    > 0.5
                )
            active = support & valid_mask
            inactive = (~support) & valid_mask
            row["active_support_fraction"] = float(np.sum(active) / valid_count)
            row["active_support_changed_fraction"] = (
                float(np.sum(changed & active) / max(1, int(np.sum(active)))) if np.any(active) else 0.0
            )
            row["inactive_support_changed_fraction"] = (
                float(np.sum(changed & inactive) / max(1, int(np.sum(inactive)))) if np.any(inactive) else 0.0
            )
        rows.append(row)
    summary: dict[str, Any] = {
        "method_name": str(method_name),
        "alpha": float(alpha),
        "no_gt_verify": no_gt,
        "scene_name": safe_scene,
        "output_model": str(out_root),
        "render_dir": str(render_dir),
        "parent_dir": str(parent_dir),
        "view_count": int(len(rows)),
        "mean_changed_fraction": float(np.mean([r.get("changed_fraction", 0.0) for r in rows])) if rows else 0.0,
        "mean_known_face_fraction": float(np.mean([r.get("known_face_fraction", 0.0) for r in rows])) if rows else 0.0,
        "mean_active_support_fraction": float(np.mean([r.get("active_support_fraction", 0.0) for r in rows])) if rows else 0.0,
        "mean_active_support_changed_fraction": float(
            np.mean([r.get("active_support_changed_fraction", 0.0) for r in rows])
        )
        if rows
        else 0.0,
        "mean_inactive_support_changed_fraction": float(
            np.mean([r.get("inactive_support_changed_fraction", 0.0) for r in rows])
        )
        if rows
        else 0.0,
        "per_view": rows,
    }
    return summary


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    selection = payload["policy_val"].get("selection", {})
    target = payload.get("target_apply") or {}
    leakage = payload.get("gt_usage_audit", {})
    audit = payload.get("audit_notes", {})
    phasej_gate = payload.get("phasej_flowers_gate", {})
    model = payload.get("model", {})
    lines = [
        f"# {payload.get('artifact_prefix', 'surface_conditioned_residual')} Audit",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- policy-val gate scope: `{audit.get('policy_val_gate_scope')}`",
        f"- Phase-J exact gate enforced by this script: `{audit.get('phasej_exact_gate_enforced_by_script')}`",
        f"- Phase-J flowers gate: `{phasej_gate.get('psnr')} / {phasej_gate.get('ssim')} / {phasej_gate.get('lpips')}`",
        f"- policy-val candidate numerically above Phase-J flowers reference: `{phasej_gate.get('policy_val_numeric_above_reference')}`",
        f"- alpha contract: `{audit.get('alpha_contract')}`",
        f"- alpha contract warning: `{audit.get('alpha_contract_warning')}`",
        f"- teacher-benefit mask mode: `{model.get('teacher_benefit_mask_mode')}`",
        f"- train teacher-benefit active fraction: `{model.get('train_teacher_benefit_active_fraction')}`",
        f"- target exact run: `{bool(target)}`",
        f"- target no-GT verifier: `{target.get('no_gt_verify', {}).get('passed')}`",
        f"- uses train-fit GT: `{leakage.get('uses_train_fit_gt')}`",
        f"- uses train-fit GT for teacher-benefit mask: `{leakage.get('uses_train_fit_gt_for_teacher_benefit_mask')}`",
        f"- uses policy-val GT: `{leakage.get('uses_policy_val_gt')}`",
        f"- uses target/test GT during apply: `{leakage.get('uses_target_or_test_gt_during_apply')}`",
        f"- uses target-view geometry for capacity: `{leakage.get('uses_target_view_geometry_for_capacity')}`",
        "",
        "## Policy-Val",
        "",
        f"- selection mode: `{selection.get('mode', 'mean')}`",
        f"- tail fraction: `{selection.get('tail_fraction', 0.2)}`",
        f"- allow no-op alpha as best: `{selection.get('allow_noop_alpha')}`",
        f"- zero alpha excluded from best: `{selection.get('zero_alpha_excluded_from_best')}`",
        "",
        "| alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR min | SSIM min | LPIPS min | PSNR CVaR | SSIM CVaR | LPIPS CVaR | changed fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.6f} | {best.get('psnr_gain', 0.0):+.9f} | "
            f"{best.get('ssim_gain', 0.0):+.9f} | {best.get('lpips_gain', 0.0):+.9f} | "
            f"{best.get('psnr_min_gain', 0.0):+.9f} | {best.get('ssim_min_gain', 0.0):+.9f} | "
            f"{best.get('lpips_min_gain', 0.0):+.9f} | {best.get('psnr_cvar_gain', 0.0):+.9f} | "
            f"{best.get('ssim_cvar_gain', 0.0):+.9f} | {best.get('lpips_cvar_gain', 0.0):+.9f} | "
            f"{best.get('mean_changed_fraction', 0.0):.6f} |"
        ),
        "",
        f"- best all-axis row: `{best_all}`",
        "",
        "## Target",
        "",
        f"- alpha: `{target.get('alpha')}`",
        f"- changed fraction: `{target.get('mean_changed_fraction')}`",
        f"- render dir: `{target.get('render_dir')}`",
        "- official metrics: run `ecsr_populate_eval_gt_from_target_evidence.py` and `evaluate_render_split_metrics.py` after no-GT apply.",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- report: `{path}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a surface-conditioned residual model for Phase-J teacher baking.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_FIT_EVIDENCE)
    parser.add_argument("--target_evidence_dir", default="")
    parser.add_argument("--target_eval_evidence_dir", default="")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument(
        "--residual_l1_key",
        default="",
        help=(
            "Optional scalar residual field for surface face/bin selection. "
            "Defaults to the residual_rgb_key-derived *_l1 field, or teacher_residual_l1."
        ),
    )
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--train_max_side", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--model_type",
        choices=["unet", "surface_texture_mlp", "surface_texture_unet", "lowrank_surface_texture"],
        default="unet",
    )
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--max_delta", type=float, default=0.20)
    parser.add_argument(
        "--confidence_mode",
        choices=["none", "sigmoid"],
        default="none",
        help="Optional learned residual confidence head. sigmoid predicts per-pixel residual strength from surface/view features.",
    )
    parser.add_argument("--confidence_bias", type=float, default=2.0)
    parser.add_argument("--confidence_min", type=float, default=0.0)
    parser.add_argument("--confidence_max", type=float, default=1.0)
    parser.add_argument("--face_embedding_dim", type=int, default=0)
    parser.add_argument(
        "--face_embedding_max_unique",
        type=int,
        default=0,
        help="Optional deterministic cap for train-fit face ids. 0 keeps all train-fit faces.",
    )
    parser.add_argument("--surface_texture_size", type=int, default=8)
    parser.add_argument("--surface_feature_dim", type=int, default=8)
    parser.add_argument("--surface_decoder_hidden", type=int, default=64)
    parser.add_argument("--surface_decoder_layers", type=int, default=3)
    parser.add_argument("--surface_face_max_unique", type=int, default=8192)
    parser.add_argument("--surface_face_min_alpha", type=float, default=0.03)
    parser.add_argument("--surface_face_min_residual_l1", type=float, default=0.0)
    parser.add_argument(
        "--enable_surface_support_gate",
        action="store_true",
        help="For surface neural texture models, zero UV rows with insufficient train-fit residual support.",
    )
    parser.add_argument(
        "--surface_support_gate_floor",
        type=float,
        default=0.0,
        help=(
            "When the support gate is enabled, keep this residual-capacity floor for known face/UV rows "
            "that are visible but below the train-fit support threshold. Default 0 preserves the hard gate."
        ),
    )
    parser.add_argument(
        "--surface_support_unknown_gate_floor",
        type=float,
        default=0.0,
        help=(
            "When the support gate is enabled, keep this residual-capacity floor for valid target pixels "
            "whose face is outside the learned LUT. This enables a bounded dense decoder fallback without target GT."
        ),
    )
    parser.add_argument(
        "--surface_target_visible_evidence_dir",
        default="",
        help="Optional no-GT target evidence used only to prioritize target-visible faces in the surface capacity budget.",
    )
    parser.add_argument(
        "--enable_surface_evidence_texture",
        action="store_true",
        help=(
            "For surface_texture_unet, append a fixed train-fit teacher residual evidence texture "
            "(mean residual, variance, support, and source camera direction) indexed by face/UV bin."
        ),
    )
    parser.add_argument(
        "--surface_evidence_residual_rgb_key",
        default="",
        help="Residual RGB key used to build the fixed surface evidence texture. Defaults to --residual_rgb_key.",
    )
    parser.add_argument(
        "--surface_evidence_min_bin_support",
        type=int,
        default=1,
        help="Minimum train-fit samples per face/UV bin for the evidence texture active channel.",
    )
    parser.add_argument(
        "--surface_evidence_residual_prior_weight",
        type=float,
        default=0.0,
        help="Directly inject this weight times the view-gated mean train-fit residual evidence into the predicted delta.",
    )
    parser.add_argument(
        "--surface_evidence_view_gate_power",
        type=float,
        default=0.0,
        help="Power for source-camera agreement gating of residual evidence. 0 disables view gating.",
    )
    parser.add_argument(
        "--enable_surface_source_evidence_bank",
        action="store_true",
        help=(
            "For surface_texture_unet, replace the mean evidence texture with a top-K per-face/UV source bank "
            "containing source residuals and source camera directions."
        ),
    )
    parser.add_argument(
        "--surface_source_evidence_top_k",
        type=int,
        default=4,
        help="Top-K source residual observations retained per face/UV bin when --enable_surface_source_evidence_bank is active.",
    )
    parser.add_argument("--lowrank_rank", type=int, default=4)
    parser.add_argument("--lowrank_min_bin_support", type=int, default=16)
    parser.add_argument("--lowrank_basis_init_std", type=float, default=0.01)
    parser.add_argument("--teacher_l1_weight", type=float, default=1.0)
    parser.add_argument("--teacher_ssim_weight", type=float, default=0.20)
    parser.add_argument("--teacher_lpips_weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher_lpips_noharm_weight",
        type=float,
        default=0.0,
        help=(
            "Perceptual safety loss: penalize train patches where the adapted image is worse than the "
            "parent render under LPIPS to the teacher target."
        ),
    )
    parser.add_argument("--teacher_lpips_noharm_margin", type=float, default=0.0)
    parser.add_argument("--teacher_grad_weight", type=float, default=0.0)
    parser.add_argument("--teacher_highfreq_weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher_residual_cosine_weight",
        type=float,
        default=0.0,
        help="Directly align predicted residual direction with the Phase-J teacher-parent residual.",
    )
    parser.add_argument(
        "--teacher_residual_energy_weight",
        type=float,
        default=0.0,
        help="Match the RMS energy of the predicted residual to the Phase-J teacher-parent residual.",
    )
    parser.add_argument(
        "--teacher_residual_projection_min_l1",
        type=float,
        default=0.5 / 255.0,
        help="Minimum per-pixel teacher residual L1 used by the residual projection losses.",
    )
    parser.add_argument("--gt_l1_weight", type=float, default=0.10)
    parser.add_argument("--gt_ssim_weight", type=float, default=0.0)
    parser.add_argument("--gt_lpips_weight", type=float, default=0.0)
    parser.add_argument(
        "--gt_lpips_noharm_weight",
        type=float,
        default=0.0,
        help=(
            "Perceptual safety loss: penalize train patches where the adapted image is worse than the "
            "parent render under LPIPS to the train-fit GT target."
        ),
    )
    parser.add_argument("--gt_lpips_noharm_margin", type=float, default=0.0)
    parser.add_argument("--gt_grad_weight", type=float, default=0.0)
    parser.add_argument("--gt_highfreq_weight", type=float, default=0.0)
    parser.add_argument(
        "--support_loss_mask",
        choices=["none", "active"],
        default="none",
        help=(
            "When active, normalize train losses over bins writable by the surface support gate. "
            "This keeps hard no-op pixels from diluting teacher/GT supervision."
        ),
    )
    parser.add_argument(
        "--support_patch_resample_attempts",
        type=int,
        default=1,
        help="If support_loss_mask=active, resample train patches up to this many times to find writable support.",
    )
    parser.add_argument(
        "--support_patch_min_active_fraction",
        type=float,
        default=0.0,
        help="Minimum active support fraction accepted by support_patch_resample_attempts.",
    )
    parser.add_argument("--lpips_loss_max_side", type=int, default=128)
    parser.add_argument("--grad_loss_max_side", type=int, default=256)
    parser.add_argument("--highfreq_loss_max_side", type=int, default=256)
    parser.add_argument("--highfreq_loss_levels", type=int, default=3)
    parser.add_argument("--delta_l1_weight", type=float, default=1.0e-4)
    parser.add_argument(
        "--residual_debt_mask",
        action="store_true",
        help=(
            "Use train-fit-only residual-debt masking: learn corrections only where parent-vs-GT/teacher "
            "error is scene-adaptively high, and pull other pixels back to no-op."
        ),
    )
    parser.add_argument("--residual_debt_quantile", type=float, default=0.70)
    parser.add_argument("--residual_debt_min_l1", type=float, default=1.0 / 255.0)
    parser.add_argument("--residual_debt_dilate", type=int, default=1)
    parser.add_argument("--residual_debt_noop_weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher_benefit_mask_mode",
        choices=["off", "teacher_better", "positive_gain", "better_and_positive_gain"],
        default="off",
        help=(
            "Train-fit-only Phase-J benefit mask. When enabled, teacher supervision is kept only where "
            "teacher evidence says Phase-J improves over the parent; elsewhere the target is parent/no-op."
        ),
    )
    parser.add_argument("--teacher_benefit_min_gain_l1", type=float, default=0.0)
    parser.add_argument("--teacher_benefit_dilate", type=int, default=1)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument(
        "--policy_select_mode",
        choices=["mean", "tail_guard"],
        default="mean",
        help="mean preserves the legacy average-gain selector; tail_guard also enforces min/CVaR gain thresholds.",
    )
    parser.add_argument("--policy_tail_fraction", type=float, default=0.20)
    parser.add_argument("--policy_min_psnr_gain", type=float, default=-1.0e9)
    parser.add_argument("--policy_min_ssim_gain", type=float, default=-1.0e9)
    parser.add_argument("--policy_min_lpips_gain", type=float, default=-1.0e9)
    parser.add_argument("--policy_cvar_psnr_gain", type=float, default=-1.0e9)
    parser.add_argument("--policy_cvar_ssim_gain", type=float, default=-1.0e9)
    parser.add_argument("--policy_cvar_lpips_gain", type=float, default=-1.0e9)
    parser.add_argument(
        "--policy_allow_noop_alpha",
        action="store_true",
        help="Allow alpha=0/no-op to be selected as policy best. v169 runs should leave this disabled.",
    )
    parser.add_argument(
        "--alpha_conditioned_residual",
        action="store_true",
        help=(
            "Append the selected alpha as an input channel and train the model to emit the final delta for that alpha. "
            "This avoids selecting an untrained post-hoc alpha multiplier at policy-val/apply time."
        ),
    )
    parser.add_argument("--target_alpha", type=float, default=None)
    parser.add_argument("--phasej_flowers_psnr", type=float, default=20.304358)
    parser.add_argument("--phasej_flowers_ssim", type=float, default=0.557770)
    parser.add_argument("--phasej_flowers_lpips", type=float, default=0.329222)
    parser.add_argument("--method_name", default="ours_26000_v184_surface_conditioned_unet_flowers")
    parser.add_argument("--scene_name", default="flowers")
    parser.add_argument("--eval_tile", type=int, default=512)
    parser.add_argument("--eval_overlap", type=int, default=32)
    parser.add_argument("--ssim_max_side", type=int, default=-1)
    parser.add_argument("--lpips_max_side", type=int, default=256)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--skip_policy_val_renders", action="store_true")
    parser.add_argument("--output_dir", default="/dev/shm/peilincai_spcarnet_v184_surface_conditioned_unet")
    parser.add_argument(
        "--artifact_prefix",
        default="v184_surface_conditioned_unet",
        help="Filename prefix for checkpoint/report artifacts. Use a run-specific prefix to avoid stale v184 names.",
    )
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=184)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(args.artifact_prefix).strip()
    )
    if not artifact_prefix:
        artifact_prefix = "surface_conditioned_residual"
    checkpoint_path = output_dir / f"{artifact_prefix}.pt"
    report_json_path = output_dir / f"{artifact_prefix}_report.json"
    report_md_path = output_dir / f"{artifact_prefix}_report.md"
    wandb_run = None
    if bool(args.enable_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or output_dir.name),
                config=vars(args),
                dir=str(output_dir),
            )
        except Exception as exc:
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)
            wandb_run = None
    paths = evidence_views(Path(args.fit_evidence_dir))
    if not paths:
        raise FileNotFoundError(args.fit_evidence_dir)
    fit_paths, val_paths = _policy_split(paths, int(args.policy_val_stride))
    face_lut = None
    surface_support_stats = None
    surface_evidence_stats = None
    surface_evidence_summary = None
    surface_model_types = {"surface_texture_mlp", "surface_texture_unet", "lowrank_surface_texture"}
    residual_l1_key = str(args.residual_l1_key).strip()
    if not residual_l1_key:
        residual_l1_key = (
            str(args.residual_rgb_key).replace("_rgb", "_l1")
            if str(args.residual_rgb_key).endswith("_rgb")
            else "teacher_residual_l1"
        )
    if str(args.model_type) in surface_model_types and int(args.face_embedding_dim) > 0:
        raise ValueError("--face_embedding_dim is only supported for --model_type unet")
    if str(args.model_type) in surface_model_types:
        priority_face_counts = None
        target_visible_summary = None
        if str(args.surface_target_visible_evidence_dir):
            target_visible_dir = Path(str(args.surface_target_visible_evidence_dir))
            target_visible_no_gt = verify_target_no_gt(target_visible_dir)
            if not bool(target_visible_no_gt.get("passed")):
                raise RuntimeError(f"target-visible evidence must be no-GT: {target_visible_no_gt}")
            priority_face_counts, target_visible_summary = _collect_visible_face_counts(
                target_visible_dir,
                min_alpha=float(args.surface_face_min_alpha),
            )
            target_visible_summary["no_gt_verify"] = target_visible_no_gt
        face_lut, face_lut_summary = _collect_residual_top_face_lut(
            fit_paths,
            residual_l1_key=residual_l1_key,
            residual_rgb_key=str(args.residual_rgb_key),
            max_unique=int(args.surface_face_max_unique),
            min_alpha=float(args.surface_face_min_alpha),
            min_residual_l1=float(args.surface_face_min_residual_l1),
            priority_face_counts=priority_face_counts,
        )
        if face_lut.size == 0:
            raise RuntimeError("surface texture model requested but no residual top faces were collected")
        texture_rows = int(face_lut.size) * int(args.surface_texture_size) * int(args.surface_texture_size) + 1
        support_summary = None
        if str(args.model_type) == "lowrank_surface_texture" or bool(args.enable_surface_support_gate):
            surface_support_stats, support_summary = _collect_surface_texture_support_stats(
                fit_paths,
                face_lut=face_lut,
                residual_l1_key=residual_l1_key,
                residual_rgb_key=str(args.residual_rgb_key),
                texture_size=int(args.surface_texture_size),
                min_alpha=float(args.surface_face_min_alpha),
                min_residual_l1=float(args.surface_face_min_residual_l1),
                min_bin_support=int(args.lowrank_min_bin_support),
            )
        if bool(args.enable_surface_evidence_texture) and bool(args.enable_surface_source_evidence_bank):
            raise ValueError("use either --enable_surface_evidence_texture or --enable_surface_source_evidence_bank, not both")
        if bool(args.enable_surface_evidence_texture):
            if str(args.model_type) != "surface_texture_unet":
                raise ValueError("--enable_surface_evidence_texture currently requires --model_type surface_texture_unet")
            evidence_rgb_key = str(args.surface_evidence_residual_rgb_key).strip() or str(args.residual_rgb_key)
            surface_evidence_stats, surface_evidence_summary = _collect_surface_evidence_texture_stats(
                fit_paths,
                face_lut=face_lut,
                residual_rgb_key=evidence_rgb_key,
                residual_l1_key=residual_l1_key,
                texture_size=int(args.surface_texture_size),
                min_alpha=float(args.surface_face_min_alpha),
                min_residual_l1=float(args.surface_face_min_residual_l1),
                min_bin_support=int(args.surface_evidence_min_bin_support),
            )
        if bool(args.enable_surface_source_evidence_bank):
            if str(args.model_type) != "surface_texture_unet":
                raise ValueError("--enable_surface_source_evidence_bank currently requires --model_type surface_texture_unet")
            evidence_rgb_key = str(args.surface_evidence_residual_rgb_key).strip() or str(args.residual_rgb_key)
            surface_evidence_stats, surface_evidence_summary = _collect_surface_source_evidence_bank_stats(
                fit_paths,
                face_lut=face_lut,
                residual_rgb_key=evidence_rgb_key,
                residual_l1_key=residual_l1_key,
                texture_size=int(args.surface_texture_size),
                top_k=int(args.surface_source_evidence_top_k),
                min_alpha=float(args.surface_face_min_alpha),
                min_residual_l1=float(args.surface_face_min_residual_l1),
                min_bin_support=int(args.surface_evidence_min_bin_support),
            )
        _write_json(
            output_dir / "surface_texture_lut_summary.json",
            {
                **face_lut_summary,
                "texture_size": int(args.surface_texture_size),
                "feature_dim": int(args.surface_feature_dim),
                "embedding_rows": int(texture_rows),
                "estimated_parameter_count": int(
                    texture_rows
                    * (int(args.lowrank_rank) * 3 if str(args.model_type) == "lowrank_surface_texture" else int(args.surface_feature_dim))
                ),
                "lowrank_support_summary": support_summary,
                "surface_evidence_summary": surface_evidence_summary,
                "target_visible_priority_summary": target_visible_summary,
            },
        )
    elif int(args.face_embedding_dim) > 0:
        face_lut = _collect_train_face_lut(fit_paths, int(args.face_embedding_max_unique))
        _write_json(
            output_dir / "face_lut_summary.json",
            {
                "schema": "spcarnet_train_fit_face_lut_v1",
                "fit_view_count": int(len(fit_paths)),
                "unique_train_fit_faces": int(face_lut.size),
                "embedding_rows": int(face_lut.size + 1),
                "embedding_dim": int(args.face_embedding_dim),
                "target_or_test_gt_used": False,
            },
        )
    train_examples = [
        _load_example(
            p,
            str(args.residual_rgb_key),
            int(args.train_max_side),
            include_gt=True,
            face_lut=face_lut,
            residual_debt_mask=bool(args.residual_debt_mask),
            residual_debt_quantile=float(args.residual_debt_quantile),
            residual_debt_min_l1=float(args.residual_debt_min_l1),
            residual_debt_dilate=int(args.residual_debt_dilate),
            teacher_benefit_mask_mode=str(args.teacher_benefit_mask_mode),
            teacher_benefit_min_gain_l1=float(args.teacher_benefit_min_gain_l1),
            teacher_benefit_dilate=int(args.teacher_benefit_dilate),
        )
        for p in tqdm(fit_paths, desc="preload train-fit")
    ]
    alpha_grid = _parse_float_grid(str(args.alpha_grid))
    train_alpha_grid = [float(a) for a in alpha_grid if float(a) > 0.0]
    if not train_alpha_grid:
        train_alpha_grid = [1.0]
    in_ch = int(train_examples[0]["features"].shape[0]) + (1 if bool(args.alpha_conditioned_residual) else 0)
    if str(args.model_type) == "surface_texture_mlp":
        if face_lut is None or face_lut.size == 0:
            raise RuntimeError("surface texture model requested but no train-fit face ids were collected")
        model = SurfaceTextureResidualMLP(
            in_ch,
            int(args.surface_decoder_hidden),
            float(args.max_delta),
            int(face_lut.size),
            int(args.surface_texture_size),
            int(args.surface_feature_dim),
            int(args.surface_decoder_layers),
            confidence_mode=str(args.confidence_mode),
            confidence_bias=float(args.confidence_bias),
            confidence_min=float(args.confidence_min),
            confidence_max=float(args.confidence_max),
        ).to(device)
    elif str(args.model_type) == "surface_texture_unet":
        if face_lut is None or face_lut.size == 0:
            raise RuntimeError("surface texture U-Net requested but no train-fit face ids were collected")
        model = SurfaceTextureConditionedUNet(
            in_ch,
            int(args.base_channels),
            float(args.max_delta),
            int(face_lut.size),
            int(args.surface_texture_size),
            int(args.surface_feature_dim),
            surface_support_stats if bool(args.enable_surface_support_gate) else None,
            confidence_mode=str(args.confidence_mode),
            confidence_bias=float(args.confidence_bias),
            confidence_min=float(args.confidence_min),
            confidence_max=float(args.confidence_max),
            support_gate_floor=float(args.surface_support_gate_floor),
            support_unknown_gate_floor=float(args.surface_support_unknown_gate_floor),
            evidence_stats=surface_evidence_stats
            if (bool(args.enable_surface_evidence_texture) or bool(args.enable_surface_source_evidence_bank))
            else None,
            evidence_residual_prior_weight=float(args.surface_evidence_residual_prior_weight),
            evidence_view_gate_power=float(args.surface_evidence_view_gate_power),
            evidence_source_bank_top_k=(
                int(args.surface_source_evidence_top_k) if bool(args.enable_surface_source_evidence_bank) else 0
            ),
            evidence_source_bank_channels_per_source=8,
        ).to(device)
    elif str(args.model_type) == "lowrank_surface_texture":
        if face_lut is None or face_lut.size == 0 or surface_support_stats is None:
            raise RuntimeError("low-rank surface texture requested but no support stats were collected")
        model = SupportAwareLowRankSurfaceTexture(
            in_ch,
            int(args.surface_decoder_hidden),
            float(args.max_delta),
            int(face_lut.size),
            int(args.surface_texture_size),
            int(args.lowrank_rank),
            int(args.surface_decoder_layers),
            surface_support_stats,
            basis_init_std=float(args.lowrank_basis_init_std),
            confidence_bias=float(args.confidence_bias),
            confidence_min=float(args.confidence_min),
            confidence_max=float(args.confidence_max),
        ).to(device)
    elif int(args.face_embedding_dim) > 0:
        if face_lut is None or face_lut.size == 0:
            raise RuntimeError("face embedding requested but no train-fit face ids were collected")
        model = SurfaceConditionedFaceEmbeddingUNet(
            in_ch,
            int(args.base_channels),
            float(args.max_delta),
            int(face_lut.size + 1),
            int(args.face_embedding_dim),
            confidence_mode=str(args.confidence_mode),
            confidence_bias=float(args.confidence_bias),
            confidence_min=float(args.confidence_min),
            confidence_max=float(args.confidence_max),
        ).to(device)
    else:
        model = SurfaceConditionedResidualUNet(
            in_ch,
            int(args.base_channels),
            float(args.max_delta),
            confidence_mode=str(args.confidence_mode),
            confidence_bias=float(args.confidence_bias),
            confidence_min=float(args.confidence_min),
            confidence_max=float(args.confidence_max),
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-5)
    train_lpips_model = None
    if (
        float(args.teacher_lpips_weight) > 0.0
        or float(args.gt_lpips_weight) > 0.0
        or float(args.teacher_lpips_noharm_weight) > 0.0
        or float(args.gt_lpips_noharm_weight) > 0.0
    ):
        train_lpips_model = build_lpips_model().to(device).eval()
    rng = random.Random(int(args.seed))
    for step in tqdm(range(1, int(args.steps) + 1), desc=f"train {artifact_prefix}"):
        model.train()
        loss_mask = None
        loss_mask_active_fraction = torch.ones((), device=device)
        debt_mask = None
        debt_mask_active_fraction = torch.ones((), device=device)
        benefit_mask = None
        benefit_mask_active_fraction = torch.ones((), device=device)
        support_attempts = max(1, int(args.support_patch_resample_attempts))
        for attempt in range(support_attempts):
            ex = rng.choice(train_examples)
            batch = _sample_patch(ex, int(args.patch_size), rng)
            train_alpha = float(rng.choice(train_alpha_grid)) if bool(args.alpha_conditioned_residual) else 1.0
            feat_base = batch.features
            if bool(args.alpha_conditioned_residual):
                feat_base = _append_alpha_channel_chw(feat_base, train_alpha)
            feat = feat_base.unsqueeze(0).to(device)
            face_batch = None if batch.face_ids is None else batch.face_ids.unsqueeze(0).to(device)
            parent = batch.parent.unsqueeze(0).to(device)
            raw_teacher = batch.teacher.unsqueeze(0).to(device)
            if bool(args.residual_debt_mask) and batch.residual_debt_mask is not None:
                debt_mask = batch.residual_debt_mask.unsqueeze(0).to(device=device, dtype=parent.dtype)
                debt_mask_active_fraction = torch.mean((debt_mask > 0.0).float())
            else:
                debt_mask = None
                debt_mask_active_fraction = torch.ones((), device=device)
            if batch.teacher_benefit_mask is not None:
                benefit_mask = batch.teacher_benefit_mask.unsqueeze(0).to(device=device, dtype=parent.dtype)
                benefit_mask_active_fraction = torch.mean((benefit_mask > 0.0).float())
            else:
                benefit_mask = None
                benefit_mask_active_fraction = torch.ones((), device=device)
            if bool(args.alpha_conditioned_residual):
                alpha_blend = min(max(train_alpha, 0.0), 1.0)
                teacher = torch.clamp(parent + float(alpha_blend) * (raw_teacher - parent), 0.0, 1.0)
            else:
                teacher = raw_teacher
            teacher = _blend_target_with_parent(parent, teacher, benefit_mask)
            teacher = _blend_target_with_parent(parent, teacher, debt_mask)
            if str(args.support_loss_mask) == "active" and face_batch is not None and hasattr(model, "support_mask"):
                with torch.no_grad():
                    loss_mask = model.support_mask(feat, face_batch).to(device=device, dtype=feat.dtype)
                loss_mask_active_fraction = torch.mean((loss_mask > 0.0).float())
                if (
                    float(loss_mask_active_fraction.detach().cpu().item())
                    >= float(args.support_patch_min_active_fraction)
                    or attempt == support_attempts - 1
                ):
                    break
            else:
                break
        pred_delta = model(feat, face_batch)
        adapted = torch.clamp(parent + pred_delta, 0.0, 1.0)
        if loss_mask is not None:
            loss_mask = loss_mask.to(dtype=adapted.dtype)
        loss_mask_active_fraction = (
            torch.mean((loss_mask > 0.0).float()) if loss_mask is not None else torch.ones((), device=device)
        )
        loss_teacher_l1 = _masked_l1_loss(adapted, teacher, loss_mask)
        loss_teacher_ssim = _masked_ssim_loss(adapted, teacher, loss_mask)
        loss_teacher_lpips = _lpips_train_loss(
            train_lpips_model,
            _fill_inactive_with_target(adapted, teacher, loss_mask),
            teacher,
            int(args.lpips_loss_max_side),
        )
        loss_teacher_lpips_noharm = torch.zeros((), device=device)
        if float(args.teacher_lpips_noharm_weight) > 0.0:
            parent_teacher_lpips = _lpips_train_loss(
                train_lpips_model,
                _fill_inactive_with_target(parent, teacher, loss_mask),
                teacher,
                int(args.lpips_loss_max_side),
            ).detach()
            adapted_teacher_lpips = (
                loss_teacher_lpips
                if float(args.teacher_lpips_weight) > 0.0
                else _lpips_train_loss(
                    train_lpips_model,
                    _fill_inactive_with_target(adapted, teacher, loss_mask),
                    teacher,
                    int(args.lpips_loss_max_side),
                )
            )
            loss_teacher_lpips_noharm = torch.relu(
                adapted_teacher_lpips - parent_teacher_lpips + float(args.teacher_lpips_noharm_margin)
            )
        loss_teacher_grad = (
            _luma_gradient_loss(adapted, teacher, int(args.grad_loss_max_side), loss_mask)
            if float(args.teacher_grad_weight) > 0.0
            else torch.zeros((), device=device)
        )
        loss_teacher_highfreq = (
            _multiscale_highfreq_loss(
                adapted,
                teacher,
                int(args.highfreq_loss_max_side),
                int(args.highfreq_loss_levels),
                loss_mask,
            )
            if float(args.teacher_highfreq_weight) > 0.0
            else torch.zeros((), device=device)
        )
        loss_teacher_residual_cosine = torch.zeros((), device=device)
        loss_teacher_residual_energy = torch.zeros((), device=device)
        teacher_residual_cosine = torch.zeros((), device=device)
        teacher_residual_projection_active_fraction = torch.zeros((), device=device)
        if float(args.teacher_residual_cosine_weight) > 0.0 or float(args.teacher_residual_energy_weight) > 0.0:
            (
                loss_teacher_residual_cosine,
                loss_teacher_residual_energy,
                teacher_residual_cosine,
                teacher_residual_projection_active_fraction,
            ) = _teacher_residual_projection_losses(
                pred_delta,
                teacher - parent,
                loss_mask,
                float(args.teacher_residual_projection_min_l1),
            )
        loss_gt = torch.zeros((), device=device)
        loss_gt_ssim = torch.zeros((), device=device)
        loss_gt_lpips = torch.zeros((), device=device)
        loss_gt_lpips_noharm = torch.zeros((), device=device)
        loss_gt_grad = torch.zeros((), device=device)
        loss_gt_highfreq = torch.zeros((), device=device)
        if batch.gt is not None and (
            float(args.gt_l1_weight) > 0.0
            or float(args.gt_ssim_weight) > 0.0
            or float(args.gt_lpips_weight) > 0.0
            or float(args.gt_lpips_noharm_weight) > 0.0
            or float(args.gt_grad_weight) > 0.0
            or float(args.gt_highfreq_weight) > 0.0
        ):
            raw_gt = batch.gt.unsqueeze(0).to(device)
            if bool(args.alpha_conditioned_residual):
                alpha_blend = min(max(train_alpha, 0.0), 1.0)
                gt = torch.clamp(parent + float(alpha_blend) * (raw_gt - parent), 0.0, 1.0)
            else:
                gt = raw_gt
            gt = _blend_target_with_parent(parent, gt, benefit_mask)
            gt = _blend_target_with_parent(parent, gt, debt_mask)
            loss_gt = _masked_l1_loss(adapted, gt, loss_mask)
            if float(args.gt_ssim_weight) > 0.0:
                loss_gt_ssim = _masked_ssim_loss(adapted, gt, loss_mask)
            if float(args.gt_lpips_weight) > 0.0:
                loss_gt_lpips = _lpips_train_loss(
                    train_lpips_model,
                    _fill_inactive_with_target(adapted, gt, loss_mask),
                    gt,
                    int(args.lpips_loss_max_side),
                )
            if float(args.gt_lpips_noharm_weight) > 0.0:
                parent_gt_lpips = _lpips_train_loss(
                    train_lpips_model,
                    _fill_inactive_with_target(parent, gt, loss_mask),
                    gt,
                    int(args.lpips_loss_max_side),
                ).detach()
                adapted_gt_lpips = (
                    loss_gt_lpips
                    if float(args.gt_lpips_weight) > 0.0
                    else _lpips_train_loss(
                        train_lpips_model,
                        _fill_inactive_with_target(adapted, gt, loss_mask),
                        gt,
                        int(args.lpips_loss_max_side),
                    )
                )
                loss_gt_lpips_noharm = torch.relu(
                    adapted_gt_lpips - parent_gt_lpips + float(args.gt_lpips_noharm_margin)
                )
            if float(args.gt_grad_weight) > 0.0:
                loss_gt_grad = _luma_gradient_loss(adapted, gt, int(args.grad_loss_max_side), loss_mask)
            if float(args.gt_highfreq_weight) > 0.0:
                loss_gt_highfreq = _multiscale_highfreq_loss(
                    adapted,
                    gt,
                    int(args.highfreq_loss_max_side),
                    int(args.highfreq_loss_levels),
                    loss_mask,
                )
        loss_debt_noop = torch.zeros((), device=device)
        if debt_mask is not None and float(args.residual_debt_noop_weight) > 0.0:
            debt_r = _resize_mask_bchw(debt_mask.to(device=pred_delta.device, dtype=pred_delta.dtype), pred_delta.shape[-2:])
            inactive = 1.0 - debt_r
            denom = torch.clamp(inactive.sum() * pred_delta.shape[1], min=1.0)
            loss_debt_noop = torch.sum(torch.abs(pred_delta) * inactive) / denom
        loss_mag = torch.mean(torch.abs(pred_delta))
        loss = (
            float(args.teacher_l1_weight) * loss_teacher_l1
            + float(args.teacher_ssim_weight) * loss_teacher_ssim
            + float(args.teacher_lpips_weight) * loss_teacher_lpips
            + float(args.teacher_lpips_noharm_weight) * loss_teacher_lpips_noharm
            + float(args.teacher_grad_weight) * loss_teacher_grad
            + float(args.teacher_highfreq_weight) * loss_teacher_highfreq
            + float(args.teacher_residual_cosine_weight) * loss_teacher_residual_cosine
            + float(args.teacher_residual_energy_weight) * loss_teacher_residual_energy
            + float(args.gt_l1_weight) * loss_gt
            + float(args.gt_ssim_weight) * loss_gt_ssim
            + float(args.gt_lpips_weight) * loss_gt_lpips
            + float(args.gt_lpips_noharm_weight) * loss_gt_lpips_noharm
            + float(args.gt_grad_weight) * loss_gt_grad
            + float(args.gt_highfreq_weight) * loss_gt_highfreq
            + float(args.residual_debt_noop_weight) * loss_debt_noop
            + float(args.delta_l1_weight) * loss_mag
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if wandb_run is not None and (step == 1 or step % 50 == 0):
            wandb_run.log(
                {
                    "train/loss": float(loss.detach().cpu().item()),
                    "train/teacher_l1": float(loss_teacher_l1.detach().cpu().item()),
                    "train/teacher_ssim_loss": float(loss_teacher_ssim.detach().cpu().item()),
                    "train/teacher_lpips_loss": float(loss_teacher_lpips.detach().cpu().item()),
                    "train/teacher_lpips_noharm_loss": float(
                        loss_teacher_lpips_noharm.detach().cpu().item()
                    ),
                    "train/teacher_grad_loss": float(loss_teacher_grad.detach().cpu().item()),
                    "train/teacher_highfreq_loss": float(loss_teacher_highfreq.detach().cpu().item()),
                    "train/teacher_residual_cosine_loss": float(
                        loss_teacher_residual_cosine.detach().cpu().item()
                    ),
                    "train/teacher_residual_energy_loss": float(
                        loss_teacher_residual_energy.detach().cpu().item()
                    ),
                    "train/teacher_residual_cosine": float(teacher_residual_cosine.detach().cpu().item()),
                    "train/teacher_residual_projection_active_fraction": float(
                        teacher_residual_projection_active_fraction.detach().cpu().item()
                    ),
                    "train/gt_l1": float(loss_gt.detach().cpu().item()),
                    "train/gt_ssim_loss": float(loss_gt_ssim.detach().cpu().item()),
                    "train/gt_lpips_loss": float(loss_gt_lpips.detach().cpu().item()),
                    "train/gt_lpips_noharm_loss": float(loss_gt_lpips_noharm.detach().cpu().item()),
                    "train/gt_grad_loss": float(loss_gt_grad.detach().cpu().item()),
                    "train/gt_highfreq_loss": float(loss_gt_highfreq.detach().cpu().item()),
                    "train/residual_debt_noop_loss": float(loss_debt_noop.detach().cpu().item()),
                    "train/delta_l1": float(loss_mag.detach().cpu().item()),
                    "train/support_loss_active_fraction": float(loss_mask_active_fraction.detach().cpu().item()),
                    "train/residual_debt_active_fraction": float(debt_mask_active_fraction.detach().cpu().item()),
                    "train/teacher_benefit_active_fraction": float(
                        benefit_mask_active_fraction.detach().cpu().item()
                    ),
                    "train/alpha": float(train_alpha),
                    "train/step": int(step),
                }
            )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "input_channels": int(in_ch),
            "face_lut": None if face_lut is None else face_lut,
            "surface_support_stats": surface_support_stats,
            "surface_evidence_stats": surface_evidence_stats,
            "surface_evidence_summary": surface_evidence_summary,
            "face_embedding_rows": int(0 if face_lut is None else face_lut.size + 1),
            "surface_texture_rows": int(
                0
                if face_lut is None or str(args.model_type) not in surface_model_types
                else face_lut.size * int(args.surface_texture_size) * int(args.surface_texture_size) + 1
            ),
        },
        checkpoint_path,
    )
    policy_val = evaluate_policy_val(
        model,
        val_paths,
        residual_key=str(args.residual_rgb_key),
        face_lut=face_lut,
        alpha_grid=alpha_grid,
        alpha_conditioned_residual=bool(args.alpha_conditioned_residual),
        policy_allow_noop_alpha=bool(args.policy_allow_noop_alpha),
        policy_select_mode=str(args.policy_select_mode),
        policy_tail_fraction=float(args.policy_tail_fraction),
        policy_min_psnr_gain=float(args.policy_min_psnr_gain),
        policy_min_ssim_gain=float(args.policy_min_ssim_gain),
        policy_min_lpips_gain=float(args.policy_min_lpips_gain),
        policy_cvar_psnr_gain=float(args.policy_cvar_psnr_gain),
        policy_cvar_ssim_gain=float(args.policy_cvar_ssim_gain),
        policy_cvar_lpips_gain=float(args.policy_cvar_lpips_gain),
        device=device,
        eval_tile=int(args.eval_tile),
        eval_overlap=int(args.eval_overlap),
        ssim_max_side=int(args.ssim_max_side),
        lpips_max_side=int(args.lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
        output_dir=None if bool(args.skip_policy_val_renders) else output_dir / "policy_val_best",
    )
    all_axis = policy_val.get("best_all_axis") is not None
    selected_for_phasej = policy_val.get("best_all_axis") or policy_val.get("best") or {}
    phasej_policy_val_numeric_above_reference = bool(
        selected_for_phasej
        and float(selected_for_phasej.get("candidate_psnr", -math.inf)) > float(args.phasej_flowers_psnr)
        and float(selected_for_phasej.get("candidate_ssim", -math.inf)) > float(args.phasej_flowers_ssim)
        and (
            not bool(args.compute_lpips)
            or float(selected_for_phasej.get("candidate_lpips", math.inf)) < float(args.phasej_flowers_lpips)
        )
    )
    target_apply = None
    target_no_gt_precheck = None
    if str(args.target_evidence_dir):
        target_no_gt_precheck = verify_target_no_gt(Path(args.target_evidence_dir))
    if str(args.target_evidence_dir) and all_axis:
        selected_alpha = (
            float(args.target_alpha)
            if args.target_alpha is not None
            else float((policy_val.get("best_all_axis") or policy_val["best"])["alpha"])
        )
        target_apply = apply_target(
            model,
            Path(str(args.target_evidence_dir)),
            face_lut=face_lut,
            scene_name=str(args.scene_name),
            method_name=str(args.method_name),
            alpha=selected_alpha,
            alpha_conditioned_residual=bool(args.alpha_conditioned_residual),
            device=device,
            eval_tile=int(args.eval_tile),
            eval_overlap=int(args.eval_overlap),
            output_dir=output_dir,
        )
    elif str(args.target_evidence_dir):
        target_apply = {"skipped": True, "reason": "policy-val all-axis gate failed"}
    train_gt_weight_sum = (
        float(args.gt_l1_weight)
        + float(args.gt_ssim_weight)
        + float(args.gt_lpips_weight)
        + float(args.gt_lpips_noharm_weight)
        + float(args.gt_grad_weight)
        + float(args.gt_highfreq_weight)
    )
    train_fit_gt_available = any("gt" in example for example in train_examples)
    residual_debt_active_fractions = [
        float(torch.mean(example["residual_debt_mask"]).item())
        for example in train_examples
        if "residual_debt_mask" in example
    ]
    teacher_benefit_active_fractions = [
        float(torch.mean(example["teacher_benefit_mask"]).item())
        for example in train_examples
        if "teacher_benefit_mask" in example
    ]
    gt_usage_audit = {
        "schema": "spcarnet_gt_usage_audit_v1",
        "uses_train_fit_gt": bool(
            train_fit_gt_available
            and (
                train_gt_weight_sum > 0.0
                or bool(args.residual_debt_mask)
                or str(args.teacher_benefit_mask_mode) != "off"
            )
        ),
        "train_fit_gt_available": bool(train_fit_gt_available),
        "train_fit_gt_weight_sum": float(train_gt_weight_sum),
        "uses_train_fit_gt_for_residual_debt_mask": bool(args.residual_debt_mask and train_fit_gt_available),
        "uses_train_fit_gt_for_teacher_benefit_mask": bool(
            str(args.teacher_benefit_mask_mode) != "off" and train_fit_gt_available
        ),
        "uses_policy_val_gt": True,
        "policy_val_gt_purpose": "candidate certification and alpha selection only",
        "uses_target_or_test_gt_during_apply": False,
        "target_or_test_gt_after_apply_purpose": "final evaluation only, if separately populated",
        "uses_target_view_geometry_for_capacity": bool(str(args.surface_target_visible_evidence_dir)),
        "target_view_geometry_capacity_purpose": (
            "transductive no-RGB-GT face-capacity allocation only"
            if str(args.surface_target_visible_evidence_dir)
            else ""
        ),
        "target_no_gt_verifier_passed": bool(
            (target_apply or {}).get("no_gt_verify", {}).get(
                "passed",
                (target_no_gt_precheck or {}).get("passed", False),
            )
        ),
        "target_gt_visible_to_apply": bool(
            (target_apply or {}).get("no_gt_verify", {}).get(
                "target_gt_visible_to_apply",
                (target_no_gt_precheck or {}).get("target_gt_visible_to_apply", False),
            )
        ),
        "target_residual_visible_to_apply": bool(
            (target_apply or {}).get("no_gt_verify", {}).get(
                "target_residual_visible_to_apply",
                (target_no_gt_precheck or {}).get("target_residual_visible_to_apply", False),
            )
        ),
    }
    payload = {
        "schema": "spcarnet_surface_conditioned_residual_model_v2",
        "artifact_prefix": artifact_prefix,
        "args": vars(args),
        "splits": {
            "all_views": int(len(paths)),
            "train_fit_views": int(len(fit_paths)),
            "policy_val_views": int(len(val_paths)),
            "policy_val_view_names": [p.stem for p in val_paths],
        },
        "model": {
            "model_type": str(args.model_type),
            "input_channels": int(in_ch),
            "alpha_conditioned_residual": bool(args.alpha_conditioned_residual),
            "base_channels": int(args.base_channels),
            "max_delta": float(args.max_delta),
            "confidence_mode": str(args.confidence_mode),
            "confidence_bias": float(args.confidence_bias),
            "confidence_min": float(args.confidence_min),
            "confidence_max": float(args.confidence_max),
            "face_embedding_dim": int(args.face_embedding_dim),
            "face_embedding_rows": int(0 if face_lut is None else face_lut.size + 1),
            "face_embedding_train_fit_unique_faces": int(0 if face_lut is None else face_lut.size),
            "surface_texture_size": int(args.surface_texture_size),
            "surface_feature_dim": int(args.surface_feature_dim),
            "surface_decoder_hidden": int(args.surface_decoder_hidden),
            "surface_decoder_layers": int(args.surface_decoder_layers),
            "surface_target_visible_evidence_dir": str(args.surface_target_visible_evidence_dir),
            "surface_texture_rows": int(
                0
                if face_lut is None or str(args.model_type) not in surface_model_types
                else face_lut.size * int(args.surface_texture_size) * int(args.surface_texture_size) + 1
            ),
            "surface_support_gate_floor": float(args.surface_support_gate_floor),
            "surface_support_unknown_gate_floor": float(args.surface_support_unknown_gate_floor),
            "surface_evidence_texture_enabled": bool(args.enable_surface_evidence_texture),
            "surface_source_evidence_bank_enabled": bool(args.enable_surface_source_evidence_bank),
            "surface_source_evidence_top_k": int(args.surface_source_evidence_top_k),
            "surface_evidence_channels": int(0 if surface_evidence_stats is None else surface_evidence_stats.shape[1]),
            "surface_evidence_min_bin_support": int(args.surface_evidence_min_bin_support),
            "surface_evidence_residual_rgb_key": str(args.surface_evidence_residual_rgb_key or args.residual_rgb_key),
            "surface_evidence_residual_prior_weight": float(args.surface_evidence_residual_prior_weight),
            "surface_evidence_view_gate_power": float(args.surface_evidence_view_gate_power),
            "surface_evidence_active_rows": int(
                0 if surface_evidence_summary is None else surface_evidence_summary.get("active_rows", 0)
            ),
            "lowrank_rank": int(args.lowrank_rank),
            "lowrank_min_bin_support": int(args.lowrank_min_bin_support),
            "lowrank_basis_init_std": float(args.lowrank_basis_init_std),
            "lowrank_active_support_rows": int(
                0 if surface_support_stats is None else np.sum(np.asarray(surface_support_stats)[:, 2] > 0.0)
            ),
            "residual_debt_mask": bool(args.residual_debt_mask),
            "residual_debt_quantile": float(args.residual_debt_quantile),
            "residual_debt_min_l1": float(args.residual_debt_min_l1),
            "residual_debt_dilate": int(args.residual_debt_dilate),
            "residual_debt_noop_weight": float(args.residual_debt_noop_weight),
            "train_residual_debt_active_fraction": (
                float(np.mean(residual_debt_active_fractions)) if residual_debt_active_fractions else None
            ),
            "teacher_benefit_mask_mode": str(args.teacher_benefit_mask_mode),
            "teacher_benefit_min_gain_l1": float(args.teacher_benefit_min_gain_l1),
            "teacher_benefit_dilate": int(args.teacher_benefit_dilate),
            "train_teacher_benefit_active_fraction": (
                float(np.mean(teacher_benefit_active_fractions)) if teacher_benefit_active_fractions else None
            ),
            "checkpoint": str(checkpoint_path),
        },
        "policy_val": policy_val,
        "policy_val_all_axis_pass": bool(all_axis),
        "phasej_flowers_gate": {
            "psnr": float(args.phasej_flowers_psnr),
            "ssim": float(args.phasej_flowers_ssim),
            "lpips": float(args.phasej_flowers_lpips),
            "policy_val_numeric_above_reference": bool(phasej_policy_val_numeric_above_reference),
            "official_exact_status": "not_evaluated_by_this_training_script",
            "note": "Policy-val numeric comparison is diagnostic only and is not a Phase-J exact win; v169 completion still requires official flowers exact metrics.",
        },
        "target_apply": target_apply,
        "target_no_gt_precheck": target_no_gt_precheck,
        "gt_usage_audit": gt_usage_audit,
        "audit_notes": {
            "policy_val_gate_scope": "candidate improvement over parent on held-out fit/policy-val views",
            "phasej_gate_recorded_by_script": True,
            "phasej_exact_gate_enforced_by_script": False,
            "phasej_gate_must_be_checked_by_official_eval": True,
            "target_alpha_selection": "manual_target_alpha" if args.target_alpha is not None else "policy_val_selected",
            "alpha_contract": (
                "model_outputs_final_delta_for_selected_alpha"
                if bool(args.alpha_conditioned_residual)
                else "posthoc_policy_val_alpha_multiplier"
            ),
            "alpha_conditioned_train_grid": train_alpha_grid if bool(args.alpha_conditioned_residual) else [],
            "alpha_contract_warning": (
                ""
                if bool(args.alpha_conditioned_residual)
                else "policy-val may select an alpha multiplier that was not part of the training objective"
            ),
            "target_eval_evidence_dir_integrated": False,
            "target_eval_evidence_dir_note": (
                "--target_eval_evidence_dir is recorded in args only; run official eval scripts after no-GT apply"
            ),
        },
        "references": {
            "phasej_flowers_gate": "20.304358 / 0.557770 / 0.329222",
            "v183_flowers_exact": "19.832029 / 0.505779 / 0.405907",
        },
    }
    payload["output_json"] = str(report_json_path)
    _write_json(report_json_path, payload)
    _write_md(report_md_path, payload)
    if wandb_run is not None:
        best = policy_val["best"]
        wandb_run.log(
            {
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "policy_val/all_axis_pass": int(all_axis),
            }
        )
        if target_apply and "mean_changed_fraction" in target_apply:
            wandb_run.log({"target/mean_changed_fraction": float(target_apply.get("mean_changed_fraction", 0.0))})
        wandb_run.finish()
    print("OUT", report_json_path, flush=True)
    print("BEST", json.dumps(policy_val["best"], indent=2, sort_keys=True), flush=True)
    print("BEST_ALL_AXIS", json.dumps(policy_val.get("best_all_axis"), indent=2, sort_keys=True), flush=True)
    if target_apply is not None:
        print("TARGET", json.dumps({k: v for k, v in target_apply.items() if k != "per_view"}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
