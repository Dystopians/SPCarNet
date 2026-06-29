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


def _luma_gradient_loss(pred: torch.Tensor, target: torch.Tensor, max_side: int) -> torch.Tensor:
    pred_luma = _luma(_resize_bchw_tensor(pred, int(max_side)))
    target_luma = _luma(_resize_bchw_tensor(target, int(max_side)))
    pred_dx = pred_luma[:, :, :, 1:] - pred_luma[:, :, :, :-1]
    pred_dy = pred_luma[:, :, 1:, :] - pred_luma[:, :, :-1, :]
    target_dx = target_luma[:, :, :, 1:] - target_luma[:, :, :, :-1]
    target_dy = target_luma[:, :, 1:, :] - target_luma[:, :, :-1, :]
    return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))


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


def _load_example(
    path: Path,
    residual_key: str,
    max_side: int,
    include_gt: bool,
    face_lut: np.ndarray | None = None,
) -> dict[str, Any]:
    z = np.load(path)
    parent = torch.from_numpy(np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32))
    residual = torch.from_numpy(np.asarray(z[residual_key], dtype=np.float32)[:3])
    teacher = torch.clamp(parent + residual, 0.0, 1.0)
    features = torch.from_numpy(_load_input_chw(z))
    out: dict[str, Any] = {
        "name": path.stem,
        "features": _resize_chw_tensor(features, max_side),
        "face_ids": _load_face_ids_tensor(z, face_lut, max_side),
        "parent": _resize_chw_tensor(parent, max_side),
        "teacher": _resize_chw_tensor(teacher, max_side),
    }
    if include_gt and "rgb_gt" in z:
        gt = torch.from_numpy(np.clip(_to_chw(z["rgb_gt"])[:3], 0.0, 1.0).astype(np.float32))
        out["gt"] = _resize_chw_tensor(gt, max_side)
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


def _sample_patch(example: dict[str, Any], patch_size: int, rng: random.Random) -> TrainBatch:
    features = example["features"]
    face_ids = example.get("face_ids")
    parent = example["parent"]
    teacher = example["teacher"]
    gt = example.get("gt")
    _, h, w = parent.shape
    if int(patch_size) <= 0 or h <= int(patch_size) or w <= int(patch_size):
        return TrainBatch(features, face_ids, parent, teacher, gt)
    ph = pw = int(patch_size)
    y = rng.randint(0, h - ph)
    x = rng.randint(0, w - pw)
    gt_patch = None if gt is None else gt[:, y : y + ph, x : x + pw]
    face_patch = None if face_ids is None else face_ids[y : y + ph, x : x + pw]
    return TrainBatch(
        features[:, y : y + ph, x : x + pw],
        face_patch,
        parent[:, y : y + ph, x : x + pw],
        teacher[:, y : y + ph, x : x + pw],
        gt_patch,
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
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    model.eval()
    for path in tqdm(val_paths, desc="policy-val v184"):
        z = np.load(path)
        features = torch.from_numpy(_load_input_chw(z))
        face_ids = _load_face_ids_tensor(z, face_lut, max_side=-1)
        parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
        gt = np.clip(_to_chw(z["rgb_gt"])[:3], 0.0, 1.0).astype(np.float32)
        delta = _predict_delta_tiled(
            model,
            features,
            face_ids=face_ids,
            device=device,
            tile=eval_tile,
            overlap=eval_overlap,
        ).numpy()
        cache[path.stem] = (delta, parent, gt)
        p_psnr = _psnr(parent, gt)
        p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
        p_lpips = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
        for alpha in alpha_grid:
            cand = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
            row = {
                "view": path.stem,
                "parent_psnr": float(p_psnr),
                "candidate_psnr": float(_psnr(cand, gt)),
                "parent_ssim": float(p_ssim),
                "candidate_ssim": float(image_ssim_chw(cand, gt, int(ssim_max_side))),
            }
            row["psnr_gain"] = row["candidate_psnr"] - row["parent_psnr"]
            row["ssim_gain"] = row["candidate_ssim"] - row["parent_ssim"]
            row["changed_fraction"] = float(np.mean(np.any(np.abs(float(alpha) * delta) > (0.5 / 255.0), axis=0)))
            if compute_lpips:
                c_lpips = image_lpips_chw(cand, gt, int(lpips_max_side), lpips_model)
                row["parent_lpips"] = float(p_lpips)
                row["candidate_lpips"] = float(c_lpips)
                row["lpips_gain"] = float(p_lpips - c_lpips)
            rows_by_alpha[float(alpha)].append(row)
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
        }
        if compute_lpips:
            lpips_gain = [float(r["lpips_gain"]) for r in rows]
            summary["parent_lpips"] = float(np.mean([r["parent_lpips"] for r in rows]))
            summary["candidate_lpips"] = float(np.mean([r["candidate_lpips"] for r in rows]))
            summary["lpips_gain"] = float(np.mean(lpips_gain))
            summary["lpips_positive_view_fraction"] = float(np.mean(np.asarray(lpips_gain) > 0.0))
        summaries.append(summary)
    best = max(
        summaries,
        key=lambda row: (
            float(row.get("psnr_gain", 0.0))
            + 20.0 * float(row.get("ssim_gain", 0.0))
            + 20.0 * float(row.get("lpips_gain", 0.0))
        ),
    )
    pass_rows = [
        row
        for row in summaries
        if float(row.get("psnr_gain", 0.0)) > 0.0
        and float(row.get("ssim_gain", 0.0)) > 0.0
        and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
    ]
    best_all_axis = None
    if pass_rows:
        best_all_axis = max(
            pass_rows,
            key=lambda row: (
                float(row.get("psnr_gain", 0.0))
                + 20.0 * float(row.get("ssim_gain", 0.0))
                + 20.0 * float(row.get("lpips_gain", 0.0))
            ),
        )
    if output_dir is not None:
        alpha = float((best_all_axis or best)["alpha"])
        (output_dir / "renders").mkdir(parents=True, exist_ok=True)
        (output_dir / "parent").mkdir(parents=True, exist_ok=True)
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)
        for stem, (delta, parent, gt) in cache.items():
            save_image_chw(output_dir / "renders" / f"{stem}.png", np.clip(parent + alpha * delta, 0.0, 1.0))
            save_image_chw(output_dir / "parent" / f"{stem}.png", parent)
            save_image_chw(output_dir / "gt" / f"{stem}.png", gt)
    return {
        "best": best,
        "best_all_axis": best_all_axis,
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
        delta = _predict_delta_tiled(
            model,
            features,
            face_ids=face_ids,
            device=device,
            tile=eval_tile,
            overlap=eval_overlap,
        ).numpy()
        cand = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
        save_image_chw(render_dir / f"{path.stem}.png", cand)
        save_image_chw(parent_dir / f"{path.stem}.png", parent)
        row: dict[str, Any] = {
            "view": path.stem,
            "changed_fraction": float(np.mean(np.any(np.abs(float(alpha) * delta) > (0.5 / 255.0), axis=0))),
        }
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
        "per_view": rows,
    }
    return summary


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    target = payload.get("target_apply") or {}
    leakage = payload.get("gt_usage_audit", {})
    lines = [
        "# v184 Surface-Conditioned Residual U-Net Audit",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- target exact run: `{bool(target)}`",
        f"- target no-GT verifier: `{target.get('no_gt_verify', {}).get('passed')}`",
        f"- uses train-fit GT: `{leakage.get('uses_train_fit_gt')}`",
        f"- uses policy-val GT: `{leakage.get('uses_policy_val_gt')}`",
        f"- uses target/test GT during apply: `{leakage.get('uses_target_or_test_gt_during_apply')}`",
        "",
        "## Policy-Val",
        "",
        "| alpha | PSNR gain | SSIM gain | LPIPS gain | changed fraction |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.6f} | {best.get('psnr_gain', 0.0):+.9f} | "
            f"{best.get('ssim_gain', 0.0):+.9f} | {best.get('lpips_gain', 0.0):+.9f} | "
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
    parser = argparse.ArgumentParser(description="Train a surface-conditioned residual U-Net for Phase-J teacher baking.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_FIT_EVIDENCE)
    parser.add_argument("--target_evidence_dir", default="")
    parser.add_argument("--target_eval_evidence_dir", default="")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--train_max_side", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
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
    parser.add_argument("--teacher_l1_weight", type=float, default=1.0)
    parser.add_argument("--teacher_ssim_weight", type=float, default=0.20)
    parser.add_argument("--teacher_lpips_weight", type=float, default=0.0)
    parser.add_argument("--teacher_grad_weight", type=float, default=0.0)
    parser.add_argument("--gt_l1_weight", type=float, default=0.10)
    parser.add_argument("--gt_ssim_weight", type=float, default=0.0)
    parser.add_argument("--gt_lpips_weight", type=float, default=0.0)
    parser.add_argument("--gt_grad_weight", type=float, default=0.0)
    parser.add_argument("--lpips_loss_max_side", type=int, default=128)
    parser.add_argument("--grad_loss_max_side", type=int, default=256)
    parser.add_argument("--delta_l1_weight", type=float, default=1.0e-4)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--target_alpha", type=float, default=None)
    parser.add_argument("--method_name", default="ours_26000_v184_surface_conditioned_unet_flowers")
    parser.add_argument("--scene_name", default="flowers")
    parser.add_argument("--eval_tile", type=int, default=512)
    parser.add_argument("--eval_overlap", type=int, default=32)
    parser.add_argument("--ssim_max_side", type=int, default=-1)
    parser.add_argument("--lpips_max_side", type=int, default=256)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--skip_policy_val_renders", action="store_true")
    parser.add_argument("--output_dir", default="/dev/shm/peilincai_spcarnet_v184_surface_conditioned_unet")
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
    if int(args.face_embedding_dim) > 0:
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
        _load_example(p, str(args.residual_rgb_key), int(args.train_max_side), include_gt=True, face_lut=face_lut)
        for p in tqdm(fit_paths, desc="preload train-fit")
    ]
    in_ch = int(train_examples[0]["features"].shape[0])
    if int(args.face_embedding_dim) > 0:
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
    if float(args.teacher_lpips_weight) > 0.0 or float(args.gt_lpips_weight) > 0.0:
        train_lpips_model = build_lpips_model().to(device).eval()
    rng = random.Random(int(args.seed))
    for step in tqdm(range(1, int(args.steps) + 1), desc="train v184 surface U-Net"):
        model.train()
        ex = rng.choice(train_examples)
        batch = _sample_patch(ex, int(args.patch_size), rng)
        feat = batch.features.unsqueeze(0).to(device)
        face_batch = None if batch.face_ids is None else batch.face_ids.unsqueeze(0).to(device)
        parent = batch.parent.unsqueeze(0).to(device)
        teacher = batch.teacher.unsqueeze(0).to(device)
        pred_delta = model(feat, face_batch)
        adapted = torch.clamp(parent + pred_delta, 0.0, 1.0)
        loss_teacher_l1 = torch.mean(torch.abs(adapted - teacher))
        loss_teacher_ssim = 1.0 - ssim(adapted, teacher)
        loss_teacher_lpips = _lpips_train_loss(
            train_lpips_model,
            adapted,
            teacher,
            int(args.lpips_loss_max_side),
        )
        loss_teacher_grad = (
            _luma_gradient_loss(adapted, teacher, int(args.grad_loss_max_side))
            if float(args.teacher_grad_weight) > 0.0
            else torch.zeros((), device=device)
        )
        loss_gt = torch.zeros((), device=device)
        loss_gt_ssim = torch.zeros((), device=device)
        loss_gt_lpips = torch.zeros((), device=device)
        loss_gt_grad = torch.zeros((), device=device)
        if batch.gt is not None and (
            float(args.gt_l1_weight) > 0.0
            or float(args.gt_ssim_weight) > 0.0
            or float(args.gt_lpips_weight) > 0.0
            or float(args.gt_grad_weight) > 0.0
        ):
            gt = batch.gt.unsqueeze(0).to(device)
            loss_gt = torch.mean(torch.abs(adapted - gt))
            if float(args.gt_ssim_weight) > 0.0:
                loss_gt_ssim = 1.0 - ssim(adapted, gt)
            if float(args.gt_lpips_weight) > 0.0:
                loss_gt_lpips = _lpips_train_loss(
                    train_lpips_model,
                    adapted,
                    gt,
                    int(args.lpips_loss_max_side),
                )
            if float(args.gt_grad_weight) > 0.0:
                loss_gt_grad = _luma_gradient_loss(adapted, gt, int(args.grad_loss_max_side))
        loss_mag = torch.mean(torch.abs(pred_delta))
        loss = (
            float(args.teacher_l1_weight) * loss_teacher_l1
            + float(args.teacher_ssim_weight) * loss_teacher_ssim
            + float(args.teacher_lpips_weight) * loss_teacher_lpips
            + float(args.teacher_grad_weight) * loss_teacher_grad
            + float(args.gt_l1_weight) * loss_gt
            + float(args.gt_ssim_weight) * loss_gt_ssim
            + float(args.gt_lpips_weight) * loss_gt_lpips
            + float(args.gt_grad_weight) * loss_gt_grad
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
                    "train/teacher_grad_loss": float(loss_teacher_grad.detach().cpu().item()),
                    "train/gt_l1": float(loss_gt.detach().cpu().item()),
                    "train/gt_ssim_loss": float(loss_gt_ssim.detach().cpu().item()),
                    "train/gt_lpips_loss": float(loss_gt_lpips.detach().cpu().item()),
                    "train/gt_grad_loss": float(loss_gt_grad.detach().cpu().item()),
                    "train/delta_l1": float(loss_mag.detach().cpu().item()),
                    "train/step": int(step),
                }
            )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "input_channels": int(in_ch),
            "face_lut": None if face_lut is None else face_lut,
            "face_embedding_rows": int(0 if face_lut is None else face_lut.size + 1),
        },
        output_dir / "v184_surface_conditioned_unet.pt",
    )
    policy_val = evaluate_policy_val(
        model,
        val_paths,
        residual_key=str(args.residual_rgb_key),
        face_lut=face_lut,
        alpha_grid=_parse_float_grid(str(args.alpha_grid)),
        device=device,
        eval_tile=int(args.eval_tile),
        eval_overlap=int(args.eval_overlap),
        ssim_max_side=int(args.ssim_max_side),
        lpips_max_side=int(args.lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
        output_dir=None if bool(args.skip_policy_val_renders) else output_dir / "policy_val_best",
    )
    all_axis = policy_val.get("best_all_axis") is not None
    target_apply = None
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
        + float(args.gt_grad_weight)
    )
    train_fit_gt_available = any("gt" in example for example in train_examples)
    gt_usage_audit = {
        "schema": "spcarnet_gt_usage_audit_v1",
        "uses_train_fit_gt": bool(train_fit_gt_available and train_gt_weight_sum > 0.0),
        "train_fit_gt_available": bool(train_fit_gt_available),
        "train_fit_gt_weight_sum": float(train_gt_weight_sum),
        "uses_policy_val_gt": True,
        "policy_val_gt_purpose": "candidate certification and alpha selection only",
        "uses_target_or_test_gt_during_apply": False,
        "target_or_test_gt_after_apply_purpose": "final evaluation only, if separately populated",
        "target_no_gt_verifier_passed": bool((target_apply or {}).get("no_gt_verify", {}).get("passed", False)),
        "target_gt_visible_to_apply": bool(
            (target_apply or {}).get("no_gt_verify", {}).get("target_gt_visible_to_apply", False)
        ),
        "target_residual_visible_to_apply": bool(
            (target_apply or {}).get("no_gt_verify", {}).get("target_residual_visible_to_apply", False)
        ),
    }
    payload = {
        "schema": "spcarnet_v184_surface_conditioned_residual_unet_v1",
        "args": vars(args),
        "splits": {
            "all_views": int(len(paths)),
            "train_fit_views": int(len(fit_paths)),
            "policy_val_views": int(len(val_paths)),
            "policy_val_view_names": [p.stem for p in val_paths],
        },
        "model": {
            "input_channels": int(in_ch),
            "base_channels": int(args.base_channels),
            "max_delta": float(args.max_delta),
            "confidence_mode": str(args.confidence_mode),
            "confidence_bias": float(args.confidence_bias),
            "confidence_min": float(args.confidence_min),
            "confidence_max": float(args.confidence_max),
            "face_embedding_dim": int(args.face_embedding_dim),
            "face_embedding_rows": int(0 if face_lut is None else face_lut.size + 1),
            "face_embedding_train_fit_unique_faces": int(0 if face_lut is None else face_lut.size),
            "checkpoint": str(output_dir / "v184_surface_conditioned_unet.pt"),
        },
        "policy_val": policy_val,
        "policy_val_all_axis_pass": bool(all_axis),
        "target_apply": target_apply,
        "gt_usage_audit": gt_usage_audit,
        "references": {
            "phasej_flowers_gate": "20.304358 / 0.557770 / 0.329222",
            "v183_flowers_exact": "19.832029 / 0.505779 / 0.405907",
        },
    }
    payload["output_json"] = str(output_dir / "v184_surface_conditioned_unet_report.json")
    _write_json(output_dir / "v184_surface_conditioned_unet_report.json", payload)
    _write_md(output_dir / "v184_surface_conditioned_unet_report.md", payload)
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
    print("OUT", output_dir / "v184_surface_conditioned_unet_report.json", flush=True)
    print("BEST", json.dumps(policy_val["best"], indent=2, sort_keys=True), flush=True)
    print("BEST_ALL_AXIS", json.dumps(policy_val.get("best_all_axis"), indent=2, sort_keys=True), flush=True)
    if target_apply is not None:
        print("TARGET", json.dumps({k: v for k, v in target_apply.items() if k != "per_view"}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
