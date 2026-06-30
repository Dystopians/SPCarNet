#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


DEFAULT_EVIDENCE = "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence"
DEFAULT_TARGET_NO_GT = (
    "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/"
    "flowers/target_evidence_no_gt"
)
DEFAULT_TARGET_EVAL = (
    "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/"
    "flowers/target_evidence_reparented"
)
PHASEJ_FLOWERS = {"psnr": 20.304358, "ssim": 0.557770, "lpips": 0.329222}
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


def _tail(values: list[float], fraction: float = 0.10) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "cvar": 0.0, "p10": 0.0}
    arr = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(math.ceil(float(fraction) * arr.size)))
    return {"min": float(arr[0]), "cvar": float(np.mean(arr[:count])), "p10": float(np.quantile(arr, fraction))}


def _parse_float_grid(text: str, fallback: float) -> list[float]:
    values = [float(x) for x in str(text or "").split(",") if str(x).strip()]
    if not values:
        values = [float(fallback)]
    return sorted({float(x) for x in values})


def _luma(chw: np.ndarray) -> np.ndarray:
    return (
        0.299 * np.asarray(chw[0], dtype=np.float32)
        + 0.587 * np.asarray(chw[1], dtype=np.float32)
        + 0.114 * np.asarray(chw[2], dtype=np.float32)
    ).astype(np.float32)


def _gradient_magnitude_2d(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return np.sqrt(np.square(gx) + np.square(gy)).astype(np.float32)


def _structure_gate_map(
    parent: np.ndarray,
    final_delta: np.ndarray,
    *,
    mode: str,
    strength: float,
    floor: float,
    eps: float,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    h, w = int(parent.shape[1]), int(parent.shape[2])
    if str(mode) == "none" or float(strength) <= 0.0:
        return np.ones((h, w), dtype=np.float32), {
            "enabled": False,
            "mode": str(mode),
            "strength": float(strength),
            "floor": float(floor),
            "eps": float(eps),
            "mean": 1.0,
            "active_mean": 1.0,
        }
    if str(mode) != "parent_luma_gradient":
        raise ValueError(f"unknown apply_gate_mode={mode}")

    parent_luma = _luma(parent)
    delta_luma = _luma(final_delta)
    parent_edge = _gradient_magnitude_2d(parent_luma)
    delta_edge = _gradient_magnitude_2d(delta_luma)
    denom = np.maximum(parent_edge + float(eps), float(eps))
    risk = (delta_edge / denom) + 0.35 * (np.abs(delta_luma) / denom)
    risk = np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0e6, neginf=0.0), 0.0, 1.0e6)
    gate = float(floor) + (1.0 - float(floor)) * np.exp(-float(strength) * risk)
    gate = np.clip(np.nan_to_num(gate, nan=float(floor), posinf=1.0, neginf=float(floor)), float(floor), 1.0).astype(
        np.float32
    )
    if active_mask is not None:
        gate = np.where(np.asarray(active_mask, dtype=bool), gate, 1.0).astype(np.float32)
    active_values = gate[np.asarray(active_mask, dtype=bool)] if active_mask is not None and np.any(active_mask) else gate
    return gate, {
        "enabled": True,
        "mode": str(mode),
        "strength": float(strength),
        "floor": float(floor),
        "eps": float(eps),
        "mean": float(np.mean(gate)),
        "active_mean": float(np.mean(active_values)) if active_values.size else 1.0,
        "active_p10": float(np.quantile(active_values, 0.10)) if active_values.size else 1.0,
        "active_p50": float(np.quantile(active_values, 0.50)) if active_values.size else 1.0,
        "active_p90": float(np.quantile(active_values, 0.90)) if active_values.size else 1.0,
    }


def _apply_delta(
    parent: np.ndarray,
    delta: np.ndarray,
    *,
    alpha: float,
    confidence: np.ndarray | None,
    confidence_threshold: float,
    apply_gate_mode: str,
    apply_gate_strength: float,
    apply_gate_floor: float,
    apply_gate_eps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    final_delta = float(alpha) * np.asarray(delta, dtype=np.float32)
    if confidence is not None and float(confidence_threshold) > 0.0:
        keep = np.asarray(confidence, dtype=np.float32) >= float(confidence_threshold)
        final_delta = np.where(keep.reshape(1, keep.shape[0], keep.shape[1]), final_delta, 0.0).astype(np.float32)
    active_mask = np.any(np.abs(final_delta) > 1.0e-8, axis=0)
    gate, gate_summary = _structure_gate_map(
        np.asarray(parent, dtype=np.float32),
        final_delta,
        mode=str(apply_gate_mode),
        strength=float(apply_gate_strength),
        floor=float(apply_gate_floor),
        eps=float(apply_gate_eps),
        active_mask=active_mask,
    )
    applied_delta = final_delta * gate.reshape(1, gate.shape[0], gate.shape[1])
    adapted = np.clip(np.asarray(parent, dtype=np.float32) + applied_delta, 0.0, 1.0).astype(np.float32)
    gate_summary["confidence_threshold"] = float(confidence_threshold)
    gate_summary["confidence_keep_fraction"] = (
        float(np.mean(np.asarray(confidence, dtype=np.float32) >= float(confidence_threshold)))
        if confidence is not None and float(confidence_threshold) > 0.0
        else 1.0
    )
    return adapted, applied_delta, gate_summary


def _verify_target_no_gt(target_dir: Path) -> dict[str, Any]:
    paths = evidence_views(target_dir)
    leaked: dict[str, list[str]] = {}
    checked = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
        forbidden = sorted(keys & FORBIDDEN_TARGET_KEYS)
        if forbidden:
            leaked[str(path)] = forbidden
        checked += 1
    return {
        "target_dir": str(target_dir),
        "checked_views": int(checked),
        "pass": not leaked and checked > 0,
        "leaked": leaked,
        "forbidden_key_set": sorted(FORBIDDEN_TARGET_KEYS),
    }


def _feature_dim(feature_mode: str) -> int:
    if str(feature_mode) == "basic":
        return 18
    if str(feature_mode) == "fourier_v1":
        return 49
    raise ValueError(f"unknown feature_mode={feature_mode}")


def _load_feature_rows(
    z: np.lib.npyio.NpzFile,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    feature_mode: str = "basic",
) -> np.ndarray:
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    normal = np.asarray(z["normal"], dtype=np.float32)
    render = np.asarray(z["rgb_render"], dtype=np.float32)
    depth = np.asarray(z["depth"], dtype=np.float32)
    alpha = np.asarray(z["alpha"], dtype=np.float32)
    camera = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
    camera = camera / max(float(np.linalg.norm(camera)), 1.0e-8)

    u = np.clip(bary[1, ys, xs], 0.0, 1.0).reshape(-1, 1)
    v = np.clip(bary[2, ys, xs], 0.0, 1.0).reshape(-1, 1)
    n = np.stack([normal[0, ys, xs], normal[1, ys, xs], normal[2, ys, xs]], axis=1)
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-8)
    cam = np.repeat(camera.reshape(1, 3), int(ys.size), axis=0).astype(np.float32)
    ndot = np.sum(n * cam, axis=1, keepdims=True).astype(np.float32)
    parent = np.stack([render[0, ys, xs], render[1, ys, xs], render[2, ys, xs]], axis=1)
    parent = np.clip(np.nan_to_num(parent, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    inv_depth = (1.0 / (1.0 + np.maximum(depth[ys, xs].reshape(-1, 1), 0.0))).astype(np.float32)
    a = np.clip(np.nan_to_num(alpha[ys, xs].reshape(-1, 1), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    base = np.concatenate(
        [
            np.ones((int(ys.size), 1), dtype=np.float32),
            u,
            v,
            u * u,
            v * v,
            u * v,
            n,
            cam,
            ndot,
            parent,
            inv_depth,
            a.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if str(feature_mode) == "basic":
        return base
    if str(feature_mode) != "fourier_v1":
        raise ValueError(f"unknown feature_mode={feature_mode}")

    w = np.clip(bary[0, ys, xs], 0.0, 1.0).reshape(-1, 1)
    coords = np.concatenate([w, u, v], axis=1).astype(np.float32)
    extra: list[np.ndarray] = [w.astype(np.float32)]
    for freq in (1.0, 2.0, 4.0, 8.0):
        angle = coords * float(2.0 * math.pi * freq)
        extra.append(np.sin(angle).astype(np.float32))
        extra.append(np.cos(angle).astype(np.float32))
    luma = (0.299 * parent[:, 0] + 0.587 * parent[:, 1] + 0.114 * parent[:, 2]).reshape(-1, 1)
    extra.extend(
        [
            (n * cam).astype(np.float32),
            np.abs(ndot).astype(np.float32),
            np.square(ndot).astype(np.float32),
            luma.astype(np.float32),
        ]
    )
    return np.concatenate([base, *extra], axis=1).astype(np.float32)


def _face_indices(faces: np.ndarray, candidate_faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(candidate_faces, faces)
    inside = (pos >= 0) & (pos < int(candidate_faces.size))
    ok = np.zeros_like(inside, dtype=bool)
    ok[inside] = candidate_faces[pos[inside]] == faces[inside]
    return pos.astype(np.int64), ok


def _valid_mask(
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray | None,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
) -> np.ndarray:
    face_id = np.asarray(z["face_id"], dtype=np.int64)
    valid = face_id >= 0
    if "barycentric_valid" in z:
        valid &= np.asarray(z["barycentric_valid"]).astype(bool)
    if "alpha" in z:
        valid &= np.asarray(z["alpha"], dtype=np.float32) >= float(min_alpha)
    if residual_l1_key in z:
        valid &= np.asarray(z[residual_l1_key], dtype=np.float32) >= float(min_l1)
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    valid &= np.all(np.isfinite(bary), axis=0)
    valid &= np.all(bary >= -0.05, axis=0)
    valid &= np.all(bary <= 1.05, axis=0)
    if candidate_faces is not None:
        valid &= np.isin(face_id, candidate_faces)
    return valid


def _policy_split(paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    fit, val = [], []
    for idx, path in enumerate(paths):
        if int(stride) > 1 and idx % int(stride) == 0:
            val.append(path)
        else:
            fit.append(path)
    return fit, val


def _rank_candidate_faces(
    fit_paths: list[Path],
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_faces: int,
    max_samples_per_view: int,
    target_energy_coverage: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    total_samples = 0
    total_score = 0.0
    for path in tqdm(fit_paths, desc="rank train-fit faces"):
        z = np.load(path)
        mask = _valid_mask(z, None, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys, xs = ys[take], xs[take]
        faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        if residual_rgb_key in z:
            residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
            score = np.sum(np.square(residual[:, ys, xs]), axis=0).astype(np.float64)
        else:
            score = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs].astype(np.float64)
        score = np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        total_score += float(np.sum(score))
        total_samples += int(faces.size)
        for face in np.unique(faces):
            fm = faces == int(face)
            sums[int(face)] = sums.get(int(face), 0.0) + float(np.sum(score[fm]))
            counts[int(face)] = counts.get(int(face), 0) + int(np.count_nonzero(fm))
    ranked = sorted(sums, key=lambda f: sums[f], reverse=True)
    coverage_target = float(target_energy_coverage)
    selected_score = float(sum(sums[f] for f in ranked))
    if 0.0 < coverage_target < 1.0 and total_score > 0.0:
        selected: list[int] = []
        running = 0.0
        for face in ranked:
            selected.append(int(face))
            running += float(sums[face])
            if running / max(total_score, 1.0e-12) >= coverage_target:
                break
        ranked = selected
        selected_score = running
    if int(max_faces) > 0 and len(ranked) > int(max_faces):
        ranked = ranked[: int(max_faces)]
        selected_score = float(sum(sums[f] for f in ranked))
    faces = np.asarray(sorted(ranked), dtype=np.int64)
    return faces, {
        "ranked_faces": int(len(sums)),
        "selected_faces": int(faces.size),
        "total_sampled_pixels": int(total_samples),
        "total_rank_score": float(total_score),
        "selected_rank_score": float(selected_score),
        "selected_score_coverage": float(selected_score / max(total_score, 1.0e-12)),
        "max_faces": int(max_faces),
        "target_energy_coverage": float(target_energy_coverage),
    }


class SurfaceResidualDecoder(torch.nn.Module):
    def __init__(
        self,
        face_count: int,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        max_delta: float,
        *,
        predict_confidence: bool = False,
        confidence_floor: float = 0.0,
    ):
        super().__init__()
        self.face_embedding = torch.nn.Embedding(int(face_count), int(embedding_dim))
        self.predict_confidence = bool(predict_confidence)
        self.confidence_floor = float(np.clip(float(confidence_floor), 0.0, 1.0))
        out_dim = 4 if self.predict_confidence else 3
        dims = [int(feature_dim) + int(embedding_dim)] + [int(hidden_dim)] * int(layers) + [out_dim]
        blocks: list[torch.nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1], strict=False):
            blocks += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        blocks.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.net = torch.nn.Sequential(*blocks)
        self.max_delta = float(max_delta)

    def forward_with_confidence(self, face_idx: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.face_embedding(face_idx)
        raw = self.net(torch.cat([features, emb], dim=1))
        residual = torch.tanh(raw[:, :3]) * self.max_delta
        if not self.predict_confidence:
            confidence = torch.ones((raw.shape[0],), dtype=residual.dtype, device=residual.device)
        else:
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * torch.sigmoid(raw[:, 3])
        return residual * confidence[:, None], confidence

    def forward(self, face_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual, _confidence = self.forward_with_confidence(face_idx, features)
        return residual


def _sample_batch(
    path: Path,
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    batch_size: int,
    seed: int,
    sample_weight_gamma: float,
    sample_weight_clip: float,
    confidence_target_mode: str,
    confidence_gain_floor: float,
    confidence_gain_scale: float,
    sample_weight_confidence_power: float,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    z = np.load(path)
    mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise RuntimeError(f"no valid train samples in {path}")
    residual_score = None
    if residual_l1_key in z:
        residual_score = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
        residual_score = np.clip(np.nan_to_num(residual_score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if ys.size > int(batch_size):
        probs = None
        if residual_score is not None and float(sample_weight_gamma) > 0.0:
            score = np.power(residual_score.astype(np.float64) + 1.0e-6, float(sample_weight_gamma))
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            total = float(np.sum(score))
            if total > 0.0 and np.isfinite(total):
                probs = score / total
                probs = probs / max(float(np.sum(probs)), 1.0e-12)
        take = rng.choice(ys.size, size=int(batch_size), replace=False, p=probs)
        ys, xs = ys[take], xs[take]
        if residual_score is not None:
            residual_score = residual_score[take]
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _face_indices(faces, candidate_faces)
    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
    if residual_score is not None:
        residual_score = residual_score[ok]
    features = _load_feature_rows(z, ys, xs, feature_mode=str(feature_mode))
    residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
    target = np.stack([residual[0, ys, xs], residual[1, ys, xs], residual[2, ys, xs]], axis=1).astype(np.float32)
    confidence_mode = str(confidence_target_mode)
    if confidence_mode == "gain_soft" and "teacher_gain_l1" in z:
        gain = np.asarray(z["teacher_gain_l1"], dtype=np.float32)[ys, xs]
        confidence_target = np.clip(
            (gain - float(confidence_gain_floor)) / max(float(confidence_gain_scale), 1.0e-6),
            0.0,
            1.0,
        ).astype(np.float32)
    elif confidence_mode == "gain_binary" and "teacher_gain_l1" in z:
        confidence_target = (np.asarray(z["teacher_gain_l1"], dtype=np.float32)[ys, xs] > float(confidence_gain_floor)).astype(
            np.float32
        )
    elif "teacher_better_mask" in z:
        confidence_target = np.asarray(z["teacher_better_mask"], dtype=np.float32)[ys, xs]
    elif "teacher_gain_l1" in z:
        confidence_target = (np.asarray(z["teacher_gain_l1"], dtype=np.float32)[ys, xs] > 0.0).astype(np.float32)
    else:
        confidence_target = np.ones((int(ys.size),), dtype=np.float32)
    if residual_score is None:
        residual_score = np.mean(np.abs(target), axis=1).astype(np.float32)
    denom = max(float(np.mean(residual_score)), 1.0e-6)
    weights = np.power((residual_score / denom) + 1.0e-6, max(float(sample_weight_gamma), 0.0)).astype(np.float32)
    if float(sample_weight_clip) > 0.0:
        weights = np.clip(weights, 1.0 / float(sample_weight_clip), float(sample_weight_clip))
    if float(sample_weight_confidence_power) > 0.0:
        confidence_weight = np.power(
            np.clip(confidence_target.astype(np.float32), 0.02, 1.0),
            float(sample_weight_confidence_power),
        ).astype(np.float32)
        weights = weights * confidence_weight
    weights = weights / max(float(np.mean(weights)), 1.0e-6)
    return (
        face_idx.astype(np.int64),
        features.astype(np.float32),
        target,
        weights.astype(np.float32),
        np.clip(confidence_target.astype(np.float32), 0.0, 1.0),
    )


def _image_proxy_loss(
    model: SurfaceResidualDecoder,
    path: Path,
    candidate_faces: np.ndarray,
    *,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    stride: int,
    feature_mode: str,
    device: torch.device,
) -> torch.Tensor:
    z = np.load(path)
    mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
    mask = mask[:: int(stride), :: int(stride)]
    ys_lr, xs_lr = np.nonzero(mask)
    if ys_lr.size == 0:
        return torch.zeros((), device=device)
    ys = ys_lr * int(stride)
    xs = xs_lr * int(stride)
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    face_idx, ok = _face_indices(faces, candidate_faces)
    if not np.any(ok):
        return torch.zeros((), device=device)
    ys_lr, xs_lr, ys, xs, face_idx = ys_lr[ok], xs_lr[ok], ys[ok], xs[ok], face_idx[ok]
    features = torch.from_numpy(_load_feature_rows(z, ys, xs, feature_mode=str(feature_mode))).to(device)
    face_t = torch.from_numpy(face_idx.astype(np.int64)).to(device)
    pred = model(face_t, features)

    parent_np = np.asarray(z["rgb_render"], dtype=np.float32)[:, :: int(stride), :: int(stride)]
    residual_np = np.asarray(z[residual_rgb_key], dtype=np.float32)[:, :: int(stride), :: int(stride)]
    parent = torch.from_numpy(parent_np).to(device)
    target_img = torch.clamp(parent + torch.from_numpy(residual_np).to(device), 0.0, 1.0)
    adapted = parent.clone()
    adapted[:, torch.from_numpy(ys_lr).to(device), torch.from_numpy(xs_lr).to(device)] = torch.clamp(
        adapted[:, torch.from_numpy(ys_lr).to(device), torch.from_numpy(xs_lr).to(device)] + pred.T,
        0.0,
        1.0,
    )
    l1 = torch.mean(torch.abs(adapted - target_img))
    ssim_loss = 1.0 - ssim(adapted.unsqueeze(0), target_img.unsqueeze(0))
    lum_a = 0.299 * adapted[0] + 0.587 * adapted[1] + 0.114 * adapted[2]
    lum_t = 0.299 * target_img[0] + 0.587 * target_img[1] + 0.114 * target_img[2]
    grad_a = torch.abs(lum_a[:, 1:] - lum_a[:, :-1]).mean() + torch.abs(lum_a[1:, :] - lum_a[:-1, :]).mean()
    grad_t = torch.abs(lum_t[:, 1:] - lum_t[:, :-1]).mean() + torch.abs(lum_t[1:, :] - lum_t[:-1, :]).mean()
    edge = torch.abs(grad_a - grad_t)
    return l1 + 0.25 * ssim_loss + 0.5 * edge


def _evaluate(
    model: SurfaceResidualDecoder,
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    alpha_grid: list[float],
    apply_confidence_threshold_grid: list[float],
    apply_gate_mode: str,
    apply_gate_strength_grid: list[float],
    apply_gate_floor: float,
    apply_gate_eps: float,
    chunk_size: int,
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    policy_grid = [
        (float(a), float(g), float(t))
        for a in alpha_grid
        for g in apply_gate_strength_grid
        for t in apply_confidence_threshold_grid
    ]
    rows_by_policy: dict[str, list[dict[str, Any]]] = {
        f"{a:.8g}|{g:.8g}|{t:.8g}": [] for a, g, t in policy_grid
    }
    if output_dir is not None:
        (output_dir / "renders").mkdir(parents=True, exist_ok=True)
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for path in tqdm(val_paths, desc="policy-val neural decoder"):
            z = np.load(path)
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
            ys, xs = np.nonzero(mask)
            delta = np.zeros((3, parent.shape[1], parent.shape[2]), dtype=np.float32)
            confidence = np.zeros((parent.shape[1], parent.shape[2]), dtype=np.float32)
            if ys.size:
                faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
                face_idx, ok = _face_indices(faces, candidate_faces)
                ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
                for start in range(0, int(ys.size), int(chunk_size)):
                    end = min(int(ys.size), start + int(chunk_size))
                    feat = torch.from_numpy(
                        _load_feature_rows(z, ys[start:end], xs[start:end], feature_mode=str(feature_mode))
                    ).to(device)
                    face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                    pred_t, conf_t = model.forward_with_confidence(face_t, feat)
                    pred = pred_t.detach().cpu().numpy().astype(np.float32)
                    conf = conf_t.detach().cpu().numpy().astype(np.float32)
                    delta[:, ys[start:end], xs[start:end]] = pred.T
                    confidence[ys[start:end], xs[start:end]] = conf
            p_psnr = _psnr(parent, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
            for alpha, gate_strength, confidence_threshold in policy_grid:
                adapted, applied_delta, gate_summary = _apply_delta(
                    parent,
                    delta,
                    alpha=float(alpha),
                    confidence=confidence,
                    confidence_threshold=float(confidence_threshold),
                    apply_gate_mode=str(apply_gate_mode),
                    apply_gate_strength=float(gate_strength),
                    apply_gate_floor=float(apply_gate_floor),
                    apply_gate_eps=float(apply_gate_eps),
                )
                c_psnr = _psnr(adapted, gt)
                c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
                row = {
                    "view": path.stem,
                    "alpha": float(alpha),
                    "apply_gate_strength": float(gate_strength),
                    "apply_confidence_threshold": float(confidence_threshold),
                    "apply_confidence_keep_fraction": float(gate_summary.get("confidence_keep_fraction", 1.0)),
                    "apply_gate_active_mean": float(gate_summary.get("active_mean", 1.0)),
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                    "changed_fraction": float(np.mean(np.any(np.abs(applied_delta) > (0.5 / 255.0), axis=0))),
                }
                if compute_lpips:
                    row.update(
                        {
                            "parent_lpips": float(p_lp),
                            "candidate_lpips": float(c_lp),
                            "lpips_gain": float(p_lp - c_lp),
                        }
                    )
                rows_by_policy[f"{float(alpha):.8g}|{float(gate_strength):.8g}|{float(confidence_threshold):.8g}"].append(row)
    summaries: list[dict[str, Any]] = []
    for _policy_key, rows in rows_by_policy.items():
        parent_psnr = [r["parent_psnr"] for r in rows]
        cand_psnr = [r["candidate_psnr"] for r in rows]
        parent_ssim = [r["parent_ssim"] for r in rows]
        cand_ssim = [r["candidate_ssim"] for r in rows]
        psnr_gain = [r["psnr_gain"] for r in rows]
        ssim_gain = [r["ssim_gain"] for r in rows]
        alpha = float(rows[0]["alpha"]) if rows else 0.0
        gate_strength = float(rows[0]["apply_gate_strength"]) if rows else 0.0
        confidence_threshold = float(rows[0]["apply_confidence_threshold"]) if rows else 0.0
        summary = {
            "alpha": float(alpha),
            "apply_confidence_threshold": float(confidence_threshold),
            "apply_gate_mode": str(apply_gate_mode),
            "apply_gate_strength": float(gate_strength),
            "apply_gate_floor": float(apply_gate_floor),
            "apply_gate_eps": float(apply_gate_eps),
            "parent_psnr": float(np.mean(parent_psnr)),
            "candidate_psnr": float(np.mean(cand_psnr)),
            "psnr_gain": float(np.mean(psnr_gain)),
            "parent_ssim": float(np.mean(parent_ssim)),
            "candidate_ssim": float(np.mean(cand_ssim)),
            "ssim_gain": float(np.mean(ssim_gain)),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
            "mean_changed_fraction": float(np.mean([r["changed_fraction"] for r in rows])),
            "mean_apply_confidence_keep_fraction": float(np.mean([r["apply_confidence_keep_fraction"] for r in rows])),
            "mean_apply_gate_active": float(np.mean([r["apply_gate_active_mean"] for r in rows])),
            "per_view": rows,
        }
        if compute_lpips:
            parent_lpips = [r["parent_lpips"] for r in rows]
            cand_lpips = [r["candidate_lpips"] for r in rows]
            lpips_gain = [r["lpips_gain"] for r in rows]
            summary.update(
                {
                    "parent_lpips": float(np.mean(parent_lpips)),
                    "candidate_lpips": float(np.mean(cand_lpips)),
                    "lpips_gain": float(np.mean(lpips_gain)),
                    "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)),
                }
            )
        summaries.append(summary)
    best = max(
        summaries,
        key=lambda r: (
            float(r.get("psnr_gain", 0.0)) + 20.0 * float(r.get("ssim_gain", 0.0)) + 20.0 * float(r.get("lpips_gain", 0.0))
        ),
    )
    best_all_axis = None
    for row in summaries:
        if (
            float(row.get("psnr_gain", 0.0)) > 0.0
            and float(row.get("ssim_gain", 0.0)) > 0.0
            and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
        ):
            score = (
                float(row.get("psnr_gain", 0.0))
                + 20.0 * float(row.get("ssim_gain", 0.0))
                + 20.0 * float(row.get("lpips_gain", 0.0))
            )
            cand = {k: v for k, v in row.items() if k != "per_view"}
            cand["balanced_score"] = float(score)
            if best_all_axis is None or score > float(best_all_axis.get("balanced_score", -1.0)):
                best_all_axis = cand
    if output_dir is not None:
        best_alpha = float(best["alpha"])
        best_confidence_threshold = float(best.get("apply_confidence_threshold", apply_confidence_threshold_grid[0]))
        best_gate_strength = float(best.get("apply_gate_strength", apply_gate_strength_grid[0]))
        with torch.no_grad():
            for path in tqdm(val_paths, desc="write best policy-val renders"):
                z = np.load(path)
                parent = np.asarray(z["rgb_render"], dtype=np.float32)
                gt = np.asarray(z["rgb_gt"], dtype=np.float32)
                mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
                ys, xs = np.nonzero(mask)
                delta = np.zeros_like(parent, dtype=np.float32)
                confidence = np.zeros((parent.shape[1], parent.shape[2]), dtype=np.float32)
                if ys.size:
                    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
                    face_idx, ok = _face_indices(faces, candidate_faces)
                    ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
                    for start in range(0, int(ys.size), int(chunk_size)):
                        end = min(int(ys.size), start + int(chunk_size))
                        feat = torch.from_numpy(
                            _load_feature_rows(z, ys[start:end], xs[start:end], feature_mode=str(feature_mode))
                        ).to(device)
                        face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                        pred_t, conf_t = model.forward_with_confidence(face_t, feat)
                        pred = pred_t.detach().cpu().numpy().astype(np.float32)
                        conf = conf_t.detach().cpu().numpy().astype(np.float32)
                        delta[:, ys[start:end], xs[start:end]] = pred.T
                        confidence[ys[start:end], xs[start:end]] = conf
                adapted, _applied_delta, _gate_summary = _apply_delta(
                    parent,
                    delta,
                    alpha=float(best_alpha),
                    confidence=confidence,
                    confidence_threshold=float(best_confidence_threshold),
                    apply_gate_mode=str(apply_gate_mode),
                    apply_gate_strength=float(best_gate_strength),
                    apply_gate_floor=float(apply_gate_floor),
                    apply_gate_eps=float(apply_gate_eps),
                )
                save_image_chw(output_dir / "renders" / f"{path.stem}.png", adapted)
                save_image_chw(output_dir / "gt" / f"{path.stem}.png", gt)
    return {
        "best": {k: v for k, v in best.items() if k != "per_view"},
        "best_all_axis": best_all_axis,
        "rows": [{k: v for k, v in row.items() if k != "per_view"} for row in summaries],
        "per_view_by_policy": {str(k): v for k, v in rows_by_policy.items()},
    }


def _predict_delta_image(
    model: SurfaceResidualDecoder,
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    chunk_size: int,
    device: torch.device,
    return_confidence: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
    ys, xs = np.nonzero(mask)
    delta = np.zeros_like(parent, dtype=np.float32)
    confidence = np.zeros((parent.shape[1], parent.shape[2]), dtype=np.float32)
    active_count = 0
    if ys.size:
        faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        face_idx, ok = _face_indices(faces, candidate_faces)
        ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
        active_count = int(ys.size)
        with torch.no_grad():
            for start in range(0, int(ys.size), int(chunk_size)):
                end = min(int(ys.size), start + int(chunk_size))
                feat = torch.from_numpy(
                    _load_feature_rows(z, ys[start:end], xs[start:end], feature_mode=str(feature_mode))
                ).to(device)
                face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                pred_t, conf_t = model.forward_with_confidence(face_t, feat)
                pred = pred_t.detach().cpu().numpy().astype(np.float32)
                conf = conf_t.detach().cpu().numpy().astype(np.float32)
                delta[:, ys[start:end], xs[start:end]] = pred.T
                confidence[ys[start:end], xs[start:end]] = conf
    active_fraction = float(active_count / max(int(parent.shape[1] * parent.shape[2]), 1))
    if return_confidence:
        return delta, active_fraction, confidence
    return delta, active_fraction


def _target_exact_eval(
    model: SurfaceResidualDecoder,
    target_evidence_dir: Path,
    target_eval_evidence_dir: Path,
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    alpha: float,
    apply_confidence_threshold: float,
    apply_gate_mode: str,
    apply_gate_strength: float,
    apply_gate_floor: float,
    apply_gate_eps: float,
    chunk_size: int,
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    apply_paths = evidence_views(target_evidence_dir)
    eval_paths = {path.stem: path for path in evidence_views(target_eval_evidence_dir)}
    lpips_model = build_lpips_model() if compute_lpips else None
    rows: list[dict[str, Any]] = []
    render_dir = None
    if output_dir is not None:
        render_dir = output_dir / "target_exact_fixed_policy"
        render_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for apply_path in tqdm(apply_paths, desc="target exact neural decoder"):
        if apply_path.stem not in eval_paths:
            raise FileNotFoundError(f"missing target eval view for {apply_path.stem}")
        with np.load(apply_path, allow_pickle=False) as z_apply, np.load(eval_paths[apply_path.stem], allow_pickle=False) as z_eval:
            leaked = sorted(set(z_apply.files) & FORBIDDEN_TARGET_KEYS)
            if leaked:
                raise RuntimeError(f"target apply evidence leaks GT keys for {apply_path}: {leaked}")
            parent = np.asarray(z_apply["rgb_render"], dtype=np.float32)
            gt = np.asarray(z_eval["rgb_gt"], dtype=np.float32)
            delta, active_fraction, confidence = _predict_delta_image(
                model,
                z_apply,
                candidate_faces,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                feature_mode=str(feature_mode),
                chunk_size=int(chunk_size),
                device=device,
                return_confidence=True,
            )
            adapted, applied_delta, gate_summary = _apply_delta(
                parent,
                delta,
                alpha=float(alpha),
                confidence=confidence,
                confidence_threshold=float(apply_confidence_threshold),
                apply_gate_mode=str(apply_gate_mode),
                apply_gate_strength=float(apply_gate_strength),
                apply_gate_floor=float(apply_gate_floor),
                apply_gate_eps=float(apply_gate_eps),
            )
            p_psnr = _psnr(parent, gt)
            c_psnr = _psnr(adapted, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
            row = {
                "view": apply_path.stem,
                "parent_psnr": float(p_psnr),
                "candidate_psnr": float(c_psnr),
                "psnr_gain": float(c_psnr - p_psnr),
                "parent_ssim": float(p_ssim),
                "candidate_ssim": float(c_ssim),
                "ssim_gain": float(c_ssim - p_ssim),
                "active_fraction": float(active_fraction),
                "changed_fraction": float(np.mean(np.any(np.abs(applied_delta) > (0.5 / 255.0), axis=0))),
                "apply_confidence_threshold": float(apply_confidence_threshold),
                "apply_confidence_keep_fraction": float(gate_summary.get("confidence_keep_fraction", 1.0)),
                "apply_gate_active_mean": float(gate_summary.get("active_mean", 1.0)),
            }
            if compute_lpips:
                p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model)
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model)
                row.update(
                    {
                        "parent_lpips": float(p_lp),
                        "candidate_lpips": float(c_lp),
                        "lpips_gain": float(p_lp - c_lp),
                    }
                )
            rows.append(row)
            if render_dir is not None:
                save_image_chw(render_dir / f"{apply_path.stem}.png", adapted)
                save_image_chw(render_dir / f"{apply_path.stem}_parent.png", parent)
                save_image_chw(render_dir / f"{apply_path.stem}_gt.png", gt)

    psnr_gain = [float(r["psnr_gain"]) for r in rows]
    ssim_gain = [float(r["ssim_gain"]) for r in rows]
    summary: dict[str, Any] = {
        "view_count": int(len(rows)),
        "parent_psnr": float(np.mean([float(r["parent_psnr"]) for r in rows])) if rows else 0.0,
        "candidate_psnr": float(np.mean([float(r["candidate_psnr"]) for r in rows])) if rows else 0.0,
        "psnr_gain": float(np.mean(psnr_gain)) if psnr_gain else 0.0,
        "psnr_gain_tail": _tail(psnr_gain),
        "parent_ssim": float(np.mean([float(r["parent_ssim"]) for r in rows])) if rows else 0.0,
        "candidate_ssim": float(np.mean([float(r["candidate_ssim"]) for r in rows])) if rows else 0.0,
        "ssim_gain": float(np.mean(ssim_gain)) if ssim_gain else 0.0,
        "ssim_gain_tail": _tail(ssim_gain),
        "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)) if rows else 0.0,
        "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)) if rows else 0.0,
        "mean_active_fraction": float(np.mean([float(r["active_fraction"]) for r in rows])) if rows else 0.0,
        "mean_changed_fraction": float(np.mean([float(r["changed_fraction"]) for r in rows])) if rows else 0.0,
        "mean_apply_confidence_keep_fraction": (
            float(np.mean([float(r["apply_confidence_keep_fraction"]) for r in rows])) if rows else 1.0
        ),
        "mean_apply_gate_active": float(np.mean([float(r["apply_gate_active_mean"]) for r in rows])) if rows else 1.0,
    }
    if compute_lpips:
        lpips_gain = [float(r["lpips_gain"]) for r in rows]
        summary.update(
            {
                "parent_lpips": float(np.mean([float(r["parent_lpips"]) for r in rows])) if rows else 0.0,
                "candidate_lpips": float(np.mean([float(r["candidate_lpips"]) for r in rows])) if rows else 0.0,
                "lpips_gain": float(np.mean(lpips_gain)) if lpips_gain else 0.0,
                "lpips_gain_tail": _tail(lpips_gain),
                "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)) if rows else 0.0,
            }
        )
    comparison = {
        "reference": PHASEJ_FLOWERS,
        "candidate_psnr_minus_phasej": float(summary.get("candidate_psnr", 0.0) - PHASEJ_FLOWERS["psnr"]),
        "candidate_ssim_minus_phasej": float(summary.get("candidate_ssim", 0.0) - PHASEJ_FLOWERS["ssim"]),
        "phasej_lpips_minus_candidate": (
            float(PHASEJ_FLOWERS["lpips"] - summary.get("candidate_lpips", 0.0)) if compute_lpips else None
        ),
        "beats_phasej_all_axis_under_reported_metric_scale": bool(
            summary.get("candidate_psnr", 0.0) > PHASEJ_FLOWERS["psnr"]
            and summary.get("candidate_ssim", 0.0) > PHASEJ_FLOWERS["ssim"]
            and (not compute_lpips or summary.get("candidate_lpips", 1.0e9) < PHASEJ_FLOWERS["lpips"])
        ),
    }
    return {
        "alpha": float(alpha),
        "apply_confidence_threshold": float(apply_confidence_threshold),
        "apply_gate_mode": str(apply_gate_mode),
        "apply_gate_strength": float(apply_gate_strength),
        "apply_gate_floor": float(apply_gate_floor),
        "apply_gate_eps": float(apply_gate_eps),
        "target_evidence_dir": str(target_evidence_dir),
        "target_eval_evidence_dir": str(target_eval_evidence_dir),
        "render_dir": str(render_dir) if render_dir is not None else "",
        "summary": summary,
        "phasej_reference_comparison": comparison,
        "per_view": rows,
        "selection_scope": "alpha selected only from train-policy-val; target/test GT loaded after no-GT apply",
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all_axis = payload["policy_val"].get("best_all_axis")
    lines = [
        "# v275 Learned Surface Feature Decoder Audit",
        "",
        f"- teacher signal pass: `{payload['teacher_signal_pass']}`",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- no-target-GT audit pass: `{payload.get('target_no_gt_audit', {}).get('pass')}`",
        f"- target exact fixed-policy pass vs parent: `{payload.get('target_exact_eval', {}).get('pass_vs_parent_all_axis')}`",
        f"- flowers exact Phase-J gate pass: `{payload.get('flowers_exact_phasej_gate_pass')}`",
        f"- Phase-J flowers reference: `{PHASEJ_FLOWERS}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- train steps: `{payload['train']['steps']}`",
        f"- confidence head: `{payload['train'].get('confidence_head', False)}`",
        f"- confidence target: `{payload['train'].get('confidence_target_mode')}`",
        f"- apply confidence thresholds: `{payload['train'].get('apply_confidence_threshold_grid')}`",
        f"- apply gate: `{payload['train'].get('apply_gate_mode')}` grid `{payload['train'].get('apply_gate_strength_grid')}`",
        "",
        "## Best Policy-Val Row",
        "",
        "| alpha | conf th | gate | PSNR gain | SSIM gain | LPIPS gain | changed | keep | pos views | SSIM pos | LPIPS pos |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.4f} | {best.get('apply_confidence_threshold', 0.0):.4f} | "
            f"{best.get('apply_gate_strength', 0.0):.4f} | "
            f"{best.get('psnr_gain', 0.0):.6f} | "
            f"{best.get('ssim_gain', 0.0):.6f} | {best.get('lpips_gain', 0.0):.6f} | "
            f"{best.get('mean_changed_fraction', 0.0):.6f} | "
            f"{best.get('mean_apply_confidence_keep_fraction', 1.0):.6f} | "
            f"{best.get('positive_view_fraction', 0.0):.3f} | "
            f"{best.get('ssim_positive_view_fraction', 0.0):.3f} | "
            f"{best.get('lpips_positive_view_fraction', 0.0):.3f} |"
        ),
        "",
        f"- best all-axis row: `{best_all_axis}`",
        "",
    ]
    target_eval = payload.get("target_exact_eval") or {}
    if target_eval:
        summary = target_eval.get("summary", {})
        comparison = target_eval.get("phasej_reference_comparison", {})
        lines.extend(
            [
                "## Target Exact Fixed-Policy Evaluation",
                "",
                "| PSNR | SSIM | LPIPS | PSNR gain | SSIM gain | LPIPS gain | changed fraction | keep fraction |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    "| {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {pg:+.6f} | {sg:+.6f} | {lg:+.6f} | {chg:.6f} | {keep:.6f} |"
                ).format(
                    psnr=float(summary.get("candidate_psnr", 0.0)),
                    ssim=float(summary.get("candidate_ssim", 0.0)),
                    lpips=float(summary.get("candidate_lpips", 0.0)),
                    pg=float(summary.get("psnr_gain", 0.0)),
                    sg=float(summary.get("ssim_gain", 0.0)),
                    lg=float(summary.get("lpips_gain", 0.0)),
                    chg=float(summary.get("mean_changed_fraction", 0.0)),
                    keep=float(summary.get("mean_apply_confidence_keep_fraction", 1.0)),
                ),
                "",
                f"- render dir: `{target_eval.get('render_dir', '')}`",
                (
                    f"- fixed policy: alpha `{target_eval.get('alpha', 0.0)}`, "
                    f"confidence threshold `{target_eval.get('apply_confidence_threshold', 0.0)}`, "
                    f"gate `{target_eval.get('apply_gate_strength', 0.0)}`"
                ),
                f"- Phase-J comparison: `{comparison}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- best policy-val renders: `{payload['output_render_dir']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a differentiable teacher-residual surface decoder.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--target_evidence_dir", default=DEFAULT_TARGET_NO_GT)
    parser.add_argument("--target_eval_evidence_dir", default=DEFAULT_TARGET_EVAL)
    parser.add_argument("--target_eval_mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.02)
    parser.add_argument("--max_candidate_faces", type=int, default=128)
    parser.add_argument("--max_candidate_face_samples_per_view", type=int, default=4096)
    parser.add_argument("--candidate_target_energy_coverage", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--embedding_dim", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max_delta", type=float, default=0.20)
    parser.add_argument("--confidence_head", action="store_true")
    parser.add_argument("--confidence_floor", type=float, default=0.0)
    parser.add_argument("--confidence_loss_weight", type=float, default=0.10)
    parser.add_argument("--confidence_target_mode", choices=["mask", "gain_binary", "gain_soft"], default="mask")
    parser.add_argument("--confidence_gain_floor", type=float, default=0.0)
    parser.add_argument("--confidence_gain_scale", type=float, default=0.03)
    parser.add_argument("--feature_mode", choices=["basic", "fourier_v1"], default="basic")
    parser.add_argument("--image_loss_every", type=int, default=4)
    parser.add_argument("--image_loss_stride", type=int, default=12)
    parser.add_argument("--image_loss_weight", type=float, default=0.35)
    parser.add_argument("--sample_weight_gamma", type=float, default=0.0)
    parser.add_argument("--sample_weight_clip", type=float, default=8.0)
    parser.add_argument("--sample_weight_confidence_power", type=float, default=0.0)
    parser.add_argument("--cosine_loss_weight", type=float, default=0.0)
    parser.add_argument("--energy_match_weight", type=float, default=0.0)
    parser.add_argument("--mag_reg", type=float, default=1.0e-4)
    parser.add_argument("--alpha_grid", default="0,0.0625,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--apply_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--apply_confidence_threshold_grid", default="")
    parser.add_argument("--apply_gate_mode", choices=["none", "parent_luma_gradient"], default="none")
    parser.add_argument("--apply_gate_strength", type=float, default=0.0)
    parser.add_argument("--apply_gate_strength_grid", default="")
    parser.add_argument("--apply_gate_floor", type=float, default=0.0)
    parser.add_argument("--apply_gate_eps", type=float, default=0.02)
    parser.add_argument("--eval_chunk_size", type=int, default=65536)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--policy_val_ssim_max_side", type=int, default=512)
    parser.add_argument("--policy_val_lpips_max_side", type=int, default=256)
    parser.add_argument("--output_dir", default="/tmp/peilincai_spcarnet_v180_perceptual_decoder")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-v180-perceptual-decoder")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=180)
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
        except Exception as exc:  # pragma: no cover - W&B availability is environment dependent.
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)
            wandb_run = None
    paths = evidence_views(Path(args.fit_evidence_dir))
    if not paths:
        raise FileNotFoundError(args.fit_evidence_dir)
    fit_paths, val_paths = _policy_split(paths, int(args.policy_val_stride))
    init_checkpoint: dict[str, Any] | None = None
    if str(args.init_checkpoint):
        init_checkpoint = torch.load(str(args.init_checkpoint), map_location="cpu", weights_only=False)
        candidate_faces = np.asarray(init_checkpoint["candidate_faces"], dtype=np.int64)
        face_summary = {
            "source": "init_checkpoint",
            "checkpoint": str(args.init_checkpoint),
            "selected_faces": int(candidate_faces.size),
        }
    else:
        candidate_faces, face_summary = _rank_candidate_faces(
            fit_paths,
            residual_rgb_key=str(args.residual_rgb_key),
            residual_l1_key=str(args.residual_l1_key),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            max_faces=int(args.max_candidate_faces),
            max_samples_per_view=int(args.max_candidate_face_samples_per_view),
            target_energy_coverage=float(args.candidate_target_energy_coverage),
            seed=int(args.seed),
        )
    if candidate_faces.size <= 0:
        raise RuntimeError("no candidate faces selected")
    model = SurfaceResidualDecoder(
        int(candidate_faces.size),
        feature_dim=_feature_dim(str(args.feature_mode)),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        max_delta=float(args.max_delta),
        predict_confidence=bool(args.confidence_head),
        confidence_floor=float(args.confidence_floor),
    ).to(device)
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-5)
    train_rows: list[dict[str, Any]] = []
    fit_cycle = list(fit_paths)
    train_rng = random.Random(int(args.seed))
    train_steps = 0 if bool(args.skip_training) else int(args.steps)
    for step in tqdm(range(1, int(train_steps) + 1), desc="train neural surface decoder"):
        sampled = None
        path = fit_cycle[(step - 1) % len(fit_cycle)]
        for attempt in range(max(1, len(fit_cycle))):
            path = fit_cycle[(step + attempt - 1) % len(fit_cycle)]
            try:
                sampled = _sample_batch(
                    path,
                    candidate_faces,
                    residual_rgb_key=str(args.residual_rgb_key),
                    residual_l1_key=str(args.residual_l1_key),
                    min_l1=float(args.min_l1),
                    min_alpha=float(args.min_alpha),
                    batch_size=int(args.batch_size),
                    seed=int(args.seed) + step + attempt * 1009,
                    sample_weight_gamma=float(args.sample_weight_gamma),
                    sample_weight_clip=float(args.sample_weight_clip),
                    confidence_target_mode=str(args.confidence_target_mode),
                    confidence_gain_floor=float(args.confidence_gain_floor),
                    confidence_gain_scale=float(args.confidence_gain_scale),
                    sample_weight_confidence_power=float(args.sample_weight_confidence_power),
                    feature_mode=str(args.feature_mode),
                )
                break
            except RuntimeError:
                continue
        if sampled is None:
            raise RuntimeError("no train-fit view contains the selected candidate faces")
        face_idx, features, target, sample_weights, confidence_target = sampled
        face_t = torch.from_numpy(face_idx).to(device)
        feat_t = torch.from_numpy(features).to(device)
        target_t = torch.from_numpy(target).to(device)
        confidence_target_t = torch.from_numpy(confidence_target).to(device).reshape(-1)
        weight_t = torch.from_numpy(sample_weights).to(device).reshape(-1)
        weight_t = weight_t / torch.clamp(torch.mean(weight_t), min=1.0e-6)
        pred, pred_confidence = model.forward_with_confidence(face_t, feat_t)
        rgb_per = torch.sqrt(torch.square(pred - target_t) + 1.0e-6).mean(dim=1)
        rgb_loss = torch.sum(weight_t * rgb_per) / torch.clamp(torch.sum(weight_t), min=1.0e-6)
        luma_pred = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        luma_target = 0.299 * target_t[:, 0] + 0.587 * target_t[:, 1] + 0.114 * target_t[:, 2]
        luma_per = torch.sqrt(torch.square(luma_pred - luma_target) + 1.0e-6)
        luma_loss = torch.sum(weight_t * luma_per) / torch.clamp(torch.sum(weight_t), min=1.0e-6)
        cosine = F.cosine_similarity(pred, target_t, dim=1, eps=1.0e-6)
        target_mag = torch.mean(torch.abs(target_t), dim=1)
        pred_mag = torch.mean(torch.abs(pred), dim=1)
        direction_weight = weight_t * (target_mag > 1.0e-5).float()
        cosine_loss = torch.sum(direction_weight * (1.0 - cosine)) / torch.clamp(torch.sum(direction_weight), min=1.0e-6)
        energy_loss = torch.sum(weight_t * torch.sqrt(torch.square(pred_mag - target_mag) + 1.0e-6)) / torch.clamp(
            torch.sum(weight_t),
            min=1.0e-6,
        )
        confidence_loss = torch.zeros((), device=device)
        if bool(args.confidence_head):
            confidence_loss = F.binary_cross_entropy(
                torch.clamp(pred_confidence, 1.0e-6, 1.0 - 1.0e-6),
                confidence_target_t,
                weight=torch.clamp(weight_t, 0.25, 4.0),
                reduction="sum",
            ) / torch.clamp(torch.sum(torch.clamp(weight_t, 0.25, 4.0)), min=1.0e-6)
        img_loss = torch.zeros((), device=device)
        if int(args.image_loss_every) > 0 and step % int(args.image_loss_every) == 0:
            img_path = train_rng.choice(fit_cycle)
            img_loss = _image_proxy_loss(
                model,
                img_path,
                candidate_faces,
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                stride=int(args.image_loss_stride),
                feature_mode=str(args.feature_mode),
                device=device,
            )
        mag = torch.mean(torch.square(pred))
        loss = (
            rgb_loss
            + 0.35 * luma_loss
            + float(args.cosine_loss_weight) * cosine_loss
            + float(args.energy_match_weight) * energy_loss
            + float(args.confidence_loss_weight) * confidence_loss
            + float(args.image_loss_weight) * img_loss
            + float(args.mag_reg) * mag
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, int(args.steps) // 10) == 0:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "rgb_loss": float(rgb_loss.detach().cpu()),
                "luma_loss": float(luma_loss.detach().cpu()),
                "cosine_loss": float(cosine_loss.detach().cpu()),
                "energy_loss": float(energy_loss.detach().cpu()),
                "confidence_loss": float(confidence_loss.detach().cpu()),
                "mean_confidence": float(torch.mean(pred_confidence).detach().cpu()),
                "mean_confidence_target": float(torch.mean(confidence_target_t).detach().cpu()),
                "image_proxy_loss": float(img_loss.detach().cpu()),
                "mean_abs_pred": float(torch.mean(torch.abs(pred)).detach().cpu()),
                "mean_abs_target": float(torch.mean(torch.abs(target_t)).detach().cpu()),
                "weighted_mean_abs_target": float((torch.sum(weight_t * target_mag) / torch.clamp(torch.sum(weight_t), min=1.0e-6)).detach().cpu()),
                "batch_cosine": float(torch.mean(cosine).detach().cpu()),
            }
            train_rows.append(row)
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=int(step))

    render_dir = output_dir / "policy_val_best"
    alpha_grid = _parse_float_grid(str(args.alpha_grid), fallback=0.0)
    apply_confidence_threshold_grid = _parse_float_grid(
        str(args.apply_confidence_threshold_grid),
        fallback=float(args.apply_confidence_threshold),
    )
    apply_gate_strength_grid = _parse_float_grid(str(args.apply_gate_strength_grid), fallback=float(args.apply_gate_strength))
    policy_val = _evaluate(
        model,
        val_paths,
        candidate_faces,
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        feature_mode=str(args.feature_mode),
        alpha_grid=alpha_grid,
        apply_confidence_threshold_grid=apply_confidence_threshold_grid,
        apply_gate_mode=str(args.apply_gate_mode),
        apply_gate_strength_grid=apply_gate_strength_grid,
        apply_gate_floor=float(args.apply_gate_floor),
        apply_gate_eps=float(args.apply_gate_eps),
        chunk_size=int(args.eval_chunk_size),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
        output_dir=render_dir,
        device=device,
    )
    best = policy_val["best"]
    all_axis = policy_val.get("best_all_axis") is not None
    selected_policy = policy_val.get("best_all_axis") or best
    selected_alpha = float(selected_policy["alpha"])
    selected_confidence_threshold = float(
        selected_policy.get("apply_confidence_threshold", apply_confidence_threshold_grid[0])
    )
    selected_gate_strength = float(selected_policy.get("apply_gate_strength", apply_gate_strength_grid[0]))
    target_no_gt_audit = (
        _verify_target_no_gt(Path(args.target_evidence_dir))
        if str(args.target_evidence_dir)
        else {"checked_views": 0, "pass": False}
    )
    target_eval: dict[str, Any] = {}
    run_target_eval = str(args.target_eval_mode) == "always" or (
        str(args.target_eval_mode) == "auto" and all_axis and bool(target_no_gt_audit.get("pass", False))
    )
    if run_target_eval and str(args.target_eval_evidence_dir):
        target_eval = _target_exact_eval(
            model,
            Path(args.target_evidence_dir),
            Path(args.target_eval_evidence_dir),
            candidate_faces,
            residual_l1_key=str(args.residual_l1_key),
            min_l1=float(args.min_l1),
            min_alpha=float(args.min_alpha),
            feature_mode=str(args.feature_mode),
            alpha=float(selected_alpha),
            apply_confidence_threshold=float(selected_confidence_threshold),
            apply_gate_mode=str(args.apply_gate_mode),
            apply_gate_strength=float(selected_gate_strength),
            apply_gate_floor=float(args.apply_gate_floor),
            apply_gate_eps=float(args.apply_gate_eps),
            chunk_size=int(args.eval_chunk_size),
            ssim_max_side=int(args.policy_val_ssim_max_side),
            lpips_max_side=int(args.policy_val_lpips_max_side),
            compute_lpips=bool(args.compute_lpips),
            output_dir=output_dir,
            device=device,
        )
        summary = target_eval.get("summary", {})
        target_eval["pass_vs_parent_all_axis"] = bool(
            summary.get("psnr_gain", 0.0) > 0.0
            and summary.get("ssim_gain", 0.0) > 0.0
            and (not bool(args.compute_lpips) or summary.get("lpips_gain", 0.0) > 0.0)
        )
    target_phasej_pass = bool(
        target_eval.get("phasej_reference_comparison", {}).get("beats_phasej_all_axis_under_reported_metric_scale", False)
    )
    if target_phasej_pass:
        interpretation = (
            "The learned surface-feature decoder passed policy-val and flowers exact Phase-J all-axis gates; "
            "the fixed policy is eligible for full9."
        )
    elif all_axis:
        interpretation = (
            "The learned surface-feature decoder passed policy-val, but flowers exact did not beat Phase-J all-axis. "
            "It must not be promoted to full9; inspect target exact structure/perceptual failure."
        )
    else:
        interpretation = (
            "The learned surface-feature decoder did not pass the policy-val all-axis gate. It should not be promoted "
            "to full9; inspect target exact only as forced diagnostic evidence if it was explicitly requested."
        )
    payload: dict[str, Any] = {
        "schema": "spcarnet_perceptual_surface_decoder_audit_v2",
        "created_at": "2026-06-30",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "device": str(device),
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "target_evidence_dir": str(args.target_evidence_dir),
        "target_eval_evidence_dir": str(args.target_eval_evidence_dir),
        "fit_views": len(fit_paths),
        "policy_val_views": len(val_paths),
        "candidate_face_summary": face_summary,
        "train": {
            "requested_steps": int(args.steps),
            "steps": int(train_steps),
            "skip_training": bool(args.skip_training),
            "init_checkpoint": str(args.init_checkpoint),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "confidence_head": bool(args.confidence_head),
            "confidence_floor": float(args.confidence_floor),
            "confidence_loss_weight": float(args.confidence_loss_weight),
            "confidence_target_mode": str(args.confidence_target_mode),
            "confidence_gain_floor": float(args.confidence_gain_floor),
            "confidence_gain_scale": float(args.confidence_gain_scale),
            "feature_mode": str(args.feature_mode),
            "feature_dim": int(_feature_dim(str(args.feature_mode))),
            "image_loss_every": int(args.image_loss_every),
            "image_loss_stride": int(args.image_loss_stride),
            "image_loss_weight": float(args.image_loss_weight),
            "sample_weight_gamma": float(args.sample_weight_gamma),
            "sample_weight_clip": float(args.sample_weight_clip),
            "sample_weight_confidence_power": float(args.sample_weight_confidence_power),
            "cosine_loss_weight": float(args.cosine_loss_weight),
            "energy_match_weight": float(args.energy_match_weight),
            "apply_confidence_threshold": float(args.apply_confidence_threshold),
            "apply_confidence_threshold_grid": [float(x) for x in apply_confidence_threshold_grid],
            "apply_gate_mode": str(args.apply_gate_mode),
            "apply_gate_strength": float(args.apply_gate_strength),
            "apply_gate_strength_grid": [float(x) for x in apply_gate_strength_grid],
            "apply_gate_floor": float(args.apply_gate_floor),
            "apply_gate_eps": float(args.apply_gate_eps),
            "rows": train_rows,
        },
        "teacher_signal_pass": True,
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "uses_target_or_test_gt_after_apply_for_eval": bool(target_eval),
        "policy_val_all_axis_pass": all_axis,
        "policy_val": policy_val,
        "target_no_gt_audit": target_no_gt_audit,
        "target_exact_eval": target_eval,
        "phasej_flowers_exact_reference": PHASEJ_FLOWERS,
        "flowers_exact_phasej_gate_pass": bool(target_phasej_pass),
        "flowers_exact_run_allowed_next": bool(target_phasej_pass),
        "interpretation": interpretation,
        "output_render_dir": str(render_dir),
        "output_json": str(output_dir / "v180_perceptual_surface_decoder_audit.json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v180_perceptual_surface_decoder_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_md(output_dir / "v180_perceptual_surface_decoder_audit.md", payload)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "candidate_faces": candidate_faces,
            "args": vars(args),
        },
        output_dir / "v180_perceptual_surface_decoder.pt",
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "policy_val/all_axis_pass": float(all_axis),
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "policy_val/best_alpha": float(best.get("alpha", 0.0)),
                "policy_val/best_apply_confidence_threshold": float(best.get("apply_confidence_threshold", 0.0)),
                "policy_val/best_apply_gate_strength": float(best.get("apply_gate_strength", 0.0)),
                "policy_val/selected_alpha": float(selected_alpha),
                "policy_val/selected_apply_confidence_threshold": float(selected_confidence_threshold),
                "policy_val/selected_apply_gate_strength": float(selected_gate_strength),
                "target_exact/ran": float(bool(target_eval)),
                "target_exact/psnr_gain": float(target_eval.get("summary", {}).get("psnr_gain", 0.0)),
                "target_exact/ssim_gain": float(target_eval.get("summary", {}).get("ssim_gain", 0.0)),
                "target_exact/lpips_gain": float(target_eval.get("summary", {}).get("lpips_gain", 0.0)),
                "target_exact/phasej_gate_pass": float(
                    bool(
                        target_eval.get("phasej_reference_comparison", {}).get(
                            "beats_phasej_all_axis_under_reported_metric_scale",
                            False,
                        )
                    )
                ),
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_json": payload["output_json"],
                "output_md": str(output_dir / "v180_perceptual_surface_decoder_audit.md"),
                "policy_val_all_axis_pass": all_axis,
                "flowers_exact_phasej_gate_pass": payload["flowers_exact_phasej_gate_pass"],
                "selected_alpha": selected_alpha,
                "selected_apply_confidence_threshold": selected_confidence_threshold,
                "selected_apply_gate_strength": selected_gate_strength,
                "best": best,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
