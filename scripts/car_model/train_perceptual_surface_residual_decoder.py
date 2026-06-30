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


def _chroma_shrink_residual(residual: np.ndarray, chroma_scale: float) -> np.ndarray:
    chroma_scale = float(chroma_scale)
    if chroma_scale >= 0.999:
        return np.asarray(residual, dtype=np.float32)
    luma = _luma(np.asarray(residual, dtype=np.float32)).reshape(1, residual.shape[1], residual.shape[2])
    return (luma + chroma_scale * (np.asarray(residual, dtype=np.float32) - luma)).astype(np.float32)


def _transformed_residual_target(
    z: np.lib.npyio.NpzFile,
    *,
    residual_rgb_key: str,
    residual_target_mode: str,
    residual_target_gain_floor: float,
    residual_target_gain_scale: float,
    residual_target_structure_strength: float,
    residual_target_structure_floor: float,
    residual_target_structure_eps: float,
    residual_target_chroma_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    residual = np.asarray(z[residual_rgb_key], dtype=np.float32)[:3]
    mode = str(residual_target_mode)
    if mode == "raw":
        target = _chroma_shrink_residual(residual, float(residual_target_chroma_scale))
        return target, np.ones(residual.shape[1:], dtype=np.float32), {
            "mode": mode,
            "enabled": False,
            "mean_scale": 1.0,
        }
    if mode not in {"gain_soft", "structure_safe", "structure_gain"}:
        raise ValueError(f"unknown residual_target_mode={mode}")

    scale = np.ones(residual.shape[1:], dtype=np.float32)
    if mode in {"gain_soft", "structure_gain"} and "teacher_gain_l1" in z:
        gain = np.asarray(z["teacher_gain_l1"], dtype=np.float32)
        gain_scale = np.clip(
            (gain - float(residual_target_gain_floor)) / max(float(residual_target_gain_scale), 1.0e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        scale *= gain_scale

    if mode in {"structure_safe", "structure_gain"}:
        parent = np.asarray(z["rgb_render"], dtype=np.float32)[:3]
        parent_edge = _gradient_magnitude_2d(_luma(parent))
        residual_edge = _gradient_magnitude_2d(_luma(residual))
        denom = np.maximum(parent_edge + float(residual_target_structure_eps), float(residual_target_structure_eps))
        risk = np.clip(np.nan_to_num(residual_edge / denom, nan=0.0, posinf=1.0e6, neginf=0.0), 0.0, 1.0e6)
        structure_gate = float(residual_target_structure_floor) + (1.0 - float(residual_target_structure_floor)) * np.exp(
            -float(residual_target_structure_strength) * risk
        )
        scale *= np.clip(structure_gate, float(residual_target_structure_floor), 1.0).astype(np.float32)

    scale = np.clip(np.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    target = residual * scale.reshape(1, scale.shape[0], scale.shape[1])
    target = _chroma_shrink_residual(target.astype(np.float32), float(residual_target_chroma_scale))
    active = scale[scale > 0.0]
    return target.astype(np.float32), scale, {
        "mode": mode,
        "enabled": True,
        "mean_scale": float(np.mean(scale)),
        "positive_scale_fraction": float(np.mean(scale > 0.0)),
        "active_mean_scale": float(np.mean(active)) if active.size else 0.0,
        "p10_scale": float(np.quantile(scale, 0.10)),
        "p50_scale": float(np.quantile(scale, 0.50)),
        "p90_scale": float(np.quantile(scale, 0.90)),
        "chroma_scale": float(residual_target_chroma_scale),
    }


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


SURFACE_TEXTURE_V1_DIM = 18
SURFACE_TEXTURE_V2_DIM = 22
LOWRANK_TEXTURE_BASIS_COUNT = 4
LOWRANK_TEXTURE_BASIS_OFFSET = SURFACE_TEXTURE_V2_DIM
LOWRANK_TEXTURE_EIGEN_OFFSET = LOWRANK_TEXTURE_BASIS_OFFSET + 3 * LOWRANK_TEXTURE_BASIS_COUNT
LOWRANK_TEXTURE_RELIABILITY_INDEX = LOWRANK_TEXTURE_EIGEN_OFFSET + 3
SURFACE_TEXTURE_LOWRANK_V1_DIM = LOWRANK_TEXTURE_RELIABILITY_INDEX + 1


def _base_feature_dim(feature_mode: str) -> int:
    if str(feature_mode) == "basic":
        return 18
    if str(feature_mode) == "fourier_v1":
        return 49
    raise ValueError(f"unknown feature_mode={feature_mode}")


def _surface_texture_dim(surface_feature_texture: dict[str, Any] | None) -> int:
    if not surface_feature_texture:
        return 0
    return int(surface_feature_texture.get("feature_dim", 0))


def _feature_dim(feature_mode: str, surface_feature_texture: dict[str, Any] | None = None) -> int:
    return _base_feature_dim(str(feature_mode)) + _surface_texture_dim(surface_feature_texture)


def _surface_texture_reliability_from_rows(
    features: np.ndarray,
    surface_feature_texture: dict[str, Any] | None,
) -> np.ndarray:
    if not surface_feature_texture:
        return np.ones((int(features.shape[0]),), dtype=np.float32)
    texture_dim = _surface_texture_dim(surface_feature_texture)
    if texture_dim <= 0:
        return np.ones((int(features.shape[0]),), dtype=np.float32)
    tex = np.asarray(features[:, -texture_dim:], dtype=np.float32)
    mode = str(surface_feature_texture.get("mode", "v1"))
    if mode == "lowrank_v1" and texture_dim >= SURFACE_TEXTURE_LOWRANK_V1_DIM:
        return np.clip(
            np.nan_to_num(tex[:, LOWRANK_TEXTURE_RELIABILITY_INDEX], nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32)
    if mode == "v2" and texture_dim >= SURFACE_TEXTURE_V2_DIM:
        return np.clip(np.nan_to_num(tex[:, 21], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    # v1 fallback: covered-bin support times positive teacher-gain fraction.
    if texture_dim >= 10:
        return np.clip(tex[:, 1] * tex[:, 9], 0.0, 1.0).astype(np.float32)
    return np.ones((int(features.shape[0]),), dtype=np.float32)


def _surface_uv_bin_ids(z: np.lib.npyio.NpzFile, ys: np.ndarray, xs: np.ndarray, uv_bins: int) -> np.ndarray:
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    bins = max(1, int(uv_bins))
    u_bin = np.clip(np.floor(np.clip(bary[1, ys, xs], 0.0, 0.999999) * bins).astype(np.int64), 0, bins - 1)
    v_bin = np.clip(np.floor(np.clip(bary[2, ys, xs], 0.0, 0.999999) * bins).astype(np.int64), 0, bins - 1)
    return (u_bin * bins + v_bin).astype(np.int64)


def _lowrank_residual_basis(
    mean_rgb: np.ndarray,
    second_moment6: np.ndarray,
    count: float,
    mean_norm: float,
    luma_sign_sum: float,
    positive_gain_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    mean = np.asarray(mean_rgb, dtype=np.float64).reshape(3)
    second6 = np.asarray(second_moment6, dtype=np.float64).reshape(6)
    basis = np.zeros((LOWRANK_TEXTURE_BASIS_COUNT, 3), dtype=np.float32)
    eigen_sqrt = np.zeros((3,), dtype=np.float32)
    basis[0] = mean.astype(np.float32)
    if float(count) > 1.0:
        second = np.array(
            [
                [second6[0], second6[3], second6[4]],
                [second6[3], second6[1], second6[5]],
                [second6[4], second6[5], second6[2]],
            ],
            dtype=np.float64,
        )
        cov = second - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            eigvecs = eigvecs[:, order]
            scales = np.sqrt(eigvals[:3])
            eigen_sqrt[:] = scales.astype(np.float32)
            for idx in range(3):
                basis[idx + 1] = (eigvecs[:, idx] * scales[idx]).astype(np.float32)
        except np.linalg.LinAlgError:
            pass
    direction_agreement = float(np.linalg.norm(mean) / max(float(mean_norm), 1.0e-6))
    luma_sign_consistency = float(abs(float(luma_sign_sum)) / max(float(count), 1.0))
    variance_energy = float(np.sum(np.square(eigen_sqrt)))
    mean_energy = float(np.sum(np.square(basis[0])))
    stability = mean_energy / max(mean_energy + variance_energy, 1.0e-8)
    reliability = direction_agreement * luma_sign_consistency * float(np.clip(positive_gain_fraction, 0.0, 1.0))
    reliability *= math.sqrt(max(stability, 0.0))
    reliability = float(np.clip(np.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
    basis = np.clip(np.nan_to_num(basis, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    return basis, eigen_sqrt.astype(np.float32), reliability


def _lowrank_residual_basis_batch(
    mean_rgb: np.ndarray,
    second_moment6: np.ndarray,
    counts: np.ndarray,
    mean_norm: np.ndarray,
    luma_sign_sum: np.ndarray,
    positive_gain_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean_rgb, dtype=np.float64).reshape(-1, 3)
    second6 = np.asarray(second_moment6, dtype=np.float64).reshape(-1, 6)
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = int(mean.shape[0])
    basis = np.zeros((n, LOWRANK_TEXTURE_BASIS_COUNT, 3), dtype=np.float32)
    eigen_sqrt = np.zeros((n, 3), dtype=np.float32)
    basis[:, 0, :] = mean.astype(np.float32)
    if n:
        second = np.zeros((n, 3, 3), dtype=np.float64)
        second[:, 0, 0] = second6[:, 0]
        second[:, 1, 1] = second6[:, 1]
        second[:, 2, 2] = second6[:, 2]
        second[:, 0, 1] = second[:, 1, 0] = second6[:, 3]
        second[:, 0, 2] = second[:, 2, 0] = second6[:, 4]
        second[:, 1, 2] = second[:, 2, 1] = second6[:, 5]
        cov = second - mean[:, :, None] * mean[:, None, :]
        cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
        cov[counts <= 1.0] = 0.0
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals, axis=1)[:, ::-1]
            eigvals = np.take_along_axis(eigvals, order, axis=1)
            eigvecs = np.take_along_axis(eigvecs, order[:, None, :], axis=2)
            scales = np.sqrt(np.maximum(eigvals[:, :3], 0.0))
            eigen_sqrt[:, :] = scales.astype(np.float32)
            basis[:, 1:4, :] = np.transpose(eigvecs[:, :, :3] * scales[:, None, :], (0, 2, 1)).astype(np.float32)
        except np.linalg.LinAlgError:
            pass
    direction_agreement = np.linalg.norm(mean, axis=1) / np.maximum(np.asarray(mean_norm, dtype=np.float64), 1.0e-6)
    luma_sign_consistency = np.abs(np.asarray(luma_sign_sum, dtype=np.float64)) / np.maximum(counts, 1.0)
    variance_energy = np.sum(np.square(eigen_sqrt.astype(np.float64)), axis=1)
    mean_energy = np.sum(np.square(basis[:, 0, :].astype(np.float64)), axis=1)
    stability = mean_energy / np.maximum(mean_energy + variance_energy, 1.0e-8)
    reliability = (
        direction_agreement
        * luma_sign_consistency
        * np.clip(np.asarray(positive_gain_fraction, dtype=np.float64), 0.0, 1.0)
        * np.sqrt(np.maximum(stability, 0.0))
    )
    reliability = np.clip(np.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    basis = np.clip(np.nan_to_num(basis, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    return basis, eigen_sqrt.astype(np.float32), reliability


def _lookup_surface_texture_rows(
    z: np.lib.npyio.NpzFile,
    ys: np.ndarray,
    xs: np.ndarray,
    face_idx: np.ndarray,
    surface_feature_texture: dict[str, Any] | None,
) -> np.ndarray:
    if not surface_feature_texture:
        return np.zeros((int(ys.size), 0), dtype=np.float32)
    features = np.asarray(surface_feature_texture["features"], dtype=np.float32)
    uv_bins = int(surface_feature_texture.get("uv_bins", 1))
    bin_count = max(1, uv_bins * uv_bins)
    bin_ids = _surface_uv_bin_ids(z, ys, xs, uv_bins)
    flat_ids = np.asarray(face_idx, dtype=np.int64) * bin_count + bin_ids
    rows = np.zeros((int(ys.size), int(surface_feature_texture.get("feature_dim", features.shape[1]))), dtype=np.float32)
    valid = (flat_ids >= 0) & (flat_ids < int(features.shape[0]))
    rows[valid] = features[flat_ids[valid]]
    return rows.astype(np.float32)


def _load_feature_rows(
    z: np.lib.npyio.NpzFile,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    feature_mode: str = "basic",
    face_idx: np.ndarray | None = None,
    surface_feature_texture: dict[str, Any] | None = None,
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
        rows = base
        if surface_feature_texture is not None:
            if face_idx is None:
                raise ValueError("surface_feature_texture lookup requires face_idx")
            rows = np.concatenate(
                [rows, _lookup_surface_texture_rows(z, ys, xs, face_idx, surface_feature_texture)],
                axis=1,
            )
        return rows.astype(np.float32)
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
    rows = np.concatenate([base, *extra], axis=1).astype(np.float32)
    if surface_feature_texture is not None:
        if face_idx is None:
            raise ValueError("surface_feature_texture lookup requires face_idx")
        rows = np.concatenate(
            [rows, _lookup_surface_texture_rows(z, ys, xs, face_idx, surface_feature_texture)],
            axis=1,
        )
    return rows.astype(np.float32)


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


def _fit_calibration_policy_split(paths: list[Path], policy_stride: int, calibration_stride: int) -> tuple[list[Path], list[Path], list[Path]]:
    if int(calibration_stride) <= 1:
        fit, val = _policy_split(paths, int(policy_stride))
        return fit, [], val
    fit, cal, val = [], [], []
    for idx, path in enumerate(paths):
        if int(policy_stride) > 1 and idx % int(policy_stride) == 0:
            val.append(path)
        elif idx % int(calibration_stride) == 1:
            cal.append(path)
        else:
            fit.append(path)
    if not fit and cal:
        fit, cal = cal, []
    return fit, cal, val


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


def _fit_surface_feature_texture(
    fit_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    mode: str,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    uv_bins: int,
    max_samples_per_view: int,
    residual_target_mode: str,
    residual_target_gain_floor: float,
    residual_target_gain_scale: float,
    residual_target_structure_strength: float,
    residual_target_structure_floor: float,
    residual_target_structure_eps: float,
    residual_target_chroma_scale: float,
    seed: int,
) -> dict[str, Any]:
    """Bake train-fit teacher residual statistics into a face+UV texture."""
    rng = np.random.default_rng(int(seed))
    bins = max(1, int(uv_bins))
    mode = str(mode)
    if mode not in {"v1", "v2", "lowrank_v1"}:
        raise ValueError(f"unknown surface texture mode={mode}")
    if mode == "lowrank_v1":
        feature_dim = SURFACE_TEXTURE_LOWRANK_V1_DIM
    elif mode == "v2":
        feature_dim = SURFACE_TEXTURE_V2_DIM
    else:
        feature_dim = SURFACE_TEXTURE_V1_DIM
    bin_count = bins * bins
    face_count = int(candidate_faces.size)
    total_bins = face_count * bin_count
    stat_dim = 15
    bin_sum = np.zeros((total_bins, stat_dim), dtype=np.float64)
    bin_l1_sq_sum = np.zeros((total_bins,), dtype=np.float64)
    bin_res_norm_sum = np.zeros((total_bins,), dtype=np.float64)
    bin_luma_sign_sum = np.zeros((total_bins,), dtype=np.float64)
    bin_second_moment_sum = np.zeros((total_bins, 6), dtype=np.float64)
    bin_counts = np.zeros((total_bins,), dtype=np.int64)
    face_sum = np.zeros((face_count, stat_dim), dtype=np.float64)
    face_l1_sq_sum = np.zeros((face_count,), dtype=np.float64)
    face_res_norm_sum = np.zeros((face_count,), dtype=np.float64)
    face_luma_sign_sum = np.zeros((face_count,), dtype=np.float64)
    face_second_moment_sum = np.zeros((face_count, 6), dtype=np.float64)
    face_counts = np.zeros((face_count,), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    sampled_pixels = 0
    used_pixels = 0

    for path in tqdm(fit_paths, desc="fit surface feature texture"):
        z = np.load(path)
        mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            rows.append({"view": path.stem, "sampled": 0, "used": 0})
            continue
        if int(max_samples_per_view) > 0 and ys.size > int(max_samples_per_view):
            take = rng.choice(ys.size, size=int(max_samples_per_view), replace=False)
            ys, xs = ys[take], xs[take]
        sampled_pixels += int(ys.size)
        faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        face_idx, ok = _face_indices(faces, candidate_faces)
        ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
        if ys.size == 0:
            rows.append({"view": path.stem, "sampled": int(sampled_pixels), "used": 0})
            continue
        residual, _target_scale, _target_summary = _transformed_residual_target(
            z,
            residual_rgb_key=str(residual_rgb_key),
            residual_target_mode=str(residual_target_mode),
            residual_target_gain_floor=float(residual_target_gain_floor),
            residual_target_gain_scale=float(residual_target_gain_scale),
            residual_target_structure_strength=float(residual_target_structure_strength),
            residual_target_structure_floor=float(residual_target_structure_floor),
            residual_target_structure_eps=float(residual_target_structure_eps),
            residual_target_chroma_scale=float(residual_target_chroma_scale),
        )
        res = np.stack([residual[0, ys, xs], residual[1, ys, xs], residual[2, ys, xs]], axis=1).astype(np.float32)
        res_norm = np.linalg.norm(res, axis=1).astype(np.float32)
        abs_res = np.mean(np.abs(res), axis=1).astype(np.float32)
        res_luma = (0.299 * res[:, 0] + 0.587 * res[:, 1] + 0.114 * res[:, 2]).astype(np.float32)
        luma_sign = np.sign(np.where(np.abs(res_luma) > 1.0e-6, res_luma, 0.0)).astype(np.float32)
        if "teacher_gain_l1" in z:
            gain = np.asarray(z["teacher_gain_l1"], dtype=np.float32)[ys, xs]
        elif residual_l1_key in z:
            gain = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
        else:
            gain = abs_res
        gain = np.clip(np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)
        parent = np.asarray(z["rgb_render"], dtype=np.float32)
        parent_rows = np.stack([parent[0, ys, xs], parent[1, ys, xs], parent[2, ys, xs]], axis=1)
        parent_rows = np.clip(np.nan_to_num(parent_rows, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        alpha = np.asarray(z["alpha"], dtype=np.float32)[ys, xs].reshape(-1, 1)
        alpha = np.clip(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        normal = np.asarray(z["normal"], dtype=np.float32)
        normal_rows = np.stack([normal[0, ys, xs], normal[1, ys, xs], normal[2, ys, xs]], axis=1)
        normal_rows = np.nan_to_num(normal_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normal_rows = normal_rows / np.maximum(np.linalg.norm(normal_rows, axis=1, keepdims=True), 1.0e-8)
        camera = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
        camera = camera / max(float(np.linalg.norm(camera)), 1.0e-8)
        ndot = np.sum(normal_rows * camera.reshape(1, 3), axis=1, keepdims=True).astype(np.float32)
        values = np.concatenate(
            [
                res,
                abs_res.reshape(-1, 1),
                res_luma.reshape(-1, 1),
                gain.reshape(-1, 1),
                (gain > 0.0).astype(np.float32).reshape(-1, 1),
                parent_rows,
                alpha,
                ndot,
                normal_rows,
            ],
            axis=1,
        ).astype(np.float64)
        bin_ids = _surface_uv_bin_ids(z, ys, xs, bins)
        flat_ids = face_idx.astype(np.int64) * bin_count + bin_ids
        used_pixels += int(flat_ids.size)
        bin_counts += np.bincount(flat_ids, minlength=total_bins).astype(np.int64)
        face_counts += np.bincount(face_idx, minlength=face_count).astype(np.int64)
        bin_l1_sq_sum += np.bincount(flat_ids, weights=np.square(abs_res).astype(np.float64), minlength=total_bins)
        face_l1_sq_sum += np.bincount(face_idx, weights=np.square(abs_res).astype(np.float64), minlength=face_count)
        bin_res_norm_sum += np.bincount(flat_ids, weights=res_norm.astype(np.float64), minlength=total_bins)
        face_res_norm_sum += np.bincount(face_idx, weights=res_norm.astype(np.float64), minlength=face_count)
        bin_luma_sign_sum += np.bincount(flat_ids, weights=luma_sign.astype(np.float64), minlength=total_bins)
        face_luma_sign_sum += np.bincount(face_idx, weights=luma_sign.astype(np.float64), minlength=face_count)
        second_terms = [
            np.square(res[:, 0]).astype(np.float64),
            np.square(res[:, 1]).astype(np.float64),
            np.square(res[:, 2]).astype(np.float64),
            (res[:, 0] * res[:, 1]).astype(np.float64),
            (res[:, 0] * res[:, 2]).astype(np.float64),
            (res[:, 1] * res[:, 2]).astype(np.float64),
        ]
        for channel, term in enumerate(second_terms):
            bin_second_moment_sum[:, channel] += np.bincount(flat_ids, weights=term, minlength=total_bins)
            face_second_moment_sum[:, channel] += np.bincount(face_idx, weights=term, minlength=face_count)
        for channel in range(stat_dim):
            bin_sum[:, channel] += np.bincount(flat_ids, weights=values[:, channel], minlength=total_bins)
            face_sum[:, channel] += np.bincount(face_idx, weights=values[:, channel], minlength=face_count)
        rows.append({"view": path.stem, "sampled": int(ys.size), "used": int(flat_ids.size)})

    features = np.zeros((total_bins, feature_dim), dtype=np.float32)
    bin_nonzero = bin_counts > 0
    face_nonzero = face_counts > 0
    if np.any(face_nonzero):
        face_mean = np.zeros((face_count, stat_dim), dtype=np.float64)
        face_mean[face_nonzero] = face_sum[face_nonzero] / face_counts[face_nonzero, None]
        face_var = np.zeros((face_count,), dtype=np.float64)
        face_var[face_nonzero] = np.maximum(
            face_l1_sq_sum[face_nonzero] / face_counts[face_nonzero] - np.square(face_mean[face_nonzero, 3]),
            0.0,
        )
        face_ids = np.nonzero(face_nonzero)[0]
        face_lowrank_basis = face_lowrank_eigen = face_lowrank_reliability = None
        if mode == "lowrank_v1":
            face_second = face_second_moment_sum[face_ids] / np.maximum(face_counts[face_ids, None], 1)
            face_lowrank_basis, face_lowrank_eigen, face_lowrank_reliability = _lowrank_residual_basis_batch(
                face_mean[face_ids, 0:3],
                face_second,
                face_counts[face_ids],
                face_res_norm_sum[face_ids] / np.maximum(face_counts[face_ids], 1),
                face_luma_sign_sum[face_ids],
                face_mean[face_ids, 6],
            )
        for local_face, face in enumerate(face_ids):
            start = int(face) * bin_count
            end = start + bin_count
            features[start:end, 2:5] = face_mean[face, 0:3].astype(np.float32)
            features[start:end, 5] = float(face_mean[face, 3])
            features[start:end, 6] = float(face_mean[face, 4])
            features[start:end, 7] = float(math.sqrt(float(face_var[face])))
            features[start:end, 8] = float(face_mean[face, 5])
            features[start:end, 9] = float(face_mean[face, 6])
            features[start:end, 10:13] = face_mean[face, 7:10].astype(np.float32)
            features[start:end, 13] = float(face_mean[face, 10])
            features[start:end, 14] = float(face_mean[face, 11])
            features[start:end, 15:18] = face_mean[face, 12:15].astype(np.float32)
            if mode in {"v2", "lowrank_v1"}:
                face_mean_norm = face_res_norm_sum[face] / max(float(face_counts[face]), 1.0)
                direction_agreement = float(np.linalg.norm(face_mean[face, 0:3]) / max(face_mean_norm, 1.0e-6))
                luma_sign_consistency = float(abs(face_luma_sign_sum[face]) / max(float(face_counts[face]), 1.0))
                relative_std = float(math.sqrt(float(face_var[face])) / max(float(abs(face_mean[face, 3])), 1.0e-6))
                reliability = direction_agreement * luma_sign_consistency * float(np.clip(face_mean[face, 6], 0.0, 1.0))
                features[start:end, 18] = float(np.clip(direction_agreement, 0.0, 1.0))
                features[start:end, 19] = float(np.clip(luma_sign_consistency, 0.0, 1.0))
                features[start:end, 20] = float(np.clip(relative_std, 0.0, 1.0))
                features[start:end, 21] = float(np.clip(reliability, 0.0, 1.0))
            if mode == "lowrank_v1" and face_lowrank_basis is not None:
                features[start:end, LOWRANK_TEXTURE_BASIS_OFFSET:LOWRANK_TEXTURE_EIGEN_OFFSET] = face_lowrank_basis[
                    local_face
                ].reshape(-1)
                features[start:end, LOWRANK_TEXTURE_EIGEN_OFFSET:LOWRANK_TEXTURE_RELIABILITY_INDEX] = face_lowrank_eigen[
                    local_face
                ]
                features[start:end, LOWRANK_TEXTURE_RELIABILITY_INDEX] = face_lowrank_reliability[local_face]
    if np.any(bin_nonzero):
        bin_mean = bin_sum[bin_nonzero] / bin_counts[bin_nonzero, None]
        bin_var = np.maximum(
            bin_l1_sq_sum[bin_nonzero] / bin_counts[bin_nonzero] - np.square(bin_mean[:, 3]),
            0.0,
        )
        features[bin_nonzero, 0] = (
            np.log1p(bin_counts[bin_nonzero].astype(np.float32))
            / max(float(np.log1p(np.max(bin_counts[bin_nonzero]))), 1.0e-6)
        ).astype(np.float32)
        features[bin_nonzero, 1] = 1.0
        features[bin_nonzero, 2:5] = bin_mean[:, 0:3].astype(np.float32)
        features[bin_nonzero, 5] = bin_mean[:, 3].astype(np.float32)
        features[bin_nonzero, 6] = bin_mean[:, 4].astype(np.float32)
        features[bin_nonzero, 7] = np.sqrt(bin_var).astype(np.float32)
        features[bin_nonzero, 8] = bin_mean[:, 5].astype(np.float32)
        features[bin_nonzero, 9] = bin_mean[:, 6].astype(np.float32)
        features[bin_nonzero, 10:13] = bin_mean[:, 7:10].astype(np.float32)
        features[bin_nonzero, 13] = bin_mean[:, 10].astype(np.float32)
        features[bin_nonzero, 14] = bin_mean[:, 11].astype(np.float32)
        features[bin_nonzero, 15:18] = bin_mean[:, 12:15].astype(np.float32)
        if mode in {"v2", "lowrank_v1"}:
            mean_norm = bin_res_norm_sum[bin_nonzero] / np.maximum(bin_counts[bin_nonzero], 1)
            direction_agreement = np.linalg.norm(bin_mean[:, 0:3], axis=1) / np.maximum(mean_norm, 1.0e-6)
            luma_sign_consistency = np.abs(bin_luma_sign_sum[bin_nonzero]) / np.maximum(bin_counts[bin_nonzero], 1)
            relative_std = np.sqrt(bin_var) / np.maximum(np.abs(bin_mean[:, 3]), 1.0e-6)
            reliability = direction_agreement * luma_sign_consistency * np.clip(bin_mean[:, 6], 0.0, 1.0)
            features[bin_nonzero, 18] = np.clip(direction_agreement, 0.0, 1.0).astype(np.float32)
            features[bin_nonzero, 19] = np.clip(luma_sign_consistency, 0.0, 1.0).astype(np.float32)
            features[bin_nonzero, 20] = np.clip(relative_std, 0.0, 1.0).astype(np.float32)
            features[bin_nonzero, 21] = np.clip(reliability, 0.0, 1.0).astype(np.float32)
        if mode == "lowrank_v1":
            nonzero_ids = np.nonzero(bin_nonzero)[0]
            bin_second = bin_second_moment_sum[bin_nonzero] / np.maximum(bin_counts[bin_nonzero, None], 1)
            basis, eigen_sqrt, lowrank_reliability = _lowrank_residual_basis_batch(
                bin_mean[:, 0:3],
                bin_second,
                bin_counts[bin_nonzero],
                bin_res_norm_sum[bin_nonzero] / np.maximum(bin_counts[bin_nonzero], 1),
                bin_luma_sign_sum[bin_nonzero],
                bin_mean[:, 6],
            )
            features[nonzero_ids, LOWRANK_TEXTURE_BASIS_OFFSET:LOWRANK_TEXTURE_EIGEN_OFFSET] = basis.reshape(
                basis.shape[0],
                -1,
            )
            features[nonzero_ids, LOWRANK_TEXTURE_EIGEN_OFFSET:LOWRANK_TEXTURE_RELIABILITY_INDEX] = eigen_sqrt
            features[nonzero_ids, LOWRANK_TEXTURE_RELIABILITY_INDEX] = lowrank_reliability
    features = np.clip(np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    summary = {
        "enabled": True,
        "mode": mode,
        "uv_bins": int(bins),
        "feature_dim": int(feature_dim),
        "candidate_faces": int(face_count),
        "total_bins": int(total_bins),
        "covered_bins": int(np.count_nonzero(bin_nonzero)),
        "covered_bin_fraction": float(np.mean(bin_nonzero)) if total_bins else 0.0,
        "covered_faces": int(np.count_nonzero(face_nonzero)),
        "covered_face_fraction": float(np.mean(face_nonzero)) if face_count else 0.0,
        "sampled_pixels": int(sampled_pixels),
        "used_pixels": int(used_pixels),
        "max_samples_per_view": int(max_samples_per_view),
        "mean_bin_count_on_covered": float(np.mean(bin_counts[bin_nonzero])) if np.any(bin_nonzero) else 0.0,
        "mean_direction_agreement_on_covered": (
            float(np.mean(features[bin_nonzero, 18])) if mode in {"v2", "lowrank_v1"} and np.any(bin_nonzero) else None
        ),
        "mean_luma_sign_consistency_on_covered": (
            float(np.mean(features[bin_nonzero, 19])) if mode in {"v2", "lowrank_v1"} and np.any(bin_nonzero) else None
        ),
        "mean_direction_reliability_on_covered": (
            float(np.mean(features[bin_nonzero, 21])) if mode in {"v2", "lowrank_v1"} and np.any(bin_nonzero) else None
        ),
        "lowrank_basis_count": int(LOWRANK_TEXTURE_BASIS_COUNT) if mode == "lowrank_v1" else 0,
        "mean_lowrank_reliability_on_covered": (
            float(np.mean(features[bin_nonzero, LOWRANK_TEXTURE_RELIABILITY_INDEX]))
            if mode == "lowrank_v1" and np.any(bin_nonzero)
            else None
        ),
        "mean_lowrank_basis0_l1_on_covered": (
            float(np.mean(np.mean(np.abs(features[bin_nonzero, LOWRANK_TEXTURE_BASIS_OFFSET:LOWRANK_TEXTURE_BASIS_OFFSET + 3]), axis=1)))
            if mode == "lowrank_v1" and np.any(bin_nonzero)
            else None
        ),
        "mean_lowrank_pca_energy_on_covered": (
            float(np.mean(np.sum(np.square(features[bin_nonzero, LOWRANK_TEXTURE_EIGEN_OFFSET:LOWRANK_TEXTURE_RELIABILITY_INDEX]), axis=1)))
            if mode == "lowrank_v1" and np.any(bin_nonzero)
            else None
        ),
        "rows": rows,
    }
    return {
        "mode": mode,
        "features": features,
        "counts": bin_counts.astype(np.int64),
        "uv_bins": int(bins),
        "feature_dim": int(feature_dim),
        "lowrank_basis_count": int(LOWRANK_TEXTURE_BASIS_COUNT) if mode == "lowrank_v1" else 0,
        "summary": summary,
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
        output_mode: str = "direct",
        lowrank_basis_count: int = 0,
        lowrank_basis_feature_offset: int = -1,
        lowrank_coeff_scale: float = 1.0,
    ):
        super().__init__()
        self.face_embedding = torch.nn.Embedding(int(face_count), int(embedding_dim))
        self.predict_confidence = bool(predict_confidence)
        self.confidence_floor = float(np.clip(float(confidence_floor), 0.0, 1.0))
        self.output_mode = str(output_mode)
        self.lowrank_basis_count = int(lowrank_basis_count)
        self.lowrank_basis_feature_offset = int(lowrank_basis_feature_offset)
        self.lowrank_coeff_scale = float(lowrank_coeff_scale)
        if self.output_mode not in {"direct", "lowrank_texture"}:
            raise ValueError(f"unknown decoder output mode={self.output_mode}")
        if self.output_mode == "lowrank_texture":
            if self.lowrank_basis_count <= 0 or self.lowrank_basis_feature_offset < 0:
                raise ValueError("lowrank_texture output requires positive basis count and feature offset")
            residual_out_dim = self.lowrank_basis_count
        else:
            residual_out_dim = 3
        out_dim = residual_out_dim + (1 if self.predict_confidence else 0)
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
        if self.output_mode == "lowrank_texture":
            coeff = torch.tanh(raw[:, : self.lowrank_basis_count]) * self.lowrank_coeff_scale
            basis_start = int(self.lowrank_basis_feature_offset)
            basis_end = basis_start + 3 * int(self.lowrank_basis_count)
            basis = features[:, basis_start:basis_end].reshape(-1, int(self.lowrank_basis_count), 3)
            residual = torch.sum(coeff[:, :, None] * basis, dim=1)
            residual = torch.clamp(residual, -self.max_delta, self.max_delta)
            confidence_index = self.lowrank_basis_count
        else:
            residual = torch.tanh(raw[:, :3]) * self.max_delta
            confidence_index = 3
        if not self.predict_confidence:
            confidence = torch.ones((raw.shape[0],), dtype=residual.dtype, device=residual.device)
        else:
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * torch.sigmoid(raw[:, confidence_index])
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
    residual_target_mode: str,
    residual_target_gain_floor: float,
    residual_target_gain_scale: float,
    residual_target_structure_strength: float,
    residual_target_structure_floor: float,
    residual_target_structure_eps: float,
    residual_target_chroma_scale: float,
    feature_mode: str,
    surface_feature_texture: dict[str, Any] | None,
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
    features = _load_feature_rows(
        z,
        ys,
        xs,
        feature_mode=str(feature_mode),
        face_idx=face_idx,
        surface_feature_texture=surface_feature_texture,
    )
    residual, target_scale, _target_summary = _transformed_residual_target(
        z,
        residual_rgb_key=str(residual_rgb_key),
        residual_target_mode=str(residual_target_mode),
        residual_target_gain_floor=float(residual_target_gain_floor),
        residual_target_gain_scale=float(residual_target_gain_scale),
        residual_target_structure_strength=float(residual_target_structure_strength),
        residual_target_structure_floor=float(residual_target_structure_floor),
        residual_target_structure_eps=float(residual_target_structure_eps),
        residual_target_chroma_scale=float(residual_target_chroma_scale),
    )
    target = np.stack([residual[0, ys, xs], residual[1, ys, xs], residual[2, ys, xs]], axis=1).astype(np.float32)
    confidence_mode = str(confidence_target_mode)
    if confidence_mode == "texture_direction":
        confidence_target = _surface_texture_reliability_from_rows(features, surface_feature_texture)
    elif confidence_mode == "gain_soft" and "teacher_gain_l1" in z:
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
    if str(residual_target_mode) != "raw":
        target_scale_rows = np.clip(target_scale[ys, xs].astype(np.float32), 0.02, 1.0)
        weights = weights * np.sqrt(target_scale_rows)
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
    residual_target_mode: str,
    residual_target_gain_floor: float,
    residual_target_gain_scale: float,
    residual_target_structure_strength: float,
    residual_target_structure_floor: float,
    residual_target_structure_eps: float,
    residual_target_chroma_scale: float,
    surface_feature_texture: dict[str, Any] | None,
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
    features = torch.from_numpy(
        _load_feature_rows(
            z,
            ys,
            xs,
            feature_mode=str(feature_mode),
            face_idx=face_idx,
            surface_feature_texture=surface_feature_texture,
        )
    ).to(device)
    face_t = torch.from_numpy(face_idx.astype(np.int64)).to(device)
    pred = model(face_t, features)

    parent_np = np.asarray(z["rgb_render"], dtype=np.float32)[:, :: int(stride), :: int(stride)]
    residual_np, _target_scale, _target_summary = _transformed_residual_target(
        z,
        residual_rgb_key=str(residual_rgb_key),
        residual_target_mode=str(residual_target_mode),
        residual_target_gain_floor=float(residual_target_gain_floor),
        residual_target_gain_scale=float(residual_target_gain_scale),
        residual_target_structure_strength=float(residual_target_structure_strength),
        residual_target_structure_floor=float(residual_target_structure_floor),
        residual_target_structure_eps=float(residual_target_structure_eps),
        residual_target_chroma_scale=float(residual_target_chroma_scale),
    )
    residual_np = residual_np[:, :: int(stride), :: int(stride)]
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


def _apply_face_reliability(
    pred: np.ndarray,
    face_idx: np.ndarray,
    face_reliability_scores: np.ndarray | None,
    face_reliability_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if face_reliability_scores is None:
        keep = np.ones((int(pred.shape[0]),), dtype=np.float32)
        return pred, keep
    scores = np.asarray(face_reliability_scores, dtype=np.float32)
    valid = (face_idx >= 0) & (face_idx < int(scores.size))
    keep = np.zeros((int(face_idx.size),), dtype=np.float32)
    keep[valid] = (scores[face_idx[valid]] >= float(face_reliability_threshold)).astype(np.float32)
    return (np.asarray(pred, dtype=np.float32) * keep.reshape(-1, 1)).astype(np.float32), keep


def _calibrate_face_reliability(
    model: SurfaceResidualDecoder,
    calibration_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    surface_feature_texture: dict[str, Any] | None,
    calibration_alpha: float,
    structure_weight: float,
    min_count: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    score_sum = np.zeros((int(candidate_faces.size),), dtype=np.float64)
    l1_sum = np.zeros_like(score_sum)
    structure_sum = np.zeros_like(score_sum)
    counts = np.zeros((int(candidate_faces.size),), dtype=np.int64)
    positive_counts = np.zeros_like(counts)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for path in tqdm(calibration_paths, desc="calibrate face reliability"):
            z = np.load(path)
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                rows.append({"view": path.stem, "sample_count": 0})
                continue
            faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
            face_idx, ok = _face_indices(faces, candidate_faces)
            ys, xs, face_idx = ys[ok], xs[ok], face_idx[ok]
            if ys.size == 0:
                rows.append({"view": path.stem, "sample_count": 0})
                continue
            delta = np.zeros_like(parent, dtype=np.float32)
            for start in range(0, int(ys.size), int(chunk_size)):
                end = min(int(ys.size), start + int(chunk_size))
                feat = torch.from_numpy(
                    _load_feature_rows(
                        z,
                        ys[start:end],
                        xs[start:end],
                        feature_mode=str(feature_mode),
                        face_idx=face_idx[start:end],
                        surface_feature_texture=surface_feature_texture,
                    )
                ).to(device)
                face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                pred = model(face_t, feat).detach().cpu().numpy().astype(np.float32)
                delta[:, ys[start:end], xs[start:end]] = pred.T
            adapted = np.clip(parent + float(calibration_alpha) * delta, 0.0, 1.0).astype(np.float32)
            parent_l1 = np.mean(np.abs(parent - gt), axis=0)
            adapted_l1 = np.mean(np.abs(adapted - gt), axis=0)
            l1_gain = (parent_l1 - adapted_l1).astype(np.float32)
            parent_grad = _gradient_magnitude_2d(_luma(parent))
            adapted_grad = _gradient_magnitude_2d(_luma(adapted))
            gt_grad = _gradient_magnitude_2d(_luma(gt))
            structure_gain = (np.abs(parent_grad - gt_grad) - np.abs(adapted_grad - gt_grad)).astype(np.float32)
            score = (l1_gain + float(structure_weight) * structure_gain).astype(np.float32)
            sample_score = score[ys, xs].astype(np.float64)
            sample_l1 = l1_gain[ys, xs].astype(np.float64)
            sample_structure = structure_gain[ys, xs].astype(np.float64)
            score_sum += np.bincount(face_idx, weights=sample_score, minlength=int(candidate_faces.size))
            l1_sum += np.bincount(face_idx, weights=sample_l1, minlength=int(candidate_faces.size))
            structure_sum += np.bincount(face_idx, weights=sample_structure, minlength=int(candidate_faces.size))
            counts += np.bincount(face_idx, minlength=int(candidate_faces.size)).astype(np.int64)
            positive_counts += np.bincount(face_idx, weights=(sample_score > 0.0).astype(np.float64), minlength=int(candidate_faces.size)).astype(np.int64)
            rows.append(
                {
                    "view": path.stem,
                    "sample_count": int(sample_score.size),
                    "mean_score": float(np.mean(sample_score)) if sample_score.size else 0.0,
                    "positive_fraction": float(np.mean(sample_score > 0.0)) if sample_score.size else 0.0,
                }
            )
    scores = np.full((int(candidate_faces.size),), -1.0e9, dtype=np.float32)
    mean_l1 = np.zeros_like(scores)
    mean_structure = np.zeros_like(scores)
    positive_fraction = np.zeros_like(scores)
    enough = counts >= max(1, int(min_count))
    scores[enough] = (score_sum[enough] / np.maximum(counts[enough], 1)).astype(np.float32)
    mean_l1[enough] = (l1_sum[enough] / np.maximum(counts[enough], 1)).astype(np.float32)
    mean_structure[enough] = (structure_sum[enough] / np.maximum(counts[enough], 1)).astype(np.float32)
    positive_fraction[enough] = (positive_counts[enough] / np.maximum(counts[enough], 1)).astype(np.float32)
    finite_scores = scores[enough]
    summary = {
        "enabled": bool(calibration_paths),
        "calibration_views": int(len(calibration_paths)),
        "calibration_alpha": float(calibration_alpha),
        "structure_weight": float(structure_weight),
        "min_count": int(min_count),
        "candidate_faces": int(candidate_faces.size),
        "valid_faces": int(np.count_nonzero(enough)),
        "positive_face_fraction": float(np.mean(finite_scores > 0.0)) if finite_scores.size else 0.0,
        "mean_score": float(np.mean(finite_scores)) if finite_scores.size else 0.0,
        "p10_score": float(np.quantile(finite_scores, 0.10)) if finite_scores.size else 0.0,
        "p50_score": float(np.quantile(finite_scores, 0.50)) if finite_scores.size else 0.0,
        "p90_score": float(np.quantile(finite_scores, 0.90)) if finite_scores.size else 0.0,
        "rows": rows,
    }
    return {
        "scores": scores,
        "counts": counts.astype(np.int64),
        "mean_l1_gain": mean_l1,
        "mean_structure_gain": mean_structure,
        "positive_fraction": positive_fraction,
        "summary": summary,
    }


def _evaluate(
    model: SurfaceResidualDecoder,
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    feature_mode: str,
    surface_feature_texture: dict[str, Any] | None,
    alpha_grid: list[float],
    apply_confidence_threshold_grid: list[float],
    face_reliability_scores: np.ndarray | None,
    face_reliability_threshold_grid: list[float],
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
        (float(a), float(g), float(t), float(r))
        for a in alpha_grid
        for g in apply_gate_strength_grid
        for t in apply_confidence_threshold_grid
        for r in face_reliability_threshold_grid
    ]
    rows_by_policy: dict[str, list[dict[str, Any]]] = {
        f"{a:.8g}|{g:.8g}|{t:.8g}|{r:.8g}": [] for a, g, t, r in policy_grid
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
                        _load_feature_rows(
                            z,
                            ys[start:end],
                            xs[start:end],
                            feature_mode=str(feature_mode),
                            face_idx=face_idx[start:end],
                            surface_feature_texture=surface_feature_texture,
                        )
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
            for alpha, gate_strength, confidence_threshold, face_reliability_threshold in policy_grid:
                if face_reliability_scores is not None:
                    # Rebuild the face gate cheaply without rerunning the MLP.
                    gated_delta = np.zeros_like(delta, dtype=np.float32)
                    gated_confidence = np.zeros_like(confidence, dtype=np.float32)
                    if ys.size:
                        face_keep = (
                            np.asarray(face_reliability_scores, dtype=np.float32)[face_idx]
                            >= float(face_reliability_threshold)
                        ).astype(np.float32)
                        gated_delta[:, ys, xs] = delta[:, ys, xs] * face_keep.reshape(1, -1)
                        gated_confidence[ys, xs] = confidence[ys, xs] * face_keep
                else:
                    gated_delta = delta
                    gated_confidence = confidence
                adapted, applied_delta, gate_summary = _apply_delta(
                    parent,
                    gated_delta,
                    alpha=float(alpha),
                    confidence=gated_confidence,
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
                    "face_reliability_threshold": float(face_reliability_threshold),
                    "face_reliability_keep_fraction": (
                        float(np.mean(np.asarray(face_reliability_scores, dtype=np.float32)[face_idx] >= float(face_reliability_threshold)))
                        if face_reliability_scores is not None and ys.size
                        else 1.0
                    ),
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
                rows_by_policy[
                    f"{float(alpha):.8g}|{float(gate_strength):.8g}|{float(confidence_threshold):.8g}|{float(face_reliability_threshold):.8g}"
                ].append(row)
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
        face_reliability_threshold = float(rows[0]["face_reliability_threshold"]) if rows else -1.0e9
        summary = {
            "alpha": float(alpha),
            "apply_confidence_threshold": float(confidence_threshold),
            "face_reliability_threshold": float(face_reliability_threshold),
            "apply_gate_mode": str(apply_gate_mode),
            "apply_gate_strength": float(gate_strength),
            "apply_gate_floor": float(apply_gate_floor),
            "apply_gate_eps": float(apply_gate_eps),
            "parent_psnr": float(np.mean(parent_psnr)),
            "candidate_psnr": float(np.mean(cand_psnr)),
            "psnr_gain": float(np.mean(psnr_gain)),
            "psnr_gain_tail": _tail(psnr_gain),
            "parent_ssim": float(np.mean(parent_ssim)),
            "candidate_ssim": float(np.mean(cand_ssim)),
            "ssim_gain": float(np.mean(ssim_gain)),
            "ssim_gain_tail": _tail(ssim_gain),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
            "mean_changed_fraction": float(np.mean([r["changed_fraction"] for r in rows])),
            "mean_face_reliability_keep_fraction": float(np.mean([r["face_reliability_keep_fraction"] for r in rows])),
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
                    "lpips_gain_tail": _tail(lpips_gain),
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
        best_face_reliability_threshold = float(best.get("face_reliability_threshold", face_reliability_threshold_grid[0]))
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
                            _load_feature_rows(
                                z,
                                ys[start:end],
                                xs[start:end],
                                feature_mode=str(feature_mode),
                                face_idx=face_idx[start:end],
                                surface_feature_texture=surface_feature_texture,
                            )
                        ).to(device)
                        face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                        pred_t, conf_t = model.forward_with_confidence(face_t, feat)
                        pred = pred_t.detach().cpu().numpy().astype(np.float32)
                        pred, face_keep = _apply_face_reliability(
                            pred,
                            face_idx[start:end],
                            face_reliability_scores,
                            best_face_reliability_threshold,
                        )
                        conf = conf_t.detach().cpu().numpy().astype(np.float32)
                        delta[:, ys[start:end], xs[start:end]] = pred.T
                        confidence[ys[start:end], xs[start:end]] = conf * face_keep
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
    surface_feature_texture: dict[str, Any] | None,
    chunk_size: int,
    device: torch.device,
    face_reliability_scores: np.ndarray | None = None,
    face_reliability_threshold: float = -1.0e9,
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
                    _load_feature_rows(
                        z,
                        ys[start:end],
                        xs[start:end],
                        feature_mode=str(feature_mode),
                        face_idx=face_idx[start:end],
                        surface_feature_texture=surface_feature_texture,
                    )
                ).to(device)
                face_t = torch.from_numpy(face_idx[start:end].astype(np.int64)).to(device)
                pred_t, conf_t = model.forward_with_confidence(face_t, feat)
                pred = pred_t.detach().cpu().numpy().astype(np.float32)
                pred, face_keep = _apply_face_reliability(
                    pred,
                    face_idx[start:end],
                    face_reliability_scores,
                    float(face_reliability_threshold),
                )
                conf = conf_t.detach().cpu().numpy().astype(np.float32)
                delta[:, ys[start:end], xs[start:end]] = pred.T
                confidence[ys[start:end], xs[start:end]] = conf * face_keep
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
    surface_feature_texture: dict[str, Any] | None,
    alpha: float,
    apply_confidence_threshold: float,
    face_reliability_scores: np.ndarray | None,
    face_reliability_threshold: float,
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
        with np.load(apply_path, allow_pickle=False) as z_apply:
            leaked = sorted(set(z_apply.files) & FORBIDDEN_TARGET_KEYS)
            if leaked:
                raise RuntimeError(f"target apply evidence leaks GT keys for {apply_path}: {leaked}")
            parent = np.asarray(z_apply["rgb_render"], dtype=np.float32)
            delta, active_fraction, confidence = _predict_delta_image(
                model,
                z_apply,
                candidate_faces,
                residual_l1_key=str(residual_l1_key),
                min_l1=float(min_l1),
                min_alpha=float(min_alpha),
                feature_mode=str(feature_mode),
                surface_feature_texture=surface_feature_texture,
                chunk_size=int(chunk_size),
                device=device,
                face_reliability_scores=face_reliability_scores,
                face_reliability_threshold=float(face_reliability_threshold),
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
        with np.load(eval_paths[apply_path.stem], allow_pickle=False) as z_eval:
            gt = np.asarray(z_eval["rgb_gt"], dtype=np.float32)
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
                "face_reliability_threshold": float(face_reliability_threshold),
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
        "face_reliability_threshold": float(face_reliability_threshold),
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
        "# Learned Surface Feature Decoder Audit",
        "",
        f"- teacher signal pass: `{payload['teacher_signal_pass']}`",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- no-target-GT audit pass: `{payload.get('target_no_gt_audit', {}).get('pass')}`",
        f"- target exact fixed-policy pass vs parent: `{payload.get('target_exact_eval', {}).get('pass_vs_parent_all_axis')}`",
        f"- flowers exact Phase-J gate pass: `{payload.get('flowers_exact_phasej_gate_pass')}`",
        f"- Phase-J flowers reference: `{PHASEJ_FLOWERS}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- train steps: `{payload['train']['steps']}`",
        f"- residual target mode: `{payload['train'].get('residual_target_mode')}`",
        f"- surface texture: `{payload.get('surface_feature_texture', {}).get('enabled', False)}` "
        f"mode `{payload.get('surface_feature_texture', {}).get('mode', 'none')}` "
        f"source `{payload.get('surface_feature_texture', {}).get('source', 'none')}`",
        f"- surface texture coverage: face `{payload.get('surface_feature_texture', {}).get('covered_face_fraction', 0.0):.6f}`, "
        f"bin `{payload.get('surface_feature_texture', {}).get('covered_bin_fraction', 0.0):.6f}`",
        f"- decoder output mode: `{payload['train'].get('decoder_output_mode', 'direct')}` "
        f"basis `{payload['train'].get('lowrank_basis_count', 0)}`",
        f"- calibration views: `{payload.get('calibration_views', 0)}`",
        f"- calibration face reliability: `{payload.get('calibration_face_reliability', {}).get('enabled', False)}`",
        f"- confidence head: `{payload['train'].get('confidence_head', False)}`",
        f"- confidence target: `{payload['train'].get('confidence_target_mode')}`",
        f"- apply confidence thresholds: `{payload['train'].get('apply_confidence_threshold_grid')}`",
        f"- apply gate: `{payload['train'].get('apply_gate_mode')}` grid `{payload['train'].get('apply_gate_strength_grid')}`",
        "",
        "## Best Policy-Val Row",
        "",
        "| alpha | face th | conf th | gate | PSNR gain | SSIM gain | LPIPS gain | changed | face keep | conf keep | pos views | SSIM pos | LPIPS pos |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.4f} | {best.get('face_reliability_threshold', -1.0e9):.6f} | "
            f"{best.get('apply_confidence_threshold', 0.0):.4f} | "
            f"{best.get('apply_gate_strength', 0.0):.4f} | "
            f"{best.get('psnr_gain', 0.0):.6f} | "
            f"{best.get('ssim_gain', 0.0):.6f} | {best.get('lpips_gain', 0.0):.6f} | "
            f"{best.get('mean_changed_fraction', 0.0):.6f} | "
            f"{best.get('mean_face_reliability_keep_fraction', 1.0):.6f} | "
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
    parser.add_argument("--calibration_stride", type=int, default=0)
    parser.add_argument("--enable_calibration_face_reliability", action="store_true")
    parser.add_argument("--calibration_alpha", type=float, default=0.5)
    parser.add_argument("--calibration_structure_weight", type=float, default=0.25)
    parser.add_argument("--calibration_min_face_count", type=int, default=128)
    parser.add_argument("--face_reliability_threshold", type=float, default=-1.0e9)
    parser.add_argument("--face_reliability_threshold_grid", default="")
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument(
        "--residual_target_mode",
        choices=["raw", "gain_soft", "structure_safe", "structure_gain"],
        default="raw",
    )
    parser.add_argument("--residual_target_gain_floor", type=float, default=0.003)
    parser.add_argument("--residual_target_gain_scale", type=float, default=0.04)
    parser.add_argument("--residual_target_structure_strength", type=float, default=1.0)
    parser.add_argument("--residual_target_structure_floor", type=float, default=0.0)
    parser.add_argument("--residual_target_structure_eps", type=float, default=0.02)
    parser.add_argument("--residual_target_chroma_scale", type=float, default=1.0)
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
    parser.add_argument(
        "--confidence_target_mode",
        choices=["mask", "gain_binary", "gain_soft", "texture_direction"],
        default="mask",
    )
    parser.add_argument("--confidence_gain_floor", type=float, default=0.0)
    parser.add_argument("--confidence_gain_scale", type=float, default=0.03)
    parser.add_argument("--feature_mode", choices=["basic", "fourier_v1"], default="basic")
    parser.add_argument("--surface_texture_mode", choices=["none", "v1", "v2", "lowrank_v1"], default="none")
    parser.add_argument("--surface_texture_uv_bins", type=int, default=4)
    parser.add_argument("--surface_texture_max_samples_per_view", type=int, default=250000)
    parser.add_argument("--decoder_output_mode", choices=["direct", "lowrank_texture"], default="direct")
    parser.add_argument("--lowrank_coeff_scale", type=float, default=1.0)
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
    fit_paths, calibration_paths, val_paths = _fit_calibration_policy_split(
        paths,
        int(args.policy_val_stride),
        int(args.calibration_stride) if bool(args.enable_calibration_face_reliability) else 0,
    )
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
    surface_feature_texture: dict[str, Any] | None = None
    surface_texture_payload: dict[str, Any] = {"enabled": False, "mode": str(args.surface_texture_mode)}
    if str(args.surface_texture_mode) in {"v1", "v2", "lowrank_v1"}:
        if init_checkpoint is not None and init_checkpoint.get("surface_feature_texture") is not None:
            surface_feature_texture = init_checkpoint["surface_feature_texture"]
            loaded_mode = str(surface_feature_texture.get("mode", surface_feature_texture.get("summary", {}).get("mode", "v1")))
            if loaded_mode != str(args.surface_texture_mode):
                raise ValueError(
                    f"--surface_texture_mode {args.surface_texture_mode} requested, but checkpoint contains {loaded_mode}"
                )
            surface_texture_payload = {
                **dict(surface_feature_texture.get("summary", {})),
                "source": "init_checkpoint",
                "checkpoint": str(args.init_checkpoint),
            }
        elif init_checkpoint is not None:
            raise ValueError(
                f"--surface_texture_mode {args.surface_texture_mode} requires a matching checkpoint or no --init_checkpoint"
            )
        else:
            surface_feature_texture = _fit_surface_feature_texture(
                fit_paths,
                candidate_faces,
                mode=str(args.surface_texture_mode),
                residual_rgb_key=str(args.residual_rgb_key),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                uv_bins=int(args.surface_texture_uv_bins),
                max_samples_per_view=int(args.surface_texture_max_samples_per_view),
                residual_target_mode=str(args.residual_target_mode),
                residual_target_gain_floor=float(args.residual_target_gain_floor),
                residual_target_gain_scale=float(args.residual_target_gain_scale),
                residual_target_structure_strength=float(args.residual_target_structure_strength),
                residual_target_structure_floor=float(args.residual_target_structure_floor),
                residual_target_structure_eps=float(args.residual_target_structure_eps),
                residual_target_chroma_scale=float(args.residual_target_chroma_scale),
                seed=int(args.seed) + 17001,
            )
            surface_texture_payload = {
                **dict(surface_feature_texture.get("summary", {})),
                "source": "train_fit_evidence",
            }
            np.savez_compressed(
                output_dir / f"surface_feature_texture_{str(args.surface_texture_mode)}.npz",
                candidate_faces=candidate_faces.astype(np.int64),
                features=np.asarray(surface_feature_texture["features"], dtype=np.float32),
                counts=np.asarray(surface_feature_texture["counts"], dtype=np.int64),
                uv_bins=np.asarray([int(surface_feature_texture["uv_bins"])], dtype=np.int64),
                feature_dim=np.asarray([int(surface_feature_texture["feature_dim"])], dtype=np.int64),
                mode=np.asarray([str(args.surface_texture_mode)]),
                lowrank_basis_count=np.asarray(
                    [LOWRANK_TEXTURE_BASIS_COUNT if str(args.surface_texture_mode) == "lowrank_v1" else 0],
                    dtype=np.int64,
                ),
            )
    if str(args.decoder_output_mode) == "lowrank_texture" and str(args.surface_texture_mode) != "lowrank_v1":
        raise ValueError("--decoder_output_mode lowrank_texture requires --surface_texture_mode lowrank_v1")
    feature_dim = _feature_dim(str(args.feature_mode), surface_feature_texture)
    lowrank_basis_feature_offset = (
        _base_feature_dim(str(args.feature_mode)) + LOWRANK_TEXTURE_BASIS_OFFSET
        if str(args.decoder_output_mode) == "lowrank_texture"
        else -1
    )
    model = SurfaceResidualDecoder(
        int(candidate_faces.size),
        feature_dim=int(feature_dim),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        max_delta=float(args.max_delta),
        predict_confidence=bool(args.confidence_head),
        confidence_floor=float(args.confidence_floor),
        output_mode=str(args.decoder_output_mode),
        lowrank_basis_count=LOWRANK_TEXTURE_BASIS_COUNT if str(args.decoder_output_mode) == "lowrank_texture" else 0,
        lowrank_basis_feature_offset=int(lowrank_basis_feature_offset),
        lowrank_coeff_scale=float(args.lowrank_coeff_scale),
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
                    residual_target_mode=str(args.residual_target_mode),
                    residual_target_gain_floor=float(args.residual_target_gain_floor),
                    residual_target_gain_scale=float(args.residual_target_gain_scale),
                    residual_target_structure_strength=float(args.residual_target_structure_strength),
                    residual_target_structure_floor=float(args.residual_target_structure_floor),
                    residual_target_structure_eps=float(args.residual_target_structure_eps),
                    residual_target_chroma_scale=float(args.residual_target_chroma_scale),
                    feature_mode=str(args.feature_mode),
                    surface_feature_texture=surface_feature_texture,
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
                residual_target_mode=str(args.residual_target_mode),
                residual_target_gain_floor=float(args.residual_target_gain_floor),
                residual_target_gain_scale=float(args.residual_target_gain_scale),
                residual_target_structure_strength=float(args.residual_target_structure_strength),
                residual_target_structure_floor=float(args.residual_target_structure_floor),
                residual_target_structure_eps=float(args.residual_target_structure_eps),
                residual_target_chroma_scale=float(args.residual_target_chroma_scale),
                surface_feature_texture=surface_feature_texture,
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

    face_reliability_payload: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    face_reliability_scores: np.ndarray | None = None
    if bool(args.enable_calibration_face_reliability):
        if not calibration_paths:
            face_reliability_payload = {"enabled": False, "reason": "no_calibration_views"}
        else:
            reliability = _calibrate_face_reliability(
                model,
                calibration_paths,
                candidate_faces,
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                feature_mode=str(args.feature_mode),
                surface_feature_texture=surface_feature_texture,
                calibration_alpha=float(args.calibration_alpha),
                structure_weight=float(args.calibration_structure_weight),
                min_count=int(args.calibration_min_face_count),
                chunk_size=int(args.eval_chunk_size),
                device=device,
            )
            face_reliability_scores = np.asarray(reliability["scores"], dtype=np.float32)
            face_reliability_payload = {
                **dict(reliability["summary"]),
                "score_quantiles": {
                    "p10": float(reliability["summary"].get("p10_score", 0.0)),
                    "p50": float(reliability["summary"].get("p50_score", 0.0)),
                    "p90": float(reliability["summary"].get("p90_score", 0.0)),
                },
            }
            np.savez_compressed(
                output_dir / "calibration_face_reliability.npz",
                candidate_faces=candidate_faces.astype(np.int64),
                scores=face_reliability_scores.astype(np.float32),
                counts=np.asarray(reliability["counts"], dtype=np.int64),
                mean_l1_gain=np.asarray(reliability["mean_l1_gain"], dtype=np.float32),
                mean_structure_gain=np.asarray(reliability["mean_structure_gain"], dtype=np.float32),
                positive_fraction=np.asarray(reliability["positive_fraction"], dtype=np.float32),
            )

    render_dir = output_dir / "policy_val_best"
    alpha_grid = _parse_float_grid(str(args.alpha_grid), fallback=0.0)
    apply_confidence_threshold_grid = _parse_float_grid(
        str(args.apply_confidence_threshold_grid),
        fallback=float(args.apply_confidence_threshold),
    )
    face_reliability_threshold_grid = _parse_float_grid(
        str(args.face_reliability_threshold_grid),
        fallback=float(args.face_reliability_threshold),
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
        surface_feature_texture=surface_feature_texture,
        alpha_grid=alpha_grid,
        apply_confidence_threshold_grid=apply_confidence_threshold_grid,
        face_reliability_scores=face_reliability_scores,
        face_reliability_threshold_grid=face_reliability_threshold_grid,
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
    selected_face_reliability_threshold = float(
        selected_policy.get("face_reliability_threshold", face_reliability_threshold_grid[0])
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
            surface_feature_texture=surface_feature_texture,
            alpha=float(selected_alpha),
            apply_confidence_threshold=float(selected_confidence_threshold),
            face_reliability_scores=face_reliability_scores,
            face_reliability_threshold=float(selected_face_reliability_threshold),
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
        "calibration_views": len(calibration_paths),
        "policy_val_views": len(val_paths),
        "candidate_face_summary": face_summary,
        "train": {
            "requested_steps": int(args.steps),
            "steps": int(train_steps),
            "skip_training": bool(args.skip_training),
            "init_checkpoint": str(args.init_checkpoint),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "residual_target_mode": str(args.residual_target_mode),
            "residual_target_gain_floor": float(args.residual_target_gain_floor),
            "residual_target_gain_scale": float(args.residual_target_gain_scale),
            "residual_target_structure_strength": float(args.residual_target_structure_strength),
            "residual_target_structure_floor": float(args.residual_target_structure_floor),
            "residual_target_structure_eps": float(args.residual_target_structure_eps),
            "residual_target_chroma_scale": float(args.residual_target_chroma_scale),
            "enable_calibration_face_reliability": bool(args.enable_calibration_face_reliability),
            "calibration_stride": int(args.calibration_stride),
            "calibration_alpha": float(args.calibration_alpha),
            "calibration_structure_weight": float(args.calibration_structure_weight),
            "calibration_min_face_count": int(args.calibration_min_face_count),
            "face_reliability_threshold": float(args.face_reliability_threshold),
            "face_reliability_threshold_grid": [float(x) for x in face_reliability_threshold_grid],
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
            "feature_dim": int(feature_dim),
            "base_feature_dim": int(_base_feature_dim(str(args.feature_mode))),
            "decoder_output_mode": str(args.decoder_output_mode),
            "lowrank_basis_count": int(LOWRANK_TEXTURE_BASIS_COUNT)
            if str(args.decoder_output_mode) == "lowrank_texture"
            else 0,
            "lowrank_basis_feature_offset": int(lowrank_basis_feature_offset),
            "lowrank_coeff_scale": float(args.lowrank_coeff_scale),
            "surface_texture_mode": str(args.surface_texture_mode),
            "surface_texture_uv_bins": int(args.surface_texture_uv_bins),
            "surface_texture_max_samples_per_view": int(args.surface_texture_max_samples_per_view),
            "surface_texture_feature_dim": int(_surface_texture_dim(surface_feature_texture)),
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
        "uses_train_fit_surface_texture": bool(surface_feature_texture is not None),
        "uses_calibration_gt": bool(args.enable_calibration_face_reliability and calibration_paths),
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "uses_target_or_test_gt_after_apply_for_eval": bool(target_eval),
        "policy_val_all_axis_pass": all_axis,
        "policy_val": policy_val,
        "target_no_gt_audit": target_no_gt_audit,
        "surface_feature_texture": surface_texture_payload,
        "calibration_face_reliability": face_reliability_payload,
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
            "surface_feature_texture": surface_feature_texture,
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
                "policy_val/best_face_reliability_threshold": float(
                    best.get("face_reliability_threshold", face_reliability_threshold_grid[0])
                ),
                "policy_val/selected_alpha": float(selected_alpha),
                "policy_val/selected_apply_confidence_threshold": float(selected_confidence_threshold),
                "policy_val/selected_apply_gate_strength": float(selected_gate_strength),
                "policy_val/selected_face_reliability_threshold": float(selected_face_reliability_threshold),
                "calibration/face_reliability_enabled": float(bool(face_reliability_payload.get("enabled", False))),
                "calibration/positive_face_fraction": float(
                    face_reliability_payload.get("positive_face_fraction", 0.0)
                ),
                "surface_texture/enabled": float(bool(surface_feature_texture is not None)),
                "surface_texture/covered_bin_fraction": float(
                    surface_texture_payload.get("covered_bin_fraction", 0.0)
                ),
                "surface_texture/covered_face_fraction": float(
                    surface_texture_payload.get("covered_face_fraction", 0.0)
                ),
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
                "selected_face_reliability_threshold": selected_face_reliability_threshold,
                "best": best,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
