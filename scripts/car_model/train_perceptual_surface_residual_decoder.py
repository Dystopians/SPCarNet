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


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
    return float("inf") if mse <= 1.0e-12 else float(-10.0 * math.log10(mse))


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
    def __init__(self, face_count: int, feature_dim: int, embedding_dim: int, hidden_dim: int, layers: int, max_delta: float):
        super().__init__()
        self.face_embedding = torch.nn.Embedding(int(face_count), int(embedding_dim))
        dims = [int(feature_dim) + int(embedding_dim)] + [int(hidden_dim)] * int(layers) + [3]
        blocks: list[torch.nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1], strict=False):
            blocks += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        blocks.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.net = torch.nn.Sequential(*blocks)
        self.max_delta = float(max_delta)

    def forward(self, face_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        emb = self.face_embedding(face_idx)
        return torch.tanh(self.net(torch.cat([features, emb], dim=1))) * self.max_delta


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
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if residual_score is None:
        residual_score = np.mean(np.abs(target), axis=1).astype(np.float32)
    denom = max(float(np.mean(residual_score)), 1.0e-6)
    weights = np.power((residual_score / denom) + 1.0e-6, max(float(sample_weight_gamma), 0.0)).astype(np.float32)
    if float(sample_weight_clip) > 0.0:
        weights = np.clip(weights, 1.0 / float(sample_weight_clip), float(sample_weight_clip))
    weights = weights / max(float(np.mean(weights)), 1.0e-6)
    return face_idx.astype(np.int64), features.astype(np.float32), target, weights.astype(np.float32)


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
    chunk_size: int,
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
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
                    pred = model(face_t, feat).detach().cpu().numpy().astype(np.float32)
                    delta[:, ys[start:end], xs[start:end]] = pred.T
            p_psnr = _psnr(parent, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
            for alpha in alpha_grid:
                adapted = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
                c_psnr = _psnr(adapted, gt)
                c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
                row = {
                    "view": path.stem,
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                }
                if compute_lpips:
                    row.update(
                        {
                            "parent_lpips": float(p_lp),
                            "candidate_lpips": float(c_lp),
                            "lpips_gain": float(p_lp - c_lp),
                        }
                    )
                rows_by_alpha[float(alpha)].append(row)
    summaries: list[dict[str, Any]] = []
    for alpha, rows in rows_by_alpha.items():
        parent_psnr = [r["parent_psnr"] for r in rows]
        cand_psnr = [r["candidate_psnr"] for r in rows]
        parent_ssim = [r["parent_ssim"] for r in rows]
        cand_ssim = [r["candidate_ssim"] for r in rows]
        psnr_gain = [r["psnr_gain"] for r in rows]
        ssim_gain = [r["ssim_gain"] for r in rows]
        summary = {
            "alpha": float(alpha),
            "parent_psnr": float(np.mean(parent_psnr)),
            "candidate_psnr": float(np.mean(cand_psnr)),
            "psnr_gain": float(np.mean(psnr_gain)),
            "parent_ssim": float(np.mean(parent_ssim)),
            "candidate_ssim": float(np.mean(cand_ssim)),
            "ssim_gain": float(np.mean(ssim_gain)),
            "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
            "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
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
        with torch.no_grad():
            for path in tqdm(val_paths, desc="write best policy-val renders"):
                z = np.load(path)
                parent = np.asarray(z["rgb_render"], dtype=np.float32)
                gt = np.asarray(z["rgb_gt"], dtype=np.float32)
                mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
                ys, xs = np.nonzero(mask)
                delta = np.zeros_like(parent, dtype=np.float32)
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
                        pred = model(face_t, feat).detach().cpu().numpy().astype(np.float32)
                        delta[:, ys[start:end], xs[start:end]] = pred.T
                save_image_chw(output_dir / "renders" / f"{path.stem}.png", np.clip(parent + best_alpha * delta, 0.0, 1.0))
                save_image_chw(output_dir / "gt" / f"{path.stem}.png", gt)
    return {
        "best": {k: v for k, v in best.items() if k != "per_view"},
        "best_all_axis": best_all_axis,
        "rows": [{k: v for k, v in row.items() if k != "per_view"} for row in summaries],
        "per_view_by_alpha": {str(k): v for k, v in rows_by_alpha.items()},
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all_axis = payload["policy_val"].get("best_all_axis")
    lines = [
        "# v180 Differentiable Surface Decoder Audit",
        "",
        f"- teacher signal pass: `{payload['teacher_signal_pass']}`",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- train steps: `{payload['train']['steps']}`",
        "",
        "## Best Policy-Val Row",
        "",
        "| alpha | PSNR gain | SSIM gain | LPIPS gain | pos views | SSIM pos | LPIPS pos |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.4f} | {best.get('psnr_gain', 0.0):.6f} | "
            f"{best.get('ssim_gain', 0.0):.6f} | {best.get('lpips_gain', 0.0):.6f} | "
            f"{best.get('positive_view_fraction', 0.0):.3f} | "
            f"{best.get('ssim_positive_view_fraction', 0.0):.3f} | "
            f"{best.get('lpips_positive_view_fraction', 0.0):.3f} |"
        ),
        "",
        f"- best all-axis row: `{best_all_axis}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- best policy-val renders: `{payload['output_render_dir']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a differentiable teacher-residual surface decoder.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
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
    parser.add_argument("--feature_mode", choices=["basic", "fourier_v1"], default="basic")
    parser.add_argument("--image_loss_every", type=int, default=4)
    parser.add_argument("--image_loss_stride", type=int, default=12)
    parser.add_argument("--image_loss_weight", type=float, default=0.35)
    parser.add_argument("--sample_weight_gamma", type=float, default=0.0)
    parser.add_argument("--sample_weight_clip", type=float, default=8.0)
    parser.add_argument("--cosine_loss_weight", type=float, default=0.0)
    parser.add_argument("--energy_match_weight", type=float, default=0.0)
    parser.add_argument("--mag_reg", type=float, default=1.0e-4)
    parser.add_argument("--alpha_grid", default="0,0.0625,0.125,0.25,0.5,0.75,1")
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-5)
    train_rows: list[dict[str, Any]] = []
    fit_cycle = list(fit_paths)
    train_rng = random.Random(int(args.seed))
    for step in tqdm(range(1, int(args.steps) + 1), desc="train neural surface decoder"):
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
                    feature_mode=str(args.feature_mode),
                )
                break
            except RuntimeError:
                continue
        if sampled is None:
            raise RuntimeError("no train-fit view contains the selected candidate faces")
        face_idx, features, target, sample_weights = sampled
        face_t = torch.from_numpy(face_idx).to(device)
        feat_t = torch.from_numpy(features).to(device)
        target_t = torch.from_numpy(target).to(device)
        weight_t = torch.from_numpy(sample_weights).to(device).reshape(-1)
        weight_t = weight_t / torch.clamp(torch.mean(weight_t), min=1.0e-6)
        pred = model(face_t, feat_t)
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
    alpha_grid = sorted({float(x) for x in str(args.alpha_grid).split(",") if x.strip()})
    policy_val = _evaluate(
        model,
        val_paths,
        candidate_faces,
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        feature_mode=str(args.feature_mode),
        alpha_grid=alpha_grid,
        chunk_size=int(args.eval_chunk_size),
        ssim_max_side=int(args.policy_val_ssim_max_side),
        lpips_max_side=int(args.policy_val_lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
        output_dir=render_dir,
        device=device,
    )
    best = policy_val["best"]
    all_axis = policy_val.get("best_all_axis") is not None
    interpretation = (
        "The differentiable decoder passed the policy-val all-axis gate; this justifies a strict flowers exact run."
        if all_axis
        else "The differentiable decoder did not pass the policy-val all-axis gate. It should not be promoted to flowers exact/full9; the next change must further alter the objective, supervision, or representation."
    )
    payload: dict[str, Any] = {
        "schema": "spcarnet_perceptual_surface_decoder_audit_v1",
        "created_at": "2026-06-29",
        "device": str(device),
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "fit_views": len(fit_paths),
        "policy_val_views": len(val_paths),
        "candidate_face_summary": face_summary,
        "train": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "feature_mode": str(args.feature_mode),
            "feature_dim": int(_feature_dim(str(args.feature_mode))),
            "image_loss_every": int(args.image_loss_every),
            "image_loss_stride": int(args.image_loss_stride),
            "image_loss_weight": float(args.image_loss_weight),
            "sample_weight_gamma": float(args.sample_weight_gamma),
            "sample_weight_clip": float(args.sample_weight_clip),
            "cosine_loss_weight": float(args.cosine_loss_weight),
            "energy_match_weight": float(args.energy_match_weight),
            "rows": train_rows,
        },
        "teacher_signal_pass": True,
        "uses_train_fit_teacher": True,
        "uses_policy_val_gt": True,
        "uses_target_or_test_gt": False,
        "policy_val_all_axis_pass": all_axis,
        "policy_val": policy_val,
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
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_json": payload["output_json"],
                "output_md": str(output_dir / "v180_perceptual_surface_decoder_audit.md"),
                "policy_val_all_axis_pass": all_axis,
                "best": best,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
