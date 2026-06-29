#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


DEFAULT_EVIDENCE = "/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence"


_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class MemoryEntry:
    dirs: np.ndarray
    residuals: np.ndarray
    l1: np.ndarray
    reliability: float = 0.0
    calibration_count: int = 0
    calibration_error: float = 0.0
    calibration_cosine: float = 0.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
    return float("inf") if mse <= 1.0e-12 else float(-10.0 * math.log10(mse))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_float_grid(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return sorted(set(out))


def _policy_split(paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    fit, val = [], []
    for idx, path in enumerate(paths):
        if int(stride) > 1 and idx % int(stride) == 0:
            val.append(path)
        else:
            fit.append(path)
    return fit, val


def _calibration_split(paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    memory, calib = [], []
    for idx, path in enumerate(paths):
        if int(stride) > 1 and idx % int(stride) == 0:
            calib.append(path)
        else:
            memory.append(path)
    if not calib and memory:
        calib = memory[-1:]
        memory = memory[:-1] or calib
    return memory, calib


def _camera_dir(z: np.lib.npyio.NpzFile) -> np.ndarray:
    cam = np.asarray(z["camera_center"], dtype=np.float32).reshape(3)
    return (cam / max(float(np.linalg.norm(cam)), 1.0e-8)).astype(np.float32)


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


def _sample_positions(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if int(max_samples) > 0 and ys.size > int(max_samples):
        take = rng.choice(ys.size, size=int(max_samples), replace=False)
        return ys[take], xs[take]
    return ys, xs


def _uv_bins(z: np.lib.npyio.NpzFile, ys: np.ndarray, xs: np.ndarray, texture_size: int) -> tuple[np.ndarray, np.ndarray]:
    bary = np.asarray(z["barycentric"], dtype=np.float32)
    u = np.clip(bary[1, ys, xs], 0.0, 1.0)
    v = np.clip(bary[2, ys, xs], 0.0, 1.0)
    bins_u = np.clip(np.floor(u * int(texture_size)).astype(np.int32), 0, int(texture_size) - 1)
    bins_v = np.clip(np.floor(v * int(texture_size)).astype(np.int32), 0, int(texture_size) - 1)
    return bins_u, bins_v


def _rank_candidate_faces(
    paths: list[Path],
    *,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_faces: int,
    max_samples_per_view: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    total = 0
    for path in tqdm(paths, desc="rank train-fit faces"):
        z = np.load(path)
        mask = _valid_mask(z, None, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
        ys, xs = np.nonzero(mask)
        ys, xs = _sample_positions(ys, xs, max_samples=max_samples_per_view, rng=rng)
        if ys.size == 0:
            continue
        face = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
        total += int(face.size)
        for f in np.unique(face):
            fm = face == int(f)
            sums[int(f)] = sums.get(int(f), 0.0) + float(np.sum(l1[fm]))
            counts[int(f)] = counts.get(int(f), 0) + int(np.count_nonzero(fm))
    ranked = sorted(sums, key=lambda f: sums[f], reverse=True)
    if int(max_faces) > 0:
        ranked = ranked[: int(max_faces)]
    faces = np.asarray(sorted(ranked), dtype=np.int64)
    return faces, {
        "ranked_faces": int(len(sums)),
        "selected_faces": int(faces.size),
        "total_sampled_pixels": int(total),
        "max_faces": int(max_faces),
    }


def _insert_topk(bucket: list[tuple[float, np.ndarray, np.ndarray]], item: tuple[float, np.ndarray, np.ndarray], k: int) -> None:
    bucket.append(item)
    if len(bucket) > int(k):
        bucket.sort(key=lambda row: row[0], reverse=True)
        del bucket[int(k) :]


def build_memory(
    paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    texture_size: int,
    k_per_bin: int,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    seed: int,
) -> tuple[dict[tuple[int, int, int], MemoryEntry], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    raw: dict[tuple[int, int, int], list[tuple[float, np.ndarray, np.ndarray]]] = defaultdict(list)
    sampled = 0
    for path in tqdm(paths, desc="build surface memory"):
        z = np.load(path)
        mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
        ys, xs = np.nonzero(mask)
        ys, xs = _sample_positions(ys, xs, max_samples=max_samples_per_view, rng=rng)
        if ys.size == 0:
            continue
        cam = _camera_dir(z)
        faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        bu, bv = _uv_bins(z, ys, xs, int(texture_size))
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        l1 = np.asarray(z[residual_l1_key], dtype=np.float32)[ys, xs]
        sampled += int(ys.size)
        for i in range(int(ys.size)):
            key = (int(faces[i]), int(bu[i]), int(bv[i]))
            rgb = residual[:, ys[i], xs[i]].astype(np.float32)
            _insert_topk(raw[key], (float(l1[i]), cam.copy(), rgb), int(k_per_bin))
    memory: dict[tuple[int, int, int], MemoryEntry] = {}
    entries = 0
    for key, rows in raw.items():
        rows.sort(key=lambda row: row[0], reverse=True)
        l1 = np.asarray([r[0] for r in rows], dtype=np.float32)
        dirs = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
        residuals = np.stack([r[2] for r in rows], axis=0).astype(np.float32)
        memory[key] = MemoryEntry(dirs=dirs, residuals=residuals, l1=l1)
        entries += int(len(rows))
    return memory, {
        "memory_views": int(len(paths)),
        "memory_keys": int(len(memory)),
        "memory_entries": int(entries),
        "sampled_pixels": int(sampled),
        "texture_size": int(texture_size),
        "k_per_bin": int(k_per_bin),
    }


def _predict_entry(
    entry: MemoryEntry,
    cam: np.ndarray,
    *,
    knn_k: int,
    tau: float,
    min_view_cosine: float,
    agreement_power: float,
    reliability_power: float,
    chroma_shrink: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    dirs = entry.dirs
    residuals = entry.residuals
    sims = np.clip(dirs @ cam.reshape(3), -1.0, 1.0)
    order = np.argsort(sims)[::-1]
    if int(knn_k) > 0:
        order = order[: int(knn_k)]
    sims_k = sims[order]
    residuals_k = residuals[order]
    if float(np.max(sims_k)) < float(min_view_cosine):
        return np.zeros(3, dtype=np.float32), 0.0, {"max_cosine": float(np.max(sims_k)), "agreement": 0.0}
    logits = (sims_k - float(np.max(sims_k))) / max(float(tau), 1.0e-6)
    weights = np.exp(logits).astype(np.float32)
    weights /= max(float(np.sum(weights)), 1.0e-8)
    pred = np.sum(residuals_k * weights[:, None], axis=0).astype(np.float32)
    spread = float(np.mean(np.linalg.norm(residuals_k - pred.reshape(1, 3), axis=1)))
    pred_norm = float(np.linalg.norm(pred))
    agreement = float(np.clip(1.0 - spread / max(pred_norm + 1.0e-4, 1.0e-4), 0.0, 1.0))
    sim_conf = float(np.clip((float(np.max(sims_k)) - float(min_view_cosine)) / max(1.0 - float(min_view_cosine), 1.0e-6), 0.0, 1.0))
    rel = float(np.clip(entry.reliability, 0.0, 1.0)) ** float(reliability_power)
    conf = sim_conf * (agreement ** float(agreement_power)) * rel
    if float(chroma_shrink) < 1.0:
        lum = float(np.dot(pred, _LUMA))
        luma_rgb = np.asarray([lum, lum, lum], dtype=np.float32)
        pred = luma_rgb + float(chroma_shrink) * (pred - luma_rgb)
    return pred.astype(np.float32), float(conf), {
        "max_cosine": float(np.max(sims_k)),
        "agreement": float(agreement),
        "sim_confidence": float(sim_conf),
        "reliability": float(entry.reliability),
    }


def calibrate_memory(
    memory: dict[tuple[int, int, int], MemoryEntry],
    paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    texture_size: int,
    residual_rgb_key: str,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    max_samples_per_view: int,
    knn_k: int,
    tau: float,
    min_view_cosine: float,
    agreement_power: float,
    chroma_shrink: float,
    uncalibrated_reliability: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    stats: dict[tuple[int, int, int], dict[str, float]] = defaultdict(lambda: {"w": 0.0, "err": 0.0, "norm": 0.0, "cos": 0.0})
    groups_seen = 0
    for path in tqdm(paths, desc="calibrate memory"):
        z = np.load(path)
        mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
        ys, xs = np.nonzero(mask)
        ys, xs = _sample_positions(ys, xs, max_samples=max_samples_per_view, rng=rng)
        if ys.size == 0:
            continue
        cam = _camera_dir(z)
        faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
        bu, bv = _uv_bins(z, ys, xs, int(texture_size))
        residual = np.asarray(z[residual_rgb_key], dtype=np.float32)
        grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i in range(int(ys.size)):
            grouped[(int(faces[i]), int(bu[i]), int(bv[i]))].append(i)
        for key, idxs in grouped.items():
            entry = memory.get(key)
            if entry is None:
                continue
            target = np.stack([residual[:, ys[i], xs[i]] for i in idxs], axis=0).astype(np.float32).mean(axis=0)
            pred, conf, _ = _predict_entry(
                entry,
                cam,
                knn_k=knn_k,
                tau=tau,
                min_view_cosine=min_view_cosine,
                agreement_power=agreement_power,
                reliability_power=0.0,
                chroma_shrink=chroma_shrink,
            )
            if conf <= 0.0:
                continue
            err = float(np.mean(np.abs(pred - target)))
            norm = float(np.mean(np.abs(target)))
            denom = max(float(np.linalg.norm(pred) * np.linalg.norm(target)), 1.0e-8)
            cos = float(np.dot(pred, target) / denom) if denom > 0.0 else 0.0
            weight = float(len(idxs))
            st = stats[key]
            st["w"] += weight
            st["err"] += weight * err
            st["norm"] += weight * norm
            st["cos"] += weight * cos
            groups_seen += 1
    calibrated = 0
    rel_values: list[float] = []
    for key, entry in memory.items():
        st = stats.get(key)
        if not st or st["w"] <= 0.0:
            entry.reliability = float(uncalibrated_reliability)
            continue
        err = st["err"] / st["w"]
        norm = st["norm"] / st["w"]
        cos = st["cos"] / st["w"]
        rel_err = float(np.clip(1.0 - err / max(norm, 1.0e-4), 0.0, 1.0))
        rel_cos = float(np.clip(0.5 * (cos + 1.0), 0.0, 1.0))
        rel_count = float(np.clip(math.log1p(st["w"]) / math.log1p(1024.0), 0.0, 1.0))
        entry.reliability = float(rel_err * rel_cos * rel_count)
        entry.calibration_count = int(st["w"])
        entry.calibration_error = float(err)
        entry.calibration_cosine = float(cos)
        rel_values.append(float(entry.reliability))
        calibrated += 1
    return {
        "calibration_views": int(len(paths)),
        "calibrated_keys": int(calibrated),
        "calibration_groups_seen": int(groups_seen),
        "mean_reliability": float(np.mean(rel_values)) if rel_values else 0.0,
        "median_reliability": float(np.median(rel_values)) if rel_values else 0.0,
        "uncalibrated_reliability": float(uncalibrated_reliability),
    }


def predict_delta_image(
    memory: dict[tuple[int, int, int], MemoryEntry],
    z: np.lib.npyio.NpzFile,
    candidate_faces: np.ndarray,
    *,
    texture_size: int,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    knn_k: int,
    tau: float,
    min_view_cosine: float,
    agreement_power: float,
    reliability_power: float,
    chroma_shrink: float,
    max_abs_delta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    parent = np.asarray(z["rgb_render"], dtype=np.float32)
    delta = np.zeros_like(parent, dtype=np.float32)
    mask = _valid_mask(z, candidate_faces, residual_l1_key=residual_l1_key, min_l1=min_l1, min_alpha=min_alpha)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return delta, {"candidate_pixels": 0, "applied_pixels": 0}
    cam = _camera_dir(z)
    faces = np.asarray(z["face_id"], dtype=np.int64)[ys, xs]
    bu, bv = _uv_bins(z, ys, xs, int(texture_size))
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(int(ys.size)):
        grouped[(int(faces[i]), int(bu[i]), int(bv[i]))].append(i)
    applied = 0
    conf_values: list[float] = []
    norm_values: list[float] = []
    for key, idxs in grouped.items():
        entry = memory.get(key)
        if entry is None:
            continue
        pred, conf, _ = _predict_entry(
            entry,
            cam,
            knn_k=knn_k,
            tau=tau,
            min_view_cosine=min_view_cosine,
            agreement_power=agreement_power,
            reliability_power=reliability_power,
            chroma_shrink=chroma_shrink,
        )
        if conf <= 0.0:
            continue
        pred = np.clip(pred * float(conf), -float(max_abs_delta), float(max_abs_delta)).astype(np.float32)
        if not np.any(np.abs(pred) > 0.0):
            continue
        idx_arr = np.asarray(idxs, dtype=np.int64)
        delta[:, ys[idx_arr], xs[idx_arr]] = pred.reshape(3, 1)
        applied += int(idx_arr.size)
        conf_values.append(float(conf))
        norm_values.append(float(np.mean(np.abs(pred))))
    return delta, {
        "candidate_pixels": int(ys.size),
        "applied_pixels": int(applied),
        "changed_fraction": float(applied / max(int(parent.shape[1] * parent.shape[2]), 1)),
        "mean_confidence": float(np.mean(conf_values)) if conf_values else 0.0,
        "mean_abs_delta": float(np.mean(norm_values)) if norm_values else 0.0,
        "memory_bins_touched": int(len(grouped)),
    }


def _cvar(values: list[float], fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=np.float32))
    n = max(1, int(math.ceil(float(fraction) * arr.size)))
    return float(np.mean(arr[:n]))


def evaluate_policy_val(
    memory: dict[tuple[int, int, int], MemoryEntry],
    val_paths: list[Path],
    candidate_faces: np.ndarray,
    *,
    texture_size: int,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    knn_k: int,
    tau: float,
    min_view_cosine: float,
    agreement_power: float,
    reliability_power: float,
    chroma_shrink: float,
    max_abs_delta: float,
    alpha_grid: list[float],
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    lpips_model = build_lpips_model() if compute_lpips else None
    rows_by_alpha: dict[float, list[dict[str, Any]]] = {float(a): [] for a in alpha_grid}
    delta_cache: dict[str, tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]] = {}
    for path in tqdm(val_paths, desc="evaluate policy-val memory"):
        z = np.load(path)
        parent = np.asarray(z["rgb_render"], dtype=np.float32)
        gt = np.asarray(z["rgb_gt"], dtype=np.float32)
        delta, stats = predict_delta_image(
            memory,
            z,
            candidate_faces,
            texture_size=texture_size,
            residual_l1_key=residual_l1_key,
            min_l1=min_l1,
            min_alpha=min_alpha,
            knn_k=knn_k,
            tau=tau,
            min_view_cosine=min_view_cosine,
            agreement_power=agreement_power,
            reliability_power=reliability_power,
            chroma_shrink=chroma_shrink,
            max_abs_delta=max_abs_delta,
        )
        delta_cache[path.stem] = (delta, stats, parent, gt)
        p_psnr = _psnr(parent, gt)
        p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
        p_lp = image_lpips_chw(parent, gt, int(lpips_max_side), lpips_model) if compute_lpips else None
        for alpha in alpha_grid:
            adapted = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
            c_psnr = _psnr(adapted, gt)
            c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
            row = {
                "view": path.stem,
                "parent_psnr": float(p_psnr),
                "candidate_psnr": float(c_psnr),
                "psnr_gain": float(c_psnr - p_psnr),
                "parent_ssim": float(p_ssim),
                "candidate_ssim": float(c_ssim),
                "ssim_gain": float(c_ssim - p_ssim),
                **stats,
            }
            if compute_lpips:
                c_lp = image_lpips_chw(adapted, gt, int(lpips_max_side), lpips_model)
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
            "min_psnr_gain": float(np.min(psnr_gain)),
            "min_ssim_gain": float(np.min(ssim_gain)),
            "cvar20_psnr_gain": _cvar(psnr_gain),
            "cvar20_ssim_gain": _cvar(ssim_gain),
            "mean_changed_fraction": float(np.mean([r.get("changed_fraction", 0.0) for r in rows])),
            "mean_confidence": float(np.mean([r.get("mean_confidence", 0.0) for r in rows])),
            "mean_abs_delta": float(np.mean([r.get("mean_abs_delta", 0.0) for r in rows])),
        }
        if compute_lpips:
            lpips_gain = [float(r["lpips_gain"]) for r in rows]
            summary.update(
                {
                    "parent_lpips": float(np.mean([r["parent_lpips"] for r in rows])),
                    "candidate_lpips": float(np.mean([r["candidate_lpips"] for r in rows])),
                    "lpips_gain": float(np.mean(lpips_gain)),
                    "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)),
                    "min_lpips_gain": float(np.min(lpips_gain)),
                    "cvar20_lpips_gain": _cvar(lpips_gain),
                }
            )
        summaries.append(summary)
    best = max(
        summaries,
        key=lambda row: (
            float(row.get("psnr_gain", 0.0))
            + 20.0 * float(row.get("ssim_gain", 0.0))
            + 20.0 * float(row.get("lpips_gain", 0.0))
        ),
    )
    all_axis = [
        row
        for row in summaries
        if float(row.get("psnr_gain", 0.0)) > 0.0
        and float(row.get("ssim_gain", 0.0)) > 0.0
        and (not compute_lpips or float(row.get("lpips_gain", 0.0)) > 0.0)
    ]
    best_all_axis = None
    if all_axis:
        best_all_axis = max(
            all_axis,
            key=lambda row: (
                float(row.get("psnr_gain", 0.0))
                + 20.0 * float(row.get("ssim_gain", 0.0))
                + 20.0 * float(row.get("lpips_gain", 0.0))
            ),
        )
    if output_dir is not None:
        render_dir = output_dir / "renders"
        gt_dir = output_dir / "gt"
        parent_dir = output_dir / "parent"
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        parent_dir.mkdir(parents=True, exist_ok=True)
        alpha = float((best_all_axis or best)["alpha"])
        for stem, (delta, _stats, parent, gt) in delta_cache.items():
            save_image_chw(render_dir / f"{stem}.png", np.clip(parent + alpha * delta, 0.0, 1.0))
            save_image_chw(gt_dir / f"{stem}.png", gt)
            save_image_chw(parent_dir / f"{stem}.png", parent)
    return {
        "best": best,
        "best_all_axis": best_all_axis,
        "rows": summaries,
        "per_view_by_alpha": {str(k): v for k, v in rows_by_alpha.items()},
    }


def verify_target_no_gt(evidence_dir: Path) -> dict[str, Any]:
    forbidden = {
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
    bad: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    paths = evidence_views(evidence_dir)
    for path in paths:
        z = np.load(path)
        keys = set(z.files)
        present = sorted(keys & forbidden)
        if len(samples) < 4:
            samples.append({"path": str(path), "keys": sorted(keys)})
        if present:
            bad.append({"path": str(path), "forbidden_keys": present})
    return {
        "schema": "spcarnet_target_no_gt_verify_v183",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(forbidden),
        "bad_view_count": int(len(bad)),
        "bad_views": bad[:32],
        "sample_keys": samples,
        "target_gt_visible_to_apply": bool(any("rgb_gt" in row.get("forbidden_keys", []) for row in bad)),
        "target_residual_visible_to_apply": bool(
            any(
                set(row.get("forbidden_keys", []))
                & {
                    "residual_rgb",
                    "residual_l1",
                    "teacher_residual_rgb",
                    "teacher_residual_rgb_raw",
                    "teacher_residual_l1",
                }
                for row in bad
            )
        ),
        "passed": len(bad) == 0,
    }


def apply_target(
    memory: dict[tuple[int, int, int], MemoryEntry],
    target_evidence_dir: Path,
    target_eval_evidence_dir: Path | None,
    candidate_faces: np.ndarray,
    *,
    method_name: str,
    alpha: float,
    texture_size: int,
    residual_l1_key: str,
    min_l1: float,
    min_alpha: float,
    knn_k: int,
    tau: float,
    min_view_cosine: float,
    agreement_power: float,
    reliability_power: float,
    chroma_shrink: float,
    max_abs_delta: float,
    ssim_max_side: int,
    lpips_max_side: int,
    compute_lpips: bool,
    output_dir: Path,
) -> dict[str, Any]:
    target_paths = evidence_views(target_evidence_dir)
    eval_paths = {p.stem: p for p in evidence_views(target_eval_evidence_dir)} if target_eval_evidence_dir else {}
    out_root = output_dir / "flowers_exact_target_apply"
    render_dir = out_root / "test" / method_name / "renders"
    parent_dir = out_root / "test" / method_name / "parent"
    gt_dir = out_root / "test" / method_name / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    if eval_paths:
        gt_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = build_lpips_model() if compute_lpips and eval_paths else None
    rows: list[dict[str, Any]] = []
    for path in tqdm(target_paths, desc="apply target no-GT memory"):
        z = np.load(path)
        parent = np.asarray(z["rgb_render"], dtype=np.float32)
        delta, stats = predict_delta_image(
            memory,
            z,
            candidate_faces,
            texture_size=int(texture_size),
            residual_l1_key=residual_l1_key,
            min_l1=min_l1,
            min_alpha=min_alpha,
            knn_k=knn_k,
            tau=tau,
            min_view_cosine=min_view_cosine,
            agreement_power=agreement_power,
            reliability_power=reliability_power,
            chroma_shrink=chroma_shrink,
            max_abs_delta=max_abs_delta,
        )
        adapted = np.clip(parent + float(alpha) * delta, 0.0, 1.0).astype(np.float32)
        save_image_chw(render_dir / f"{path.stem}.png", adapted)
        save_image_chw(parent_dir / f"{path.stem}.png", parent)
        row: dict[str, Any] = {"view": path.stem, **stats}
        eval_path = eval_paths.get(path.stem)
        if eval_path is not None:
            ez = np.load(eval_path)
            gt = np.asarray(ez["rgb_gt"], dtype=np.float32)
            save_image_chw(gt_dir / f"{path.stem}.png", gt)
            p_psnr = _psnr(parent, gt)
            c_psnr = _psnr(adapted, gt)
            p_ssim = image_ssim_chw(parent, gt, int(ssim_max_side))
            c_ssim = image_ssim_chw(adapted, gt, int(ssim_max_side))
            row.update(
                {
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                }
            )
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
    summary: dict[str, Any] = {
        "method_name": str(method_name),
        "alpha": float(alpha),
        "target_evidence_dir": str(target_evidence_dir),
        "target_eval_evidence_dir": "" if target_eval_evidence_dir is None else str(target_eval_evidence_dir),
        "render_dir": str(render_dir),
        "parent_dir": str(parent_dir),
        "gt_dir": str(gt_dir) if eval_paths else "",
        "view_count": int(len(rows)),
        "mean_changed_fraction": float(np.mean([r.get("changed_fraction", 0.0) for r in rows])) if rows else 0.0,
        "mean_confidence": float(np.mean([r.get("mean_confidence", 0.0) for r in rows])) if rows else 0.0,
        "mean_abs_delta": float(np.mean([r.get("mean_abs_delta", 0.0) for r in rows])) if rows else 0.0,
        "per_view": rows,
    }
    metric_rows = [r for r in rows if "candidate_psnr" in r]
    if metric_rows:
        psnr_gain = [float(r["psnr_gain"]) for r in metric_rows]
        ssim_gain = [float(r["ssim_gain"]) for r in metric_rows]
        summary.update(
            {
                "parent_psnr": float(np.mean([r["parent_psnr"] for r in metric_rows])),
                "candidate_psnr": float(np.mean([r["candidate_psnr"] for r in metric_rows])),
                "psnr_gain": float(np.mean(psnr_gain)),
                "parent_ssim": float(np.mean([r["parent_ssim"] for r in metric_rows])),
                "candidate_ssim": float(np.mean([r["candidate_ssim"] for r in metric_rows])),
                "ssim_gain": float(np.mean(ssim_gain)),
                "positive_view_fraction": float(np.mean(np.asarray(psnr_gain) > 0.0)),
                "ssim_positive_view_fraction": float(np.mean(np.asarray(ssim_gain) > 0.0)),
                "min_psnr_gain": float(np.min(psnr_gain)),
                "min_ssim_gain": float(np.min(ssim_gain)),
                "cvar20_psnr_gain": _cvar(psnr_gain),
                "cvar20_ssim_gain": _cvar(ssim_gain),
            }
        )
        if compute_lpips and "candidate_lpips" in metric_rows[0]:
            lpips_gain = [float(r["lpips_gain"]) for r in metric_rows]
            summary.update(
                {
                    "parent_lpips": float(np.mean([r["parent_lpips"] for r in metric_rows])),
                    "candidate_lpips": float(np.mean([r["candidate_lpips"] for r in metric_rows])),
                    "lpips_gain": float(np.mean(lpips_gain)),
                    "lpips_positive_view_fraction": float(np.mean(np.asarray(lpips_gain) > 0.0)),
                    "min_lpips_gain": float(np.min(lpips_gain)),
                    "cvar20_lpips_gain": _cvar(lpips_gain),
                }
            )
    return summary


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload["policy_val"]["best"]
    best_all = payload["policy_val"].get("best_all_axis")
    target_apply = payload.get("target_apply") or {}
    lines = [
        "# v183 Self-Certified Surface Residual Memory Probe",
        "",
        "This probe implements a single representation change: a train-only cross-view surface residual memory.",
        "Phase-J teacher residuals are stored per face/UV bin with view-direction samples, then calibrated on a held-out train-fit split before policy-val evaluation.",
        "",
        "## Verdict",
        "",
        f"- policy-val all-axis pass: `{payload['policy_val_all_axis_pass']}`",
        f"- flowers exact promotion allowed: `{payload['flowers_exact_promotion_allowed']}`",
        f"- interpretation: {payload['interpretation']}",
        "",
        "## Best Policy-Val Row",
        "",
        "| alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR pos | SSIM pos | LPIPS pos | changed frac | conf |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best.get('alpha', 0.0):.6f} | {best.get('psnr_gain', 0.0):+.9f} | "
            f"{best.get('ssim_gain', 0.0):+.9f} | {best.get('lpips_gain', 0.0):+.9f} | "
            f"{best.get('positive_view_fraction', 0.0):.3f} | {best.get('ssim_positive_view_fraction', 0.0):.3f} | "
            f"{best.get('lpips_positive_view_fraction', 0.0):.3f} | {best.get('mean_changed_fraction', 0.0):.8f} | "
            f"{best.get('mean_confidence', 0.0):.6f} |"
        ),
        "",
        f"- best all-axis row: `{None if best_all is None else {k: best_all[k] for k in best_all if k != 'per_view'}}`",
        "",
        "## Representation Summary",
        "",
        f"- selected faces: `{payload['candidate_face_summary']['selected_faces']}`",
        f"- memory keys: `{payload['memory_summary']['memory_keys']}`",
        f"- memory entries: `{payload['memory_summary']['memory_entries']}`",
        f"- calibrated keys: `{payload['calibration_summary']['calibrated_keys']}`",
        f"- mean reliability: `{payload['calibration_summary']['mean_reliability']:.6f}`",
        "",
        "## v169 Gate Context",
        "",
        f"- v168 flowers exact: `{payload['references']['v168_flowers_exact']}`",
        f"- Phase-J flowers gate: `{payload['references']['phasej_flowers_gate']}`",
        "",
        "## Flowers Exact Target Apply",
        "",
        f"- executed: `{bool(target_apply)}`",
        f"- no-GT verifier passed: `{payload.get('target_no_gt_verify', {}).get('passed')}`",
        f"- selected target alpha: `{target_apply.get('alpha')}`",
        f"- target metrics: `{target_apply.get('candidate_psnr')} / {target_apply.get('candidate_ssim')} / {target_apply.get('candidate_lpips')}`",
        f"- target gains vs parent: `{target_apply.get('psnr_gain')} / {target_apply.get('ssim_gain')} / {target_apply.get('lpips_gain')}`",
        f"- target render dir: `{target_apply.get('render_dir')}`",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{payload['output_json']}`",
        f"- policy-val renders: `{payload['output_render_dir']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe self-certified surface residual memory for Phase-J teacher distillation.")
    parser.add_argument("--fit_evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--calibration_stride", type=int, default=5)
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument("--max_faces", type=int, default=512)
    parser.add_argument("--k_per_bin", type=int, default=8)
    parser.add_argument("--knn_k", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--min_view_cosine", type=float, default=-1.0)
    parser.add_argument("--agreement_power", type=float, default=1.0)
    parser.add_argument("--reliability_power", type=float, default=1.0)
    parser.add_argument("--uncalibrated_reliability", type=float, default=0.15)
    parser.add_argument("--chroma_shrink", type=float, default=0.35)
    parser.add_argument("--max_abs_delta", type=float, default=0.12)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--max_rank_samples_per_view", type=int, default=32768)
    parser.add_argument("--max_memory_samples_per_view", type=int, default=65536)
    parser.add_argument("--max_calibration_samples_per_view", type=int, default=32768)
    parser.add_argument("--alpha_grid", default="0,0.005,0.01,0.015625,0.03125,0.0625,0.125,0.25")
    parser.add_argument("--ssim_max_side", type=int, default=512)
    parser.add_argument("--lpips_max_side", type=int, default=256)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--output_dir", default="/tmp/peilincai_spcarnet_v183_self_certified_memory")
    parser.add_argument("--skip_policy_val_renders", action="store_true")
    parser.add_argument("--target_evidence_dir", default="")
    parser.add_argument("--target_eval_evidence_dir", default="")
    parser.add_argument("--target_alpha", type=float, default=None)
    parser.add_argument("--allow_target_without_all_axis", action="store_true")
    parser.add_argument("--method_name", default="ours_26000_v183_self_certified_memory_flowers")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-v183-self-certified-memory")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--seed", type=int, default=183)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
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
    memory_paths, calibration_paths = _calibration_split(fit_paths, int(args.calibration_stride))
    candidate_faces, face_summary = _rank_candidate_faces(
        fit_paths,
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_faces=int(args.max_faces),
        max_samples_per_view=int(args.max_rank_samples_per_view),
        seed=int(args.seed),
    )
    if candidate_faces.size <= 0:
        raise RuntimeError("no candidate faces selected")
    memory, memory_summary = build_memory(
        memory_paths,
        candidate_faces,
        texture_size=int(args.texture_size),
        k_per_bin=int(args.k_per_bin),
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_samples_per_view=int(args.max_memory_samples_per_view),
        seed=int(args.seed) + 1,
    )
    calibration_summary = calibrate_memory(
        memory,
        calibration_paths,
        candidate_faces,
        texture_size=int(args.texture_size),
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_samples_per_view=int(args.max_calibration_samples_per_view),
        knn_k=int(args.knn_k),
        tau=float(args.tau),
        min_view_cosine=float(args.min_view_cosine),
        agreement_power=float(args.agreement_power),
        chroma_shrink=float(args.chroma_shrink),
        uncalibrated_reliability=float(args.uncalibrated_reliability),
        seed=int(args.seed) + 2,
    )
    policy_val = evaluate_policy_val(
        memory,
        val_paths,
        candidate_faces,
        texture_size=int(args.texture_size),
        residual_l1_key=str(args.residual_l1_key),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        knn_k=int(args.knn_k),
        tau=float(args.tau),
        min_view_cosine=float(args.min_view_cosine),
        agreement_power=float(args.agreement_power),
        reliability_power=float(args.reliability_power),
        chroma_shrink=float(args.chroma_shrink),
        max_abs_delta=float(args.max_abs_delta),
        alpha_grid=_parse_float_grid(str(args.alpha_grid)),
        ssim_max_side=int(args.ssim_max_side),
        lpips_max_side=int(args.lpips_max_side),
        compute_lpips=bool(args.compute_lpips),
        output_dir=None if bool(args.skip_policy_val_renders) else output_dir / "policy_val_best",
    )
    all_axis = policy_val.get("best_all_axis") is not None
    target_apply: dict[str, Any] | None = None
    target_no_gt_verify: dict[str, Any] | None = None
    if str(args.target_evidence_dir):
        if all_axis or bool(args.allow_target_without_all_axis):
            target_dir = Path(str(args.target_evidence_dir))
            target_no_gt_verify = verify_target_no_gt(target_dir)
            if not bool(target_no_gt_verify.get("passed")):
                raise RuntimeError(f"target evidence no-GT verification failed: {target_no_gt_verify}")
            selected_alpha = (
                float(args.target_alpha)
                if args.target_alpha is not None
                else float((policy_val.get("best_all_axis") or policy_val["best"])["alpha"])
            )
            target_apply = apply_target(
                memory,
                target_dir,
                Path(str(args.target_eval_evidence_dir)) if str(args.target_eval_evidence_dir) else None,
                candidate_faces,
                method_name=str(args.method_name),
                alpha=selected_alpha,
                texture_size=int(args.texture_size),
                residual_l1_key=str(args.residual_l1_key),
                min_l1=float(args.min_l1),
                min_alpha=float(args.min_alpha),
                knn_k=int(args.knn_k),
                tau=float(args.tau),
                min_view_cosine=float(args.min_view_cosine),
                agreement_power=float(args.agreement_power),
                reliability_power=float(args.reliability_power),
                chroma_shrink=float(args.chroma_shrink),
                max_abs_delta=float(args.max_abs_delta),
                ssim_max_side=int(args.ssim_max_side),
                lpips_max_side=int(args.lpips_max_side),
                compute_lpips=bool(args.compute_lpips),
                output_dir=output_dir,
            )
        else:
            target_apply = {
                "skipped": True,
                "reason": "policy-val all-axis gate failed and --allow_target_without_all_axis was not set",
            }
    interpretation = (
        "PASS: the train-only self-certified memory produced a policy-val all-axis row; target apply was run when requested."
        if all_axis
        else "FAIL: cross-view calibrated memory still cannot carry Phase-J residual into a policy-val all-axis improvement; do not promote to exact/full9."
    )
    payload = {
        "schema": "spcarnet_v183_self_certified_surface_residual_memory_probe_v1",
        "args": vars(args),
        "splits": {
            "all_views": int(len(paths)),
            "train_fit_views": int(len(fit_paths)),
            "policy_val_views": int(len(val_paths)),
            "memory_views": int(len(memory_paths)),
            "calibration_views": int(len(calibration_paths)),
            "policy_val_view_names": [p.stem for p in val_paths],
        },
        "candidate_face_summary": face_summary,
        "memory_summary": memory_summary,
        "calibration_summary": calibration_summary,
        "policy_val": policy_val,
        "policy_val_all_axis_pass": bool(all_axis),
        "flowers_exact_promotion_allowed": bool(all_axis),
        "target_no_gt_verify": target_no_gt_verify,
        "target_apply": target_apply,
        "interpretation": interpretation,
        "references": {
            "phasej_flowers_gate": "20.304358 / 0.557770 / 0.329222",
            "v168_flowers_exact": "19.832031 / 0.505779 / 0.405906",
            "v182_best_policy_val": "20.607278 / 0.717525 / 0.153338 at alpha 0.0625, LPIPS negative",
        },
    }
    payload["output_json"] = str(output_dir / "v183_self_certified_memory_probe.json")
    payload["output_render_dir"] = "" if bool(args.skip_policy_val_renders) else str(output_dir / "policy_val_best")
    _write_json(output_dir / "v183_self_certified_memory_probe.json", payload)
    _write_md(output_dir / "v183_self_certified_memory_probe.md", payload)
    if wandb_run is not None:
        best = policy_val["best"]
        wandb_run.log(
            {
                "policy_val/best_psnr_gain": float(best.get("psnr_gain", 0.0)),
                "policy_val/best_ssim_gain": float(best.get("ssim_gain", 0.0)),
                "policy_val/best_lpips_gain": float(best.get("lpips_gain", 0.0)),
                "policy_val/all_axis_pass": int(all_axis),
                "memory/keys": int(memory_summary["memory_keys"]),
                "memory/entries": int(memory_summary["memory_entries"]),
                "calibration/mean_reliability": float(calibration_summary["mean_reliability"]),
            }
        )
        if target_apply and "candidate_psnr" in target_apply:
            wandb_run.log(
                {
                    "target/candidate_psnr": float(target_apply.get("candidate_psnr", 0.0)),
                    "target/candidate_ssim": float(target_apply.get("candidate_ssim", 0.0)),
                    "target/candidate_lpips": float(target_apply.get("candidate_lpips", 0.0)),
                    "target/psnr_gain": float(target_apply.get("psnr_gain", 0.0)),
                    "target/ssim_gain": float(target_apply.get("ssim_gain", 0.0)),
                    "target/lpips_gain": float(target_apply.get("lpips_gain", 0.0)),
                }
            )
        wandb_run.finish()
    print("OUT", output_dir / "v183_self_certified_memory_probe.json", flush=True)
    print("BEST", json.dumps(policy_val["best"], indent=2, sort_keys=True), flush=True)
    print("BEST_ALL_AXIS", json.dumps(policy_val.get("best_all_axis"), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
