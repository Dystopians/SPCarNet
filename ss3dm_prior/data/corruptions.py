"""Online synthetic corruption utilities for teacher patch training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import zlib

import numpy as np
from scipy.spatial import cKDTree


def _rng_from_key(seed: int, key: str) -> np.random.Generator:
    derived_seed = (int(seed) + zlib.crc32(key.encode("utf-8"))) % (2**32)
    return np.random.default_rng(derived_seed)


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return (normals / norm).astype(np.float32)


def _resample_points(
    rng: np.random.Generator,
    points: np.ndarray,
    normals: np.ndarray,
    flags: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return (
            np.zeros((target_count, 3), dtype=np.float32),
            np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (target_count, 1)),
            np.ones((target_count,), dtype=np.float32),
        )
    replace = len(points) < target_count
    indices = rng.choice(len(points), size=target_count, replace=replace)
    return points[indices], normals[indices], flags[indices]


@dataclass
class CorruptionResult:
    corrupted_points: np.ndarray
    corrupted_normals: np.ndarray
    point_defect_target: np.ndarray
    corruption_score_target: float
    metadata: dict[str, Any]


def apply_patch_corruptions(
    *,
    clean_points: np.ndarray,
    clean_normals: np.ndarray,
    observed_points: np.ndarray,
    config: dict[str, Any],
    seed: int,
    sample_key: str,
) -> CorruptionResult:
    clean_points = np.asarray(clean_points, dtype=np.float32)
    clean_normals = _normalize_normals(clean_normals)
    observed_points = np.asarray(observed_points, dtype=np.float32)
    rng = _rng_from_key(seed, sample_key)

    points = clean_points.copy()
    normals = clean_normals.copy()
    flags = np.zeros((len(points),), dtype=np.float32)
    severity_terms: list[float] = []
    metadata: dict[str, Any] = {
        "sample_key": sample_key,
        "enabled_corruptions": [],
    }

    corruption_cfg = config.get("corruptions", config)
    target_corrupted_count = int(corruption_cfg.get("target_corrupted_count", len(clean_points)))

    point_dropout_cfg = corruption_cfg.get("point_dropout", {})
    if point_dropout_cfg.get("enabled", True):
        dropout_ratio = float(point_dropout_cfg.get("dropout_ratio", 0.15))
        keep_mask = rng.random(len(points)) > dropout_ratio
        if np.any(keep_mask):
            points = points[keep_mask]
            normals = normals[keep_mask]
            flags = flags[keep_mask]
        severity_terms.append(dropout_ratio)
        metadata["enabled_corruptions"].append("point_dropout")
        metadata["point_dropout_ratio"] = dropout_ratio

    local_hole_cfg = corruption_cfg.get("local_hole_mask", {})
    if local_hole_cfg.get("enabled", True) and len(points) > 0:
        num_holes = int(rng.integers(1, int(local_hole_cfg.get("max_holes", 2)) + 1))
        hole_radius = float(local_hole_cfg.get("hole_radius", 0.25))
        for _ in range(num_holes):
            center = points[int(rng.integers(len(points)))]
            dist = np.linalg.norm(points - center[None, :], axis=1)
            keep_mask = dist > hole_radius
            if np.any(keep_mask):
                points = points[keep_mask]
                normals = normals[keep_mask]
                flags = flags[keep_mask]
        severity_terms.append(num_holes * hole_radius)
        metadata["enabled_corruptions"].append("local_hole_mask")
        metadata["local_hole_count"] = num_holes

    gaussian_cfg = corruption_cfg.get("gaussian_jitter", {})
    if gaussian_cfg.get("enabled", True) and len(points) > 0:
        sigma = float(gaussian_cfg.get("sigma", 0.03))
        anisotropic = bool(gaussian_cfg.get("anisotropic", True))
        if anisotropic:
            axis_scale = rng.uniform(0.5, 1.5, size=(1, 3)).astype(np.float32)
            jitter = rng.normal(0.0, sigma, size=points.shape).astype(np.float32) * axis_scale
        else:
            jitter = rng.normal(0.0, sigma, size=points.shape).astype(np.float32)
        points = points + jitter
        flags = np.maximum(flags, np.linalg.norm(jitter, axis=1).astype(np.float32))
        severity_terms.append(sigma)
        metadata["enabled_corruptions"].append("gaussian_jitter")
        metadata["gaussian_sigma"] = sigma

    normal_cfg = corruption_cfg.get("normal_noise", {})
    if normal_cfg.get("enabled", True) and len(normals) > 0:
        noise_sigma = float(normal_cfg.get("sigma", 0.1))
        flip_prob = float(normal_cfg.get("flip_prob", 0.05))
        normals = _normalize_normals(normals + rng.normal(0.0, noise_sigma, size=normals.shape))
        flip_mask = rng.random(len(normals)) < flip_prob
        normals[flip_mask] *= -1.0
        flags = np.maximum(flags, flip_mask.astype(np.float32) * 0.5)
        severity_terms.append(noise_sigma + flip_prob)
        metadata["enabled_corruptions"].append("normal_noise")
        metadata["normal_flip_fraction"] = float(np.mean(flip_mask)) if len(flip_mask) else 0.0

    density_cfg = corruption_cfg.get("density_imbalance", {})
    if density_cfg.get("enabled", True) and len(points) > 0:
        region_radius = float(density_cfg.get("region_radius", 0.3))
        center = points[int(rng.integers(len(points)))]
        dist = np.linalg.norm(points - center[None, :], axis=1)
        region_mask = dist < region_radius
        thin_prob = float(density_cfg.get("thin_probability", 0.5))
        if np.any(region_mask) and rng.random() < thin_prob:
            keep_mask = ~region_mask | (rng.random(len(points)) > float(density_cfg.get("thin_ratio", 0.5)))
            if np.any(keep_mask):
                points = points[keep_mask]
                normals = normals[keep_mask]
                flags = flags[keep_mask]
        elif np.any(region_mask):
            region_points = points[region_mask]
            region_normals = normals[region_mask]
            region_flags = np.ones((len(region_points),), dtype=np.float32) * 0.35
            duplicate_count = min(len(region_points), int(density_cfg.get("duplicate_count", 64)))
            if duplicate_count > 0:
                dup_idx = rng.choice(len(region_points), size=duplicate_count, replace=True)
                dup_points = region_points[dup_idx] + rng.normal(0.0, 0.01, size=(duplicate_count, 3)).astype(np.float32)
                points = np.concatenate([points, dup_points.astype(np.float32)], axis=0)
                normals = np.concatenate([normals, region_normals[dup_idx]], axis=0)
                flags = np.concatenate([flags, region_flags[:duplicate_count]], axis=0)
        severity_terms.append(region_radius)
        metadata["enabled_corruptions"].append("density_imbalance")

    outlier_cfg = corruption_cfg.get("outlier_cluster", {})
    if outlier_cfg.get("enabled", True):
        cluster_count = int(outlier_cfg.get("cluster_size", 32))
        cluster_center = observed_points[int(rng.integers(len(observed_points)))] if len(observed_points) else rng.uniform(-0.5, 0.5, size=(3,))
        cluster_offset = rng.normal(0.0, float(outlier_cfg.get("cluster_offset_sigma", 0.5)), size=(1, 3)).astype(np.float32)
        cluster_points = cluster_center[None, :] + cluster_offset + rng.normal(
            0.0,
            float(outlier_cfg.get("cluster_spread_sigma", 0.03)),
            size=(cluster_count, 3),
        ).astype(np.float32)
        cluster_normals = _normalize_normals(rng.normal(0.0, 1.0, size=(cluster_count, 3)).astype(np.float32))
        cluster_flags = np.ones((cluster_count,), dtype=np.float32)
        points = np.concatenate([points, cluster_points.astype(np.float32)], axis=0)
        normals = np.concatenate([normals, cluster_normals], axis=0)
        flags = np.concatenate([flags, cluster_flags], axis=0)
        severity_terms.append(cluster_count / max(target_corrupted_count, 1))
        metadata["enabled_corruptions"].append("outlier_cluster")
        metadata["outlier_cluster_size"] = cluster_count

    points, normals, flags = _resample_points(rng, points, normals, flags, target_corrupted_count)
    normals = _normalize_normals(normals)

    tree = cKDTree(clean_points)
    nn_dist, _ = tree.query(points, k=1)
    point_defect_target = np.asarray(nn_dist, dtype=np.float32) + flags.astype(np.float32)
    corruption_score_target = float(np.mean(np.clip(point_defect_target, 0.0, None)))
    if severity_terms:
        corruption_score_target += float(np.mean(severity_terms))

    metadata["target_corrupted_count"] = target_corrupted_count
    metadata["corruption_score_target"] = corruption_score_target
    metadata["point_defect_target_mean"] = float(np.mean(point_defect_target))
    metadata["point_defect_target_max"] = float(np.max(point_defect_target))
    return CorruptionResult(
        corrupted_points=points.astype(np.float32),
        corrupted_normals=normals.astype(np.float32),
        point_defect_target=point_defect_target.astype(np.float32),
        corruption_score_target=corruption_score_target,
        metadata=metadata,
    )
