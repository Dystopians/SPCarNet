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


def _schedule_alpha(schedule_type: str, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    normalized = str(schedule_type).strip().lower()
    if normalized == "linear":
        return progress
    if normalized == "cosine":
        return float(0.5 - 0.5 * np.cos(np.pi * progress))
    raise ValueError(f"Unsupported severity_schedule.type: {schedule_type}")


def resolve_corruption_severity_scale(
    config: dict[str, Any],
    *,
    epoch: int | None = None,
) -> float:
    corruption_cfg = config.get("corruptions", config)
    schedule_cfg = corruption_cfg.get("severity_schedule", {}) or {}
    if not schedule_cfg:
        return 1.0
    if epoch is None:
        return float(schedule_cfg.get("end_scale", schedule_cfg.get("start_scale", 1.0)))
    warmup_epochs = max(int(schedule_cfg.get("warmup_epochs", 0) or 0), 0)
    start_scale = float(schedule_cfg.get("start_scale", 1.0))
    end_scale = float(schedule_cfg.get("end_scale", 1.0))
    if warmup_epochs <= 0:
        return end_scale
    alpha = _schedule_alpha(str(schedule_cfg.get("type", "linear")), float(epoch) / float(warmup_epochs))
    return float(start_scale + (end_scale - start_scale) * alpha)


def _scaled_count(value: Any, severity_scale: float, *, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * float(severity_scale))))


def _scale_corruption_config(corruption_cfg: dict[str, Any], severity_scale: float) -> dict[str, Any]:
    severity_scale = max(float(severity_scale), 0.0)
    scaled = {key: value for key, value in corruption_cfg.items()}
    if "point_dropout" in scaled:
        cfg = dict(scaled["point_dropout"])
        cfg["dropout_ratio"] = min(max(float(cfg.get("dropout_ratio", 0.15)) * severity_scale, 0.0), 0.95)
        scaled["point_dropout"] = cfg
    if "local_hole_mask" in scaled:
        cfg = dict(scaled["local_hole_mask"])
        cfg["hole_radius"] = float(cfg.get("hole_radius", 0.25)) * severity_scale
        cfg["max_holes"] = _scaled_count(cfg.get("max_holes", 2), severity_scale)
        scaled["local_hole_mask"] = cfg
    if "gaussian_jitter" in scaled:
        cfg = dict(scaled["gaussian_jitter"])
        cfg["sigma"] = float(cfg.get("sigma", 0.03)) * severity_scale
        scaled["gaussian_jitter"] = cfg
    if "normal_noise" in scaled:
        cfg = dict(scaled["normal_noise"])
        cfg["sigma"] = float(cfg.get("sigma", 0.1)) * severity_scale
        cfg["flip_prob"] = min(max(float(cfg.get("flip_prob", 0.05)) * severity_scale, 0.0), 1.0)
        scaled["normal_noise"] = cfg
    if "density_imbalance" in scaled:
        cfg = dict(scaled["density_imbalance"])
        cfg["region_radius"] = float(cfg.get("region_radius", 0.3)) * severity_scale
        cfg["thin_ratio"] = min(max(float(cfg.get("thin_ratio", 0.5)) * severity_scale, 0.0), 0.95)
        cfg["duplicate_count"] = _scaled_count(cfg.get("duplicate_count", 64), severity_scale)
        scaled["density_imbalance"] = cfg
    if "outlier_cluster" in scaled:
        cfg = dict(scaled["outlier_cluster"])
        cfg["cluster_size"] = _scaled_count(cfg.get("cluster_size", 32), severity_scale)
        cfg["cluster_offset_sigma"] = float(cfg.get("cluster_offset_sigma", 0.5)) * severity_scale
        cfg["cluster_spread_sigma"] = float(cfg.get("cluster_spread_sigma", 0.03)) * severity_scale
        scaled["outlier_cluster"] = cfg
    return scaled


def apply_patch_corruptions(
    *,
    clean_points: np.ndarray,
    clean_normals: np.ndarray,
    observed_points: np.ndarray,
    config: dict[str, Any],
    seed: int,
    sample_key: str,
    severity_scale: float = 1.0,
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
        "severity_scale": float(severity_scale),
    }

    corruption_cfg = _scale_corruption_config(config.get("corruptions", config), severity_scale)
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

    # --- LiDAR-realistic corruption pipeline (CarNet_v0 / Phase 3) -----------
    # All four types operate in the patch's canonical frame (origin-centred,
    # unit-radius after cache normalisation) and share a single simulated
    # scanner pose ``p_s`` sampled on a hemisphere of radius ``scanner_radius``.
    # Each type is independently togglable via its own config block; absent or
    # ``enabled: false`` entries are no-ops so the classical synthetic pipeline
    # above continues to work unchanged.
    lidar_cfg = corruption_cfg.get("lidar", {}) or {}
    lidar_any_enabled = any(
        (lidar_cfg.get(name, {}) or {}).get("enabled", False)
        for name in (
            "beam_occlusion",
            "incidence_angle_dropout",
            "range_dependent_noise",
            "azimuthal_ring_sparsity",
        )
    )
    scanner_pos = None
    if lidar_any_enabled and len(points) > 0:
        scanner_cfg = lidar_cfg.get("scanner", {}) or {}
        if "fixed_position" in scanner_cfg:
            scanner_pos = np.asarray(scanner_cfg["fixed_position"], dtype=np.float32).reshape(3)
        else:
            scanner_radius = float(scanner_cfg.get("radius", 3.0))
            min_elevation_deg = float(scanner_cfg.get("min_elevation_deg", 5.0))
            max_elevation_deg = float(scanner_cfg.get("max_elevation_deg", 45.0))
            azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
            elevation = float(
                rng.uniform(
                    np.deg2rad(min_elevation_deg),
                    np.deg2rad(max_elevation_deg),
                )
            )
            scanner_pos = np.asarray(
                [
                    scanner_radius * np.cos(elevation) * np.cos(azimuth),
                    scanner_radius * np.cos(elevation) * np.sin(azimuth),
                    scanner_radius * np.sin(elevation),
                ],
                dtype=np.float32,
            )
        metadata["lidar_scanner_position"] = scanner_pos.tolist()

    if scanner_pos is not None and len(points) > 0:
        # Precompute view direction (scanner -> point) and incidence cosine.
        view_vecs = points - scanner_pos[None, :]
        view_ranges = np.linalg.norm(view_vecs, axis=1).astype(np.float32)
        safe_ranges = np.clip(view_ranges, 1e-6, None)
        view_unit = (view_vecs / safe_ranges[:, None]).astype(np.float32)
        # Incidence cosine: how much the surface normal opposes the view ray.
        # Positive = surface facing the scanner.
        incidence_cos = np.clip(
            np.sum(normals * (-view_unit), axis=1),
            -1.0,
            1.0,
        ).astype(np.float32)

        beam_cfg = lidar_cfg.get("beam_occlusion", {}) or {}
        if beam_cfg.get("enabled", False):
            min_cos = float(beam_cfg.get("min_cosine", 0.05))
            keep_mask = incidence_cos >= min_cos
            if np.any(keep_mask):
                points = points[keep_mask]
                normals = normals[keep_mask]
                flags = flags[keep_mask]
                view_ranges = view_ranges[keep_mask]
                incidence_cos = incidence_cos[keep_mask]
            else:
                # Degenerate view — fall back to keeping all points so later
                # stages still have data. Record the event in metadata.
                metadata.setdefault("lidar_beam_occlusion_degenerate", True)
            severity_terms.append(1.0 - float(np.mean(keep_mask)) if len(keep_mask) else 0.0)
            metadata["enabled_corruptions"].append("beam_occlusion")
            metadata["lidar_beam_occlusion_min_cosine"] = min_cos

        angle_cfg = lidar_cfg.get("incidence_angle_dropout", {}) or {}
        if angle_cfg.get("enabled", False) and len(points) > 0:
            power = float(angle_cfg.get("power", 3.0))
            min_keep = float(angle_cfg.get("min_keep_probability", 0.05))
            keep_prob = np.clip(np.abs(incidence_cos), 0.0, 1.0) ** power
            keep_prob = np.maximum(keep_prob, min_keep).astype(np.float32)
            keep_mask = rng.random(len(points)) < keep_prob
            if np.any(keep_mask):
                points = points[keep_mask]
                normals = normals[keep_mask]
                flags = flags[keep_mask]
                view_ranges = view_ranges[keep_mask]
                incidence_cos = incidence_cos[keep_mask]
            severity_terms.append(1.0 - float(np.mean(keep_prob)))
            metadata["enabled_corruptions"].append("incidence_angle_dropout")
            metadata["lidar_incidence_angle_power"] = power

        range_cfg = lidar_cfg.get("range_dependent_noise", {}) or {}
        if range_cfg.get("enabled", False) and len(points) > 0:
            base_sigma = float(range_cfg.get("base_sigma", 0.005))
            range_exponent = float(range_cfg.get("range_exponent", 2.0))
            reference_range = float(range_cfg.get("reference_range", 2.0))
            per_point_sigma = base_sigma * np.power(
                view_ranges / max(reference_range, 1e-6), range_exponent
            )
            per_point_sigma = per_point_sigma.astype(np.float32)[:, None]
            noise = rng.normal(0.0, 1.0, size=points.shape).astype(np.float32) * per_point_sigma
            points = points + noise
            flags = np.maximum(flags, np.linalg.norm(noise, axis=1).astype(np.float32))
            severity_terms.append(base_sigma)
            metadata["enabled_corruptions"].append("range_dependent_noise")
            metadata["lidar_range_base_sigma"] = base_sigma

        ring_cfg = lidar_cfg.get("azimuthal_ring_sparsity", {}) or {}
        if ring_cfg.get("enabled", False) and len(points) > 0:
            num_bands = int(ring_cfg.get("num_bands", 6))
            band_width_deg = float(ring_cfg.get("band_width_deg", 4.0))
            # Azimuth around the scanner's up-axis (world Z for simplicity).
            rel = points - scanner_pos[None, :]
            phi = np.arctan2(rel[:, 1], rel[:, 0]).astype(np.float32)
            # Normalise to [0, 2π)
            phi_mod = np.mod(phi, 2.0 * np.pi)
            band_width_rad = np.deg2rad(band_width_deg)
            band_centers = np.linspace(0.0, 2.0 * np.pi, num_bands, endpoint=False, dtype=np.float32)
            drop_mask = np.zeros((len(points),), dtype=bool)
            for centre in band_centers:
                diff = np.abs(phi_mod - centre)
                diff = np.minimum(diff, 2.0 * np.pi - diff)
                drop_mask |= diff < band_width_rad * 0.5
            keep_mask = ~drop_mask
            if np.any(keep_mask):
                points = points[keep_mask]
                normals = normals[keep_mask]
                flags = flags[keep_mask]
            severity_terms.append(num_bands * band_width_deg / 360.0)
            metadata["enabled_corruptions"].append("azimuthal_ring_sparsity")
            metadata["lidar_azimuthal_bands"] = num_bands

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
