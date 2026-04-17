"""Visibility/free-space query generation for teacher patch cache v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import zlib

import numpy as np
import trimesh

from ss3dm_prior.data.lidar_io import load_lidar_frame


def _query_rng(seed: int, patch_id: str) -> np.random.Generator:
    derived_seed = (int(seed) + zlib.crc32(f"visibility::{patch_id}".encode("utf-8"))) % (2**32)
    return np.random.default_rng(derived_seed)


def _normalize_points(points_world: np.ndarray, patch_center_world: np.ndarray, patch_radius_m: float) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return ((points_world - patch_center_world[None, :]) / float(patch_radius_m)).astype(np.float32)


def _min_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if len(points_a) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(points_b) == 0:
        return np.full((len(points_a),), np.inf, dtype=np.float32)
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.linalg.norm(diff, axis=-1).min(axis=1).astype(np.float32)


def _sample_unit_ball(rng: np.random.Generator, count: int) -> np.ndarray:
    samples: list[np.ndarray] = []
    remaining = int(count)
    while remaining > 0:
        candidate_count = max(remaining * 2, 32)
        candidates = rng.uniform(-1.0, 1.0, size=(candidate_count, 3)).astype(np.float32)
        keep = np.linalg.norm(candidates, axis=1) <= 1.0
        accepted = candidates[keep]
        if len(accepted) == 0:
            continue
        samples.append(accepted[:remaining])
        remaining -= min(len(accepted), remaining)
    return np.concatenate(samples, axis=0).astype(np.float32)


def _resample_points(rng: np.random.Generator, points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0 or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    replace = len(points) < int(target_count)
    indices = rng.choice(len(points), size=int(target_count), replace=replace)
    return np.asarray(points[indices], dtype=np.float32)


def _ray_patch_intervals(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    ranges: np.ndarray,
    *,
    patch_center_world: np.ndarray,
    patch_radius_m: float,
    free_space_min_margin_m: float,
    free_space_max_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = patch_center_world.reshape(1, 3).astype(np.float32)
    oc = rays_o.astype(np.float32) - center
    d = rays_d.astype(np.float32)
    a = np.sum(d * d, axis=1)
    b = 2.0 * np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - float(patch_radius_m) ** 2
    discriminant = b * b - 4.0 * a * c
    valid = discriminant >= 0.0
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    sqrt_disc = np.sqrt(np.clip(discriminant[valid], a_min=0.0, a_max=None)).astype(np.float32)
    a_valid = np.where(a[valid] > 1e-8, a[valid], 1e-8).astype(np.float32)
    t_near = (-b[valid] - sqrt_disc) / (2.0 * a_valid)
    t_far = (-b[valid] + sqrt_disc) / (2.0 * a_valid)
    t_start = np.maximum(t_near, 0.0)
    t_end = np.minimum(t_far, np.minimum(ranges[valid] * float(free_space_max_ratio), ranges[valid] - float(free_space_min_margin_m)))
    interval_valid = t_end > t_start
    valid_indices = np.nonzero(valid)[0][interval_valid].astype(np.int64)
    return (
        valid_indices,
        t_start[interval_valid].astype(np.float32),
        t_end[interval_valid].astype(np.float32),
    )


def _sample_surface_queries(
    *,
    local_mesh: trimesh.Trimesh,
    patch_center_world: np.ndarray,
    patch_radius_m: float,
    surface_query_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if surface_query_count <= 0 or len(local_mesh.faces) == 0 or float(local_mesh.area) <= 0.0:
        return np.zeros((0, 3), dtype=np.float32)
    surface_points_world, _ = trimesh.sample.sample_surface(local_mesh, surface_query_count, seed=rng)
    return _normalize_points(
        np.asarray(surface_points_world, dtype=np.float32),
        patch_center_world,
        patch_radius_m,
    )


def _sample_free_queries_from_lidar(
    *,
    raw_sequence: Any,
    patch_center_world: np.ndarray,
    patch_radius_m: float,
    query_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    frame_stride = int(query_config.get("frame_stride", 10))
    lidar_names = list(query_config.get("lidar_names") or raw_sequence.lidar_names())
    min_range = float(query_config.get("min_range", 0.5))
    max_range = float(query_config.get("max_range", 120.0))
    max_points_per_frame = query_config.get("max_points_per_frame")
    max_points_per_frame = None if max_points_per_frame is None else int(max_points_per_frame)
    free_space_min_margin_m = float(query_config.get("free_space_min_margin_m", 0.15))
    free_space_max_ratio = float(query_config.get("free_space_max_ratio", 0.95))
    max_free_rays_per_patch = int(query_config.get("max_free_rays_per_patch", 2048))

    frame_indices = raw_sequence.iter_frame_indices(frame_stride)
    candidate_points_world: list[np.ndarray] = []
    lidar_support_count = 0

    for frame_idx in frame_indices:
        for lidar_name in lidar_names:
            lidar_path = raw_sequence.lidar_frame_path(lidar_name, frame_idx)
            if not lidar_path.exists():
                continue
            lidar_frame = load_lidar_frame(
                lidar_path,
                min_range=min_range,
                max_range=max_range,
                max_points_per_frame=max_points_per_frame,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            if len(lidar_frame.ranges) == 0:
                continue

            valid_indices, t_start, t_end = _ray_patch_intervals(
                lidar_frame.rays_o,
                lidar_frame.rays_d,
                lidar_frame.ranges,
                patch_center_world=patch_center_world,
                patch_radius_m=patch_radius_m,
                free_space_min_margin_m=free_space_min_margin_m,
                free_space_max_ratio=free_space_max_ratio,
            )
            if len(t_start) == 0:
                continue
            lidar_support_count += 1
            sample_t = rng.uniform(t_start, t_end).astype(np.float32)
            valid_origins = lidar_frame.rays_o[valid_indices].astype(np.float32)
            valid_dirs = lidar_frame.rays_d[valid_indices].astype(np.float32)
            candidate_points_world.append(valid_origins + valid_dirs * sample_t[:, None])

    if not candidate_points_world:
        return np.zeros((0, 3), dtype=np.float32), lidar_support_count

    candidates = np.concatenate(candidate_points_world, axis=0).astype(np.float32)
    if len(candidates) > max_free_rays_per_patch:
        selected = np.sort(rng.choice(len(candidates), size=max_free_rays_per_patch, replace=False))
        candidates = candidates[selected]
    return candidates, lidar_support_count


def _sample_unknown_queries(
    *,
    unknown_query_count: int,
    clean_points_local: np.ndarray,
    observed_points_local: np.ndarray,
    free_query_points_local: np.ndarray,
    query_config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if unknown_query_count <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    surface_exclusion = float(query_config.get("unknown_surface_exclusion_radius_local", 0.08))
    observed_exclusion = float(query_config.get("unknown_observed_exclusion_radius_local", 0.08))
    free_exclusion = float(query_config.get("unknown_free_exclusion_radius_local", 0.08))
    oversample_factor = int(query_config.get("unknown_candidate_oversample_factor", 8))

    accepted: list[np.ndarray] = []
    total_needed = int(unknown_query_count)
    attempts = 0
    max_attempts = 8
    while total_needed > 0 and attempts < max_attempts:
        attempts += 1
        candidates = _sample_unit_ball(rng, max(total_needed * oversample_factor, 32))
        keep_mask = np.ones((len(candidates),), dtype=bool)
        keep_mask &= _min_distances(candidates, clean_points_local) > surface_exclusion
        keep_mask &= _min_distances(candidates, observed_points_local) > observed_exclusion
        keep_mask &= _min_distances(candidates, free_query_points_local) > free_exclusion
        kept = candidates[keep_mask]
        if len(kept) == 0:
            continue
        accepted.append(kept[:total_needed])
        total_needed -= min(len(kept), total_needed)

    if total_needed > 0:
        accepted.append(_sample_unit_ball(rng, total_needed))
    return np.concatenate(accepted, axis=0).astype(np.float32)


def _estimate_camera_support(
    *,
    raw_sequence: Any,
    patch_center_world: np.ndarray,
    query_config: dict[str, Any],
) -> int:
    frame_stride = int(query_config.get("frame_stride", 10))
    support_radius_m = float(query_config.get("camera_support_radius_m", 25.0))
    frame_indices = raw_sequence.iter_frame_indices(frame_stride)
    camera_support_count = 0
    for camera_name in raw_sequence.camera_names():
        observer = raw_sequence.scenario.cameras.get(camera_name)
        if observer is None or observer.c2w is None:
            continue
        valid_indices = [idx for idx in frame_indices if idx < observer.c2w.shape[0]]
        if not valid_indices:
            continue
        centers = np.asarray(observer.c2w[valid_indices, :3, 3], dtype=np.float32)
        distances = np.linalg.norm(centers - patch_center_world.reshape(1, 3), axis=1)
        camera_support_count += int(np.sum(distances <= support_radius_m))
    return camera_support_count


def _assemble_query_arrays(
    surface_query_points: np.ndarray,
    free_query_points: np.ndarray,
    unknown_query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_points_all = np.concatenate(
        [surface_query_points, free_query_points, unknown_query_points],
        axis=0,
    ).astype(np.float32)
    query_labels_all = np.concatenate(
        [
            np.ones((len(surface_query_points),), dtype=np.int8),
            np.zeros((len(free_query_points),), dtype=np.int8),
            np.zeros((len(unknown_query_points),), dtype=np.int8),
        ],
        axis=0,
    )
    query_ignore_mask = np.concatenate(
        [
            np.zeros((len(surface_query_points),), dtype=bool),
            np.zeros((len(free_query_points),), dtype=bool),
            np.ones((len(unknown_query_points),), dtype=bool),
        ],
        axis=0,
    )
    return query_points_all, query_labels_all, query_ignore_mask


def _compute_intrinsic_difficulty(
    *,
    observed_points_local: np.ndarray,
    clean_points_local: np.ndarray,
    clean_normals_local: np.ndarray,
    visible_surface_fraction: float,
    camera_support_count: int,
    lidar_support_count: int,
    free_query_points_local: np.ndarray,
    query_config: dict[str, Any],
    planarity_hint: float,
) -> tuple[float, dict[str, float]]:
    observed_to_clean = _min_distances(observed_points_local, clean_points_local)
    observed_to_clean_error = float(np.mean(observed_to_clean)) if len(observed_to_clean) else 1.0
    observed_to_clean_error = float(np.clip(observed_to_clean_error / float(query_config.get("observed_clean_error_scale_local", 0.15)), 0.0, 1.0))

    mean_normal = np.mean(clean_normals_local.astype(np.float32), axis=0, keepdims=True) if len(clean_normals_local) else np.zeros((1, 3), dtype=np.float32)
    normal_norm = float(np.linalg.norm(mean_normal))
    normal_dispersion = float(np.clip(1.0 - normal_norm, 0.0, 1.0))
    curvature_hint = float(np.clip(1.0 - float(planarity_hint), 0.0, 1.0))
    normal_dispersion_or_curvature = max(normal_dispersion, curvature_hint)

    support_target = float(query_config.get("support_count_target", 8.0))
    support_total = float(camera_support_count + lidar_support_count)
    visible_support_deficit = float(np.clip(1.0 - support_total / max(support_target, 1.0), 0.0, 1.0))

    contradiction_radius = float(query_config.get("contradiction_radius_local", 0.06))
    free_to_clean = _min_distances(free_query_points_local, clean_points_local)
    free_to_observed = _min_distances(free_query_points_local, observed_points_local)
    free_space_contradiction_ratio = float(
        np.mean(np.minimum(free_to_clean, free_to_observed) <= contradiction_radius)
    ) if len(free_query_points_local) else 0.0

    components = {
        "observed_to_clean_nn_error": observed_to_clean_error,
        "visible_surface_coverage_error": float(np.clip(1.0 - visible_surface_fraction, 0.0, 1.0)),
        "normal_dispersion_or_curvature": normal_dispersion_or_curvature,
        "visible_support_deficit": visible_support_deficit,
        "free_space_contradiction_ratio": free_space_contradiction_ratio,
    }
    weights = {
        "observed_to_clean_nn_error": float(query_config.get("difficulty_weight_observed_clean", 0.30)),
        "visible_surface_coverage_error": float(query_config.get("difficulty_weight_visible_coverage", 0.25)),
        "normal_dispersion_or_curvature": float(query_config.get("difficulty_weight_normal_dispersion", 0.15)),
        "visible_support_deficit": float(query_config.get("difficulty_weight_support_deficit", 0.15)),
        "free_space_contradiction_ratio": float(query_config.get("difficulty_weight_free_contradiction", 0.15)),
    }
    weight_sum = float(sum(weights.values()))
    difficulty = float(
        sum(weights[name] * components[name] for name in components) / max(weight_sum, 1e-6)
    )
    return float(np.clip(difficulty, 0.0, 1.0)), components


@dataclass
class VisibilityQueryBundle:
    surface_query_points: np.ndarray
    surface_query_labels: np.ndarray
    free_query_points: np.ndarray
    free_query_labels: np.ndarray
    unknown_query_points: np.ndarray
    query_points_all: np.ndarray
    query_labels_all: np.ndarray
    query_ignore_mask: np.ndarray
    camera_support_count: int
    lidar_support_count: int
    visible_surface_fraction: float
    free_space_fraction: float
    unknown_fraction: float
    intrinsic_patch_difficulty_target: float
    difficulty_components_json: dict[str, float]


def build_patch_visibility_queries(
    *,
    raw_sequence: Any,
    local_mesh: trimesh.Trimesh,
    patch_center_world: np.ndarray,
    patch_radius_m: float,
    clean_points_local: np.ndarray,
    clean_normals_local: np.ndarray,
    observed_points_local: np.ndarray,
    planarity_hint: float,
    query_config: dict[str, Any],
    seed: int,
    patch_id: str,
) -> VisibilityQueryBundle:
    rng = _query_rng(seed, patch_id)
    surface_query_points = _sample_surface_queries(
        local_mesh=local_mesh,
        patch_center_world=patch_center_world,
        patch_radius_m=patch_radius_m,
        surface_query_count=int(query_config.get("surface_query_count", 512)),
        rng=rng,
    )
    free_query_points_world, lidar_support_count = _sample_free_queries_from_lidar(
        raw_sequence=raw_sequence,
        patch_center_world=patch_center_world,
        patch_radius_m=patch_radius_m,
        query_config=query_config,
        rng=rng,
    )
    free_query_points = _normalize_points(
        free_query_points_world,
        patch_center_world,
        patch_radius_m,
    )
    free_query_points = _resample_points(
        rng,
        free_query_points,
        int(query_config.get("free_query_count", 512)),
    )
    unknown_query_points = _sample_unknown_queries(
        unknown_query_count=int(query_config.get("unknown_query_count", 256)),
        clean_points_local=clean_points_local,
        observed_points_local=observed_points_local,
        free_query_points_local=free_query_points,
        query_config=query_config,
        rng=rng,
    )

    visible_surface_radius = float(query_config.get("surface_visibility_radius_local", 0.08))
    visible_surface_fraction = float(
        np.mean(_min_distances(surface_query_points, observed_points_local) <= visible_surface_radius)
    ) if len(surface_query_points) else 0.0
    query_points_all, query_labels_all, query_ignore_mask = _assemble_query_arrays(
        surface_query_points,
        free_query_points,
        unknown_query_points,
    )
    total_queries = max(len(query_points_all), 1)
    free_space_fraction = float(len(free_query_points) / total_queries)
    unknown_fraction = float(len(unknown_query_points) / total_queries)
    camera_support_count = _estimate_camera_support(
        raw_sequence=raw_sequence,
        patch_center_world=patch_center_world,
        query_config=query_config,
    )
    intrinsic_patch_difficulty_target, difficulty_components_json = _compute_intrinsic_difficulty(
        observed_points_local=observed_points_local,
        clean_points_local=clean_points_local,
        clean_normals_local=clean_normals_local,
        visible_surface_fraction=visible_surface_fraction,
        camera_support_count=camera_support_count,
        lidar_support_count=lidar_support_count,
        free_query_points_local=free_query_points,
        query_config=query_config,
        planarity_hint=planarity_hint,
    )
    return VisibilityQueryBundle(
        surface_query_points=surface_query_points,
        surface_query_labels=np.ones((len(surface_query_points),), dtype=np.int8),
        free_query_points=free_query_points,
        free_query_labels=np.zeros((len(free_query_points),), dtype=np.int8),
        unknown_query_points=unknown_query_points,
        query_points_all=query_points_all,
        query_labels_all=query_labels_all,
        query_ignore_mask=query_ignore_mask,
        camera_support_count=int(camera_support_count),
        lidar_support_count=int(lidar_support_count),
        visible_surface_fraction=visible_surface_fraction,
        free_space_fraction=free_space_fraction,
        unknown_fraction=unknown_fraction,
        intrinsic_patch_difficulty_target=intrinsic_patch_difficulty_target,
        difficulty_components_json=difficulty_components_json,
    )
