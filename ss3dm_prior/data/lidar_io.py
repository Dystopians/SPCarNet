"""Utilities for reading and filtering SS3DM raw LiDAR frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np


@dataclass
class LidarFrame:
    rays_o: np.ndarray
    rays_d: np.ndarray
    ranges: np.ndarray
    points: np.ndarray
    stats: dict[str, float | int]


def _make_rng(seed: int, sample_key: str) -> np.random.Generator:
    derived_seed = (int(seed) + zlib.crc32(sample_key.encode("utf-8"))) % (2**32)
    return np.random.default_rng(derived_seed)


def compute_points_from_lidar(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    ranges: np.ndarray,
    *,
    lidar_rays_world_frame: bool = True,
) -> np.ndarray:
    del lidar_rays_world_frame
    return np.asarray(rays_o + rays_d * ranges[:, None], dtype=np.float32)


def load_lidar_frame(
    npz_path: str | Path,
    *,
    min_range: float,
    max_range: float,
    max_points_per_frame: int | None,
    seed: int,
    lidar_rays_world_frame: bool = True,
) -> LidarFrame:
    path = Path(npz_path)
    with np.load(path) as payload:
        rays_o = np.asarray(payload["rays_o"], dtype=np.float32)
        rays_d = np.asarray(payload["rays_d"], dtype=np.float32)
        ranges = np.asarray(payload["ranges"], dtype=np.float32)

    valid_mask = (
        np.isfinite(rays_o).all(axis=1)
        & np.isfinite(rays_d).all(axis=1)
        & np.isfinite(ranges)
        & (ranges > float(min_range))
        & (ranges < float(max_range))
    )

    rays_o = rays_o[valid_mask]
    rays_d = rays_d[valid_mask]
    ranges = ranges[valid_mask]

    if max_points_per_frame is not None and len(ranges) > int(max_points_per_frame):
        rng = _make_rng(seed, str(path))
        selected = np.sort(rng.choice(len(ranges), size=int(max_points_per_frame), replace=False))
        rays_o = rays_o[selected]
        rays_d = rays_d[selected]
        ranges = ranges[selected]

    points = compute_points_from_lidar(
        rays_o,
        rays_d,
        ranges,
        lidar_rays_world_frame=lidar_rays_world_frame,
    )
    stats = {
        "point_count": int(len(points)),
        "range_min": float(np.min(ranges)) if len(ranges) else 0.0,
        "range_max": float(np.max(ranges)) if len(ranges) else 0.0,
        "range_mean": float(np.mean(ranges)) if len(ranges) else 0.0,
    }
    return LidarFrame(
        rays_o=rays_o,
        rays_d=rays_d,
        ranges=ranges,
        points=points,
        stats=stats,
    )
