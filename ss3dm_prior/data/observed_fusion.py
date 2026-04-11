"""Sequence-level LiDAR fusion and occupancy-aware tile center generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from ss3dm_prior.data.lidar_io import load_lidar_frame
from ss3dm_prior.data.raw_sequence import RawSequence
from ss3dm_prior.utils.io import dump_json


def _rng_from_sequence(seed: int, sequence_id: str) -> np.random.Generator:
    derived_seed = (int(seed) + zlib.crc32(sequence_id.encode("utf-8"))) % (2**32)
    return np.random.default_rng(derived_seed)


def _bbox_stats(points: np.ndarray) -> tuple[list[float], list[float]]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return points.min(axis=0).astype(float).tolist(), points.max(axis=0).astype(float).tolist()


def collect_camera_centers(sequence: RawSequence, frame_indices: list[int]) -> np.ndarray:
    centers: list[np.ndarray] = []
    for camera_name in sequence.camera_names():
        observer = sequence.scenario.cameras.get(camera_name)
        if observer is None or observer.c2w is None:
            continue
        valid_indices = [idx for idx in frame_indices if idx < observer.c2w.shape[0]]
        if not valid_indices:
            continue
        centers.append(np.asarray(observer.c2w[valid_indices, :3, 3], dtype=np.float32))
    if not centers:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(centers, axis=0)


def generate_tile_centers(
    points: np.ndarray,
    *,
    tile_stride_m: float,
    tile_min_points: int,
    tile_center_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    grid_indices = np.floor(points / float(tile_stride_m)).astype(np.int64)
    unique_cells, inverse_indices, counts = np.unique(
        grid_indices,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    keep_mask = counts >= int(tile_min_points)

    if tile_center_mode == "cell_center":
        centers = (unique_cells[keep_mask].astype(np.float32) + 0.5) * float(tile_stride_m)
    else:
        sums = np.zeros((len(unique_cells), 3), dtype=np.float64)
        np.add.at(sums, inverse_indices, points.astype(np.float64))
        centers = (sums[keep_mask] / counts[keep_mask, None]).astype(np.float32)

    return centers.astype(np.float32), unique_cells[keep_mask]


@dataclass
class ObservedSequenceCache:
    observed_points: np.ndarray
    tile_centers: np.ndarray
    camera_centers: np.ndarray
    sequence_stats: dict[str, Any]


def build_sequence_observed_cache(
    sequence: RawSequence,
    config: dict[str, Any],
) -> ObservedSequenceCache:
    frame_stride = int(config["frame_stride"])
    lidar_names = list(config["lidar_names"])
    min_range = float(config["min_range"])
    max_range = float(config["max_range"])
    max_points_per_frame = config.get("max_points_per_frame")
    max_points_per_frame = None if max_points_per_frame is None else int(max_points_per_frame)
    max_observed_points_per_sequence = config.get("max_observed_points_per_sequence")
    if max_observed_points_per_sequence is not None:
        max_observed_points_per_sequence = int(max_observed_points_per_sequence)
    tile_stride_m = float(config["tile_stride_m"])
    tile_min_points = int(config["tile_min_points"])
    tile_center_mode = str(config.get("tile_center_mode", "mean"))
    lidar_rays_world_frame = bool(config.get("lidar_rays_world_frame", True))
    seed = int(config.get("seed", 0))

    frame_indices = sequence.iter_frame_indices(frame_stride)
    all_points: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []
    point_count_per_frame: list[int] = []
    bbox_centers_per_frame: list[np.ndarray] = []
    frames_with_points = 0

    for frame_idx in frame_indices:
        frame_points: list[np.ndarray] = []
        frame_ranges: list[np.ndarray] = []
        for lidar_name in lidar_names:
            npz_path = sequence.lidar_frame_path(lidar_name, frame_idx)
            if not npz_path.exists():
                continue
            lidar_frame = load_lidar_frame(
                npz_path,
                min_range=min_range,
                max_range=max_range,
                max_points_per_frame=max_points_per_frame,
                seed=seed,
                lidar_rays_world_frame=lidar_rays_world_frame,
            )
            if len(lidar_frame.points) == 0:
                continue
            frame_points.append(lidar_frame.points)
            frame_ranges.append(lidar_frame.ranges)

        if not frame_points:
            point_count_per_frame.append(0)
            continue

        merged_points = np.concatenate(frame_points, axis=0).astype(np.float32)
        merged_ranges = np.concatenate(frame_ranges, axis=0).astype(np.float32)
        all_points.append(merged_points)
        all_ranges.append(merged_ranges)
        point_count_per_frame.append(int(len(merged_points)))
        bbox_centers_per_frame.append((merged_points.min(axis=0) + merged_points.max(axis=0)) / 2.0)
        frames_with_points += 1

    if all_points:
        observed_points = np.concatenate(all_points, axis=0).astype(np.float32)
        observed_ranges = np.concatenate(all_ranges, axis=0).astype(np.float32)
    else:
        observed_points = np.zeros((0, 3), dtype=np.float32)
        observed_ranges = np.zeros((0,), dtype=np.float32)

    if (
        max_observed_points_per_sequence is not None
        and len(observed_points) > max_observed_points_per_sequence
    ):
        rng = _rng_from_sequence(seed, sequence.sequence_id)
        selected = np.sort(
            rng.choice(
                len(observed_points),
                size=max_observed_points_per_sequence,
                replace=False,
            )
        )
        observed_points = observed_points[selected]
        observed_ranges = observed_ranges[selected]

    tile_centers, occupied_cells = generate_tile_centers(
        observed_points,
        tile_stride_m=tile_stride_m,
        tile_min_points=tile_min_points,
        tile_center_mode=tile_center_mode,
    )
    camera_centers = collect_camera_centers(sequence, frame_indices)

    if len(bbox_centers_per_frame) >= 2:
        bbox_drift = np.linalg.norm(np.diff(np.stack(bbox_centers_per_frame), axis=0), axis=1)
        bbox_drift_mean = float(np.mean(bbox_drift))
        bbox_drift_max = float(np.max(bbox_drift))
    else:
        bbox_drift_mean = 0.0
        bbox_drift_max = 0.0

    bbox_min, bbox_max = _bbox_stats(observed_points)
    sequence_stats = {
        "town_id": sequence.town_id,
        "sequence_id": sequence.sequence_id,
        "sequence_root": str(sequence.sequence_root),
        "num_frames_total": sequence.num_frames,
        "num_sampled_frames": len(frame_indices),
        "num_frames_with_points": frames_with_points,
        "frame_indices_preview": frame_indices[:10],
        "lidar_names_used": lidar_names,
        "observed_points_count": int(len(observed_points)),
        "tile_centers_count": int(len(tile_centers)),
        "camera_centers_count": int(len(camera_centers)),
        "occupied_grid_cells_count": int(len(occupied_cells)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "range_min": float(np.min(observed_ranges)) if len(observed_ranges) else 0.0,
        "range_max": float(np.max(observed_ranges)) if len(observed_ranges) else 0.0,
        "range_mean": float(np.mean(observed_ranges)) if len(observed_ranges) else 0.0,
        "point_count_per_frame_min": int(min(point_count_per_frame)) if point_count_per_frame else 0,
        "point_count_per_frame_max": int(max(point_count_per_frame)) if point_count_per_frame else 0,
        "point_count_per_frame_mean": float(np.mean(point_count_per_frame)) if point_count_per_frame else 0.0,
        "bbox_drift_mean": bbox_drift_mean,
        "bbox_drift_max": bbox_drift_max,
        "scenario_summary": sequence.scenario.summary_dict(),
        "config": {
            "frame_stride": frame_stride,
            "min_range": min_range,
            "max_range": max_range,
            "max_points_per_frame": max_points_per_frame,
            "tile_stride_m": tile_stride_m,
            "tile_min_points": tile_min_points,
            "max_observed_points_per_sequence": max_observed_points_per_sequence,
            "tile_center_mode": tile_center_mode,
            "lidar_rays_world_frame": lidar_rays_world_frame,
            "seed": seed,
        },
    }

    return ObservedSequenceCache(
        observed_points=observed_points,
        tile_centers=tile_centers,
        camera_centers=camera_centers,
        sequence_stats=sequence_stats,
    )


def save_observed_cache(
    output_dir: str | Path,
    cache: ObservedSequenceCache,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / "observed_cache.npz"
    json_path = output_path / "sequence_stats.json"

    sequence_stats_json = json.dumps(cache.sequence_stats, sort_keys=True)
    np.savez_compressed(
        npz_path,
        observed_points=cache.observed_points,
        tile_centers=cache.tile_centers,
        camera_centers=cache.camera_centers,
        sequence_stats_json=np.asarray(sequence_stats_json),
    )
    dump_json(json_path, cache.sequence_stats, indent=2)
    return npz_path, json_path
