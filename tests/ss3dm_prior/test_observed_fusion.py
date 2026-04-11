from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ss3dm_prior.data.lidar_io import load_lidar_frame
from ss3dm_prior.data.observed_fusion import build_sequence_observed_cache, generate_tile_centers
from ss3dm_prior.data.raw_sequence import RawSequence


def _write_scenario(sequence_root: Path, num_frames: int) -> None:
    sequence_root.mkdir(parents=True, exist_ok=True)
    c2w = np.stack([np.eye(4, dtype=np.float32) for _ in range(num_frames)], axis=0)
    payload = {
        "scene_id": "toy_scene",
        "metas": {"num_frames": num_frames},
        "objects": {},
        "observers": {
            "camera_FRONT": {
                "id": "camera_FRONT",
                "class_name": "Camera",
                "num_frames": num_frames,
                "data": {
                    "hw": np.asarray([[10.0, 20.0] for _ in range(num_frames)], dtype=np.float32),
                    "intr": np.stack([np.eye(3, dtype=np.float32) for _ in range(num_frames)], axis=0),
                    "c2w": c2w,
                },
            },
            "lidar_TOP": {
                "id": "lidar_TOP",
                "class_name": "RaysLidar",
                "num_frames": num_frames,
                "data": {
                    "sensor_v2w": np.stack(
                        [np.eye(4, dtype=np.float32) for _ in range(num_frames)], axis=0
                    )
                },
            },
        },
    }
    with (sequence_root / "scenario.pt").open("wb") as handle:
        pickle.dump(payload, handle)
    (sequence_root / "scenario.txt").write_text("{'scene_id': 'toy_scene', 'metas': {'num_frames': 2}}")


def _write_lidar_npz(path: Path, rays_o: np.ndarray, rays_d: np.ndarray, ranges: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rays_o=rays_o, rays_d=rays_d, ranges=ranges)


def test_load_lidar_frame_filters_invalid_points(tmp_path: Path) -> None:
    npz_path = tmp_path / "lidar_frame.npz"
    _write_lidar_npz(
        npz_path,
        rays_o=np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
        rays_d=np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        ranges=np.asarray([1.0, -1.0, np.inf], dtype=np.float32),
    )

    frame = load_lidar_frame(
        npz_path,
        min_range=0.5,
        max_range=10.0,
        max_points_per_frame=None,
        seed=0,
        lidar_rays_world_frame=True,
    )

    assert frame.points.shape == (1, 3)
    assert np.allclose(frame.points[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_build_sequence_observed_cache_and_tile_centers(tmp_path: Path) -> None:
    sequence_root = tmp_path / "DATA" / "Town01" / "2_streetsurf"
    _write_scenario(sequence_root, num_frames=2)

    points_by_frame = [
        np.asarray([[0.2, 0.2, 0.0], [0.8, 0.3, 0.0], [2.1, 2.2, 0.0]], dtype=np.float32),
        np.asarray([[0.1, 0.6, 0.0], [0.7, 0.7, 0.0], [2.4, 2.5, 0.0]], dtype=np.float32),
    ]
    for frame_idx, points in enumerate(points_by_frame):
        rays_o = np.zeros_like(points)
        rays_d = np.asarray(points, dtype=np.float32)
        ranges = np.ones((len(points),), dtype=np.float32)
        _write_lidar_npz(
            sequence_root / "lidars" / "lidar_TOP" / f"{frame_idx:08d}.npz",
            rays_o,
            rays_d,
            ranges,
        )

    manifest_entry = {
        "town_id": "Town01",
        "sequence_id": "Town01__2_streetsurf",
        "sequence_root": str(sequence_root),
        "num_frames_from_name": 2,
        "camera_names": ["camera_FRONT"],
        "lidar_names": ["lidar_TOP"],
        "lidar_dir_map": {"lidar_TOP": str(sequence_root / "lidars" / "lidar_TOP")},
    }
    sequence = RawSequence.from_manifest_entry(manifest_entry)
    cache = build_sequence_observed_cache(
        sequence,
        {
            "frame_stride": 1,
            "lidar_names": ["lidar_TOP"],
            "min_range": 0.5,
            "max_range": 10.0,
            "max_points_per_frame": None,
            "tile_stride_m": 1.0,
            "tile_min_points": 2,
            "tile_center_mode": "mean",
            "max_observed_points_per_sequence": None,
            "lidar_rays_world_frame": True,
            "seed": 0,
        },
    )

    assert cache.observed_points.shape[1] == 3
    assert len(cache.observed_points) == 6
    assert len(cache.tile_centers) == 2
    assert len(cache.camera_centers) == 2
    assert cache.sequence_stats["tile_centers_count"] == 2

    tile_centers, occupied = generate_tile_centers(
        cache.observed_points,
        tile_stride_m=1.0,
        tile_min_points=2,
        tile_center_mode="cell_center",
    )
    assert tile_centers.shape == (2, 3)
    assert occupied.shape == (2, 3)
