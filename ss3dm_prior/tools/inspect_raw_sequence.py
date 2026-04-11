"""Inspect scenario metadata and LiDAR sanity stats for one SS3DM sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ss3dm_prior.data.lidar_io import load_lidar_frame
from ss3dm_prior.data.scenario_loader import load_scenario


DEFAULT_LIDAR_NAMES = [
    "lidar_FRONT",
    "lidar_LEFT",
    "lidar_REAR",
    "lidar_RIGHT",
    "lidar_TOP",
]


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one raw SS3DM sequence and print scenario/LiDAR sanity stats."
    )
    parser.add_argument("--sequence_root", required=True, help="Path to a raw SS3DM sequence root.")
    parser.add_argument("--frame_stride", type=int, default=50, help="Stride for LiDAR inspection.")
    parser.add_argument(
        "--lidar_names",
        nargs="+",
        default=DEFAULT_LIDAR_NAMES,
        help="LiDAR sensor names to inspect.",
    )
    parser.add_argument("--min_range", type=float, default=0.5, help="Minimum valid LiDAR range.")
    parser.add_argument("--max_range", type=float, default=80.0, help="Maximum valid LiDAR range.")
    parser.add_argument(
        "--max_points_per_frame",
        type=int,
        default=10000,
        help="Maximum sampled points per LiDAR frame.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--lidar_rays_world_frame",
        action="store_true",
        default=True,
        help="Assume rays are already expressed in world frame.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    sequence_root = Path(args.sequence_root).expanduser().resolve()
    scenario = load_scenario(sequence_root)

    print(f"sequence_root: {sequence_root}")
    print(f"scene_id: {scenario.scene_id}")
    print(f"num_frames: {scenario.num_frames}")
    print(f"source_format: {scenario.source_format}")
    print(f"camera_names: {sorted(scenario.cameras.keys())}")
    print(f"lidar_names: {sorted(scenario.lidars.keys())}")
    if scenario.warnings:
        print("scenario_warnings:")
        for warning in scenario.warnings:
            print(f"  - {warning}")
    else:
        print("scenario_warnings: none")

    num_frames = scenario.num_frames if scenario.num_frames is not None else 0
    frame_indices = list(range(0, num_frames, args.frame_stride)) if num_frames else [0]
    if frame_indices and num_frames and frame_indices[-1] != num_frames - 1:
        frame_indices.append(num_frames - 1)

    all_points: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []
    bbox_centers: list[np.ndarray] = []
    per_frame_counts: list[int] = []

    for frame_idx in frame_indices:
        frame_points: list[np.ndarray] = []
        frame_ranges: list[np.ndarray] = []
        for lidar_name in args.lidar_names:
            npz_path = sequence_root / "lidars" / lidar_name / f"{frame_idx:08d}.npz"
            if not npz_path.exists():
                continue
            lidar_frame = load_lidar_frame(
                npz_path,
                min_range=args.min_range,
                max_range=args.max_range,
                max_points_per_frame=args.max_points_per_frame,
                seed=args.seed,
                lidar_rays_world_frame=args.lidar_rays_world_frame,
            )
            if len(lidar_frame.points) == 0:
                continue
            frame_points.append(lidar_frame.points)
            frame_ranges.append(lidar_frame.ranges)

        if not frame_points:
            per_frame_counts.append(0)
            continue

        merged_points = np.concatenate(frame_points, axis=0)
        merged_ranges = np.concatenate(frame_ranges, axis=0)
        all_points.append(merged_points)
        all_ranges.append(merged_ranges)
        per_frame_counts.append(int(len(merged_points)))
        bbox_centers.append((merged_points.min(axis=0) + merged_points.max(axis=0)) / 2.0)

    if all_points:
        observed_points = np.concatenate(all_points, axis=0)
        observed_ranges = np.concatenate(all_ranges, axis=0)
        bbox_min = observed_points.min(axis=0)
        bbox_max = observed_points.max(axis=0)
        if len(bbox_centers) >= 2:
            drift = np.linalg.norm(np.diff(np.stack(bbox_centers), axis=0), axis=1)
            drift_mean = float(np.mean(drift))
            drift_max = float(np.max(drift))
        else:
            drift_mean = 0.0
            drift_max = 0.0
        print(f"sampled_frames: {len(frame_indices)}")
        print(f"frames_with_points: {len(all_points)}")
        print(f"point_count_stats: min={min(per_frame_counts)} max={max(per_frame_counts)} mean={float(np.mean(per_frame_counts)):.2f}")
        print(
            "range_stats: "
            f"min={float(np.min(observed_ranges)):.4f} "
            f"max={float(np.max(observed_ranges)):.4f} "
            f"mean={float(np.mean(observed_ranges)):.4f}"
        )
        print(f"points_bbox_min: {bbox_min.tolist()}")
        print(f"points_bbox_max: {bbox_max.tolist()}")
        print(f"frame_to_frame_bbox_drift_mean: {drift_mean:.4f}")
        print(f"frame_to_frame_bbox_drift_max: {drift_max:.4f}")
    else:
        print("sampled_frames: 0")
        print("frames_with_points: 0")
        print("no LiDAR points were loaded")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
