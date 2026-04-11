"""Check observed cache files produced by the SS3DM prior pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ss3dm_prior.utils.io import load_json


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate saved observed point caches and summarize sequence counts."
    )
    parser.add_argument("--cache_dir", required=True, help="Root cache directory to inspect.")
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    npz_paths = sorted(cache_root.glob("*/*/observed_cache.npz"))

    if not npz_paths:
        print(f"no observed_cache.npz files found under {cache_root}")
        return 1

    town_counts: Counter[str] = Counter()
    warnings: list[str] = []
    total_points = 0
    total_tiles = 0

    for npz_path in npz_paths:
        sequence_dir = npz_path.parent
        town_id = sequence_dir.parent.name
        town_counts[town_id] += 1
        stats_path = sequence_dir / "sequence_stats.json"
        if not stats_path.exists():
            warnings.append(f"missing stats json for {sequence_dir}")
            continue

        with np.load(npz_path) as payload:
            observed_points = payload["observed_points"]
            tile_centers = payload["tile_centers"]
            camera_centers = payload["camera_centers"]
            sequence_stats_json = payload["sequence_stats_json"].item()

        stats = load_json(stats_path)
        if int(stats.get("observed_points_count", -1)) != int(len(observed_points)):
            warnings.append(f"observed point count mismatch for {sequence_dir}")
        if int(stats.get("tile_centers_count", -1)) != int(len(tile_centers)):
            warnings.append(f"tile center count mismatch for {sequence_dir}")
        if int(stats.get("camera_centers_count", -1)) != int(len(camera_centers)):
            warnings.append(f"camera center count mismatch for {sequence_dir}")
        if not sequence_stats_json:
            warnings.append(f"empty sequence_stats_json payload for {sequence_dir}")

        total_points += int(len(observed_points))
        total_tiles += int(len(tile_centers))

    print(f"num_sequences: {len(npz_paths)}")
    for town_id in sorted(town_counts):
        print(f"  - {town_id}: {town_counts[town_id]} sequences")
    print(f"total_observed_points: {total_points}")
    print(f"total_tile_centers: {total_tiles}")

    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
