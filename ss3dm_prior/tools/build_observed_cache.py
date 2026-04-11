"""Build sequence-level observed point caches from SS3DM raw LiDAR."""

from __future__ import annotations

import argparse
from pathlib import Path

from ss3dm_prior.data.observed_fusion import build_sequence_observed_cache, save_observed_cache
from ss3dm_prior.data.raw_sequence import RawSequence, select_manifest_entries_by_split
from ss3dm_prior.utils.io import load_json, load_yaml


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build offline observed point caches for selected SS3DM sequences."
    )
    parser.add_argument("--manifest", required=True, help="Path to the manifest JSON file.")
    parser.add_argument("--split_config", required=True, help="Path to the split YAML file.")
    parser.add_argument("--config", required=True, help="Path to the observed cache YAML file.")
    parser.add_argument("--out_dir", required=True, help="Output directory for observed caches.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Split subsets to process.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    manifest = load_json(args.manifest)
    split_config = load_yaml(args.split_config)
    config = load_yaml(args.config)
    observed_cache_config = config.get("observed_cache", config)

    entries = select_manifest_entries_by_split(
        manifest,
        split_config,
        subsets=tuple(args.subsets),
    )
    debug_max_sequences = observed_cache_config.get("debug_max_sequences")
    if debug_max_sequences is not None:
        entries = entries[: int(debug_max_sequences)]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_points = 0
    total_tiles = 0
    for entry in entries:
        sequence = RawSequence.from_manifest_entry(entry)
        cache = build_sequence_observed_cache(sequence, observed_cache_config)
        sequence_out_dir = out_dir / sequence.town_id / sequence.sequence_id
        npz_path, json_path = save_observed_cache(sequence_out_dir, cache)
        total_points += int(len(cache.observed_points))
        total_tiles += int(len(cache.tile_centers))
        print(
            f"{sequence.sequence_id}: "
            f"observed_points={len(cache.observed_points)} "
            f"tile_centers={len(cache.tile_centers)} "
            f"npz={npz_path} stats={json_path}"
        )

    print(f"processed_sequences: {len(entries)}")
    print(f"total_observed_points: {total_points}")
    print(f"total_tile_centers: {total_tiles}")
    print(f"cache_out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
