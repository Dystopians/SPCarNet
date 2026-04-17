"""Build teacher patch cache v2 with visibility/free-space supervision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, load_patch_npz
from ss3dm_prior.data.raw_sequence import RawSequence, select_manifest_entries_by_split
from ss3dm_prior.data.teacher_patch_builder import (
    build_teacher_patches_for_sequence_v2,
    load_sequence_observed_cache,
)
from ss3dm_prior.data.town_mesh_cache import load_town_mesh_cache
from ss3dm_prior.utils.io import load_json, load_yaml

_WORKER_TOWN_MESHES: dict[str, Any] = {}


def _load_patch_record_v2(patch_file: Path) -> PatchIndexRecord:
    payload = load_patch_npz(patch_file)
    difficulty_components_json = payload.get("difficulty_components_json")
    import json

    difficulty_components = (
        json.loads(str(difficulty_components_json.item()))
        if difficulty_components_json is not None
        else {}
    )
    return PatchIndexRecord(
        patch_id=str(payload["patch_id"].item()),
        town_id=str(payload["town_id"].item()),
        sequence_id=str(payload["sequence_id"].item()),
        tile_id=int(payload["tile_id"].item()),
        patch_file=str(patch_file.resolve()),
        num_local_faces=int(payload["num_local_faces"].item()),
        num_observed_points_raw=int(payload["num_observed_points_raw"].item()),
        num_clean_points=int(payload["clean_points"].shape[0]),
        num_observed_points=int(payload["observed_points"].shape[0]),
        teacher_area_local=float(payload["teacher_area_local"].item()),
        planarity_hint=float(json.loads(str(payload["patch_metadata_json"].item())).get("planarity_hint", 0.0)),
        patch_cache_format_version=int(payload["patch_cache_format_version"].item()),
        num_surface_query_points=int(payload["surface_query_points"].shape[0]),
        num_free_query_points=int(payload["free_query_points"].shape[0]),
        num_unknown_query_points=int(payload["unknown_query_points"].shape[0]),
        camera_support_count=int(payload["camera_support_count"].item()),
        lidar_support_count=int(payload["lidar_support_count"].item()),
        visible_surface_fraction=float(payload["visible_surface_fraction"].item()),
        free_space_fraction=float(payload["free_space_fraction"].item()),
        unknown_fraction=float(payload["unknown_fraction"].item()),
        intrinsic_patch_difficulty_target=float(payload["intrinsic_patch_difficulty_target"].item()),
        difficulty_components_json=difficulty_components,
    )


def _collect_completed_patch_records_v2(out_dir: Path) -> list[PatchIndexRecord]:
    records: list[PatchIndexRecord] = []
    for stats_path in sorted(out_dir.rglob("sequence_patch_stats.json")):
        sequence_dir = stats_path.parent
        for patch_file in sorted(sequence_dir.glob("*.npz")):
            records.append(_load_patch_record_v2(patch_file))
    return sorted(records, key=lambda record: (record.town_id, record.sequence_id, record.tile_id))


def _load_worker_town_mesh(cache_root: Path, town_id: str):
    cache_key = f"{cache_root}::{town_id}"
    if cache_key not in _WORKER_TOWN_MESHES:
        _WORKER_TOWN_MESHES[cache_key] = load_town_mesh_cache(cache_root / town_id, mmap=True)
    return _WORKER_TOWN_MESHES[cache_key]


def _process_entry(
    *,
    entry: dict[str, Any],
    observed_cache_root: str,
    town_mesh_cache_root: str,
    out_dir: str,
    teacher_config: dict[str, Any],
    seed: int,
    skip_completed_sequences: bool,
) -> dict[str, Any]:
    observed_cache_root_path = Path(observed_cache_root).expanduser().resolve()
    town_mesh_cache_root_path = Path(town_mesh_cache_root).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()

    town_id = str(entry["town_id"])
    sequence_id = str(entry["sequence_id"])
    observed_cache_path = observed_cache_root_path / town_id / sequence_id / "observed_cache.npz"
    town_cache_dir = town_mesh_cache_root_path / town_id
    sequence_out_dir = out_dir_path / town_id / sequence_id
    sequence_stats_path = sequence_out_dir / "sequence_patch_stats.json"

    if skip_completed_sequences and sequence_stats_path.exists():
        existing_count = sum(1 for _ in sequence_out_dir.glob("*.npz"))
        return {
            "status": "reused",
            "town_id": town_id,
            "sequence_id": sequence_id,
            "tiles_in": None,
            "patches_written": existing_count,
        }
    if not observed_cache_path.exists():
        return {
            "status": "skip_missing_observed_cache",
            "town_id": town_id,
            "sequence_id": sequence_id,
            "message": str(observed_cache_path),
        }
    if not town_cache_dir.exists():
        return {
            "status": "skip_missing_town_mesh_cache",
            "town_id": town_id,
            "sequence_id": sequence_id,
            "message": str(town_cache_dir),
        }

    raw_sequence = RawSequence.from_manifest_entry(entry)
    town_mesh_cache = _load_worker_town_mesh(town_mesh_cache_root_path, town_id)
    observed_cache = load_sequence_observed_cache(observed_cache_path)
    sequence_records = build_teacher_patches_for_sequence_v2(
        raw_sequence=raw_sequence,
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id=town_id,
        sequence_id=sequence_id,
        output_dir=sequence_out_dir,
        config=teacher_config,
        seed=seed,
    )
    return {
        "status": "built",
        "town_id": town_id,
        "sequence_id": sequence_id,
        "tiles_in": len(observed_cache.tile_centers),
        "patches_written": len(sequence_records),
    }


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build teacher patch cache v2 with visibility/free-space supervision."
    )
    parser.add_argument("--manifest", required=True, help="Path to the manifest JSON file.")
    parser.add_argument("--split_config", required=True, help="Path to the split YAML file.")
    parser.add_argument("--config", required=True, help="Path to the teacher patch v2 YAML file.")
    parser.add_argument("--observed_cache_dir", required=True, help="Root directory of observed caches.")
    parser.add_argument("--town_mesh_cache_root", required=True, help="Root directory of town mesh caches.")
    parser.add_argument("--out_dir", required=True, help="Output directory for teacher patch cache v2.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Split subsets to process.",
    )
    parser.add_argument("--debug_max_sequences", type=int, default=None, help="Optional sequence cap.")
    parser.add_argument(
        "--debug_max_tiles_per_sequence",
        type=int,
        default=None,
        help="Optional tile cap per sequence.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument(
        "--skip_completed_sequences",
        action="store_true",
        help="Reuse sequence directories that already contain sequence_patch_stats.json.",
    )
    parser.add_argument(
        "--rebuild_index_only",
        action="store_true",
        help="Do not build patches; only rebuild patch_index.jsonl from completed sequence directories.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    manifest = load_json(args.manifest)
    split_config = load_yaml(args.split_config)
    config = load_yaml(args.config)
    teacher_config = config.get("teacher_patch", config)
    if args.debug_max_sequences is not None:
        teacher_config["debug_max_sequences"] = args.debug_max_sequences
    if args.debug_max_tiles_per_sequence is not None:
        teacher_config["debug_max_tiles_per_sequence"] = args.debug_max_tiles_per_sequence

    entries = select_manifest_entries_by_split(
        manifest,
        split_config,
        subsets=tuple(args.subsets),
    )
    debug_max_sequences = teacher_config.get("debug_max_sequences")
    if debug_max_sequences is not None:
        entries = entries[: int(debug_max_sequences)]

    observed_cache_root = Path(args.observed_cache_dir).expanduser().resolve()
    town_mesh_cache_root = Path(args.town_mesh_cache_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_index_only:
        num_workers = max(1, int(args.num_workers))
        if num_workers == 1:
            for entry in entries:
                result = _process_entry(
                    entry=entry,
                    observed_cache_root=str(observed_cache_root),
                    town_mesh_cache_root=str(town_mesh_cache_root),
                    out_dir=str(out_dir),
                    teacher_config=teacher_config,
                    seed=args.seed,
                    skip_completed_sequences=bool(args.skip_completed_sequences),
                )
                if result["status"].startswith("skip_missing"):
                    print(f"{result['status']}: {result['message']}")
                    continue
                status_prefix = "reused" if result["status"] == "reused" else "built"
                print(
                    f"{status_prefix}: {result['sequence_id']} "
                    f"tiles_in={result['tiles_in']} "
                    f"patches_written={result['patches_written']}"
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _process_entry,
                        entry=entry,
                        observed_cache_root=str(observed_cache_root),
                        town_mesh_cache_root=str(town_mesh_cache_root),
                        out_dir=str(out_dir),
                        teacher_config=teacher_config,
                        seed=args.seed,
                        skip_completed_sequences=bool(args.skip_completed_sequences),
                    )
                    for entry in entries
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"].startswith("skip_missing"):
                        print(f"{result['status']}: {result['message']}")
                        continue
                    status_prefix = "reused" if result["status"] == "reused" else "built"
                    print(
                        f"{status_prefix}: {result['sequence_id']} "
                        f"tiles_in={result['tiles_in']} "
                        f"patches_written={result['patches_written']}"
                    )

    all_records = _collect_completed_patch_records_v2(out_dir)
    index_path = write_patch_index_jsonl(out_dir / "patch_index.jsonl", all_records)
    print(f"processed_sequences: {len(entries)}")
    print(f"written_patches: {len(all_records)}")
    print(f"patch_index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
