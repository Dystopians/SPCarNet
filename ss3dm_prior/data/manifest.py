"""Manifest building and validation for SS3DM sequence discovery."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ss3dm_prior.data.discovery import (
    EXPECTED_CAMERA_NAMES,
    EXPECTED_LIDAR_NAMES,
    discover_ss3dm_sequences,
)

MANIFEST_SCHEMA_VERSION = "ss3dm_prior_manifest/v1"


def build_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    entries = [record.to_dict() for record in discover_ss3dm_sequences(root_path)]
    town_ids = sorted({entry["town_id"] for entry in entries})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": "SS3DM_raw",
        "dataset_root": str(root_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_of_index": "sequence",
        "split_policy": "town_holdout_only",
        "num_sequences": len(entries),
        "town_ids": town_ids,
        "entries": entries,
    }


def summarize_manifest(manifest: dict[str, Any]) -> str:
    entries = manifest.get("entries", [])
    town_counts = Counter(entry["town_id"] for entry in entries)
    lines = [
        f"schema_version: {manifest.get('schema_version')}",
        f"dataset_root: {manifest.get('dataset_root')}",
        f"num_towns: {len(town_counts)}",
        f"num_sequences: {len(entries)}",
    ]
    for town_id in sorted(town_counts):
        lines.append(f"  - {town_id}: {town_counts[town_id]} sequences")
    return "\n".join(lines)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    entries = manifest.get("entries", [])
    expected_cameras = set(EXPECTED_CAMERA_NAMES)
    expected_lidars = set(EXPECTED_LIDAR_NAMES)

    errors: list[str] = []
    warnings: list[str] = []
    town_sequence_counts = Counter()
    complete_camera_entries = 0
    complete_lidar_entries = 0
    frame_consistent_entries = 0

    for entry in entries:
        sequence_id = entry["sequence_id"]
        town_sequence_counts[entry["town_id"]] += 1

        critical_paths = [
            ("sequence_root", entry["sequence_root"]),
            ("town_mesh_obj_path", entry["town_mesh_obj_path"]),
            ("scenario_pt_path", entry["scenario_pt_path"]),
            ("scenario_txt_path", entry["scenario_txt_path"]),
        ]
        for path_map_key in ("image_dir_map", "mask_dir_map", "depth_dir_map", "lidar_dir_map"):
            for sensor_name, sensor_path in entry.get(path_map_key, {}).items():
                critical_paths.append((f"{path_map_key}.{sensor_name}", sensor_path))

        for label, raw_path in critical_paths:
            if not Path(raw_path).exists():
                errors.append(f"{sequence_id}: missing {label} -> {raw_path}")

        if not entry.get("all_required_paths_exist", False):
            errors.append(f"{sequence_id}: all_required_paths_exist=False")

        camera_names = set(entry.get("camera_names", []))
        lidar_names = set(entry.get("lidar_names", []))
        if camera_names == expected_cameras:
            complete_camera_entries += 1
        else:
            warnings.append(
                f"{sequence_id}: camera set mismatch -> "
                f"expected={sorted(expected_cameras)} actual={sorted(camera_names)}"
            )
        if lidar_names == expected_lidars:
            complete_lidar_entries += 1
        else:
            warnings.append(
                f"{sequence_id}: lidar set mismatch -> "
                f"expected={sorted(expected_lidars)} actual={sorted(lidar_names)}"
            )

        num_frames_from_name = entry["num_frames_from_name"]
        frame_count_summary = entry.get("frame_count_summary", {})
        images_jpg = frame_count_summary.get("images_jpg", {})
        depth_png = frame_count_summary.get("depth_png", {})
        masks_npz = frame_count_summary.get("masks_npz", {})
        lidars_npz = frame_count_summary.get("lidars_npz", {})

        entry_is_frame_consistent = True
        for camera_name in sorted(camera_names):
            image_count = images_jpg.get(camera_name, 0)
            depth_count = depth_png.get(camera_name, 0)
            mask_count = masks_npz.get(camera_name, 0)
            if len({image_count, depth_count, mask_count}) != 1:
                entry_is_frame_consistent = False
                warnings.append(
                    f"{sequence_id}: frame mismatch for {camera_name} -> "
                    f"jpg={image_count}, depth={depth_count}, mask={mask_count}"
                )
            if image_count != num_frames_from_name:
                entry_is_frame_consistent = False
                warnings.append(
                    f"{sequence_id}: image count vs sequence name mismatch for {camera_name} -> "
                    f"count={image_count}, expected={num_frames_from_name}"
                )

        for lidar_name in sorted(lidar_names):
            lidar_count = lidars_npz.get(lidar_name, 0)
            if lidar_count != num_frames_from_name:
                entry_is_frame_consistent = False
                warnings.append(
                    f"{sequence_id}: lidar count vs sequence name mismatch for {lidar_name} -> "
                    f"count={lidar_count}, expected={num_frames_from_name}"
                )

        if entry_is_frame_consistent:
            frame_consistent_entries += 1

    return {
        "num_entries": len(entries),
        "town_sequence_counts": dict(sorted(town_sequence_counts.items())),
        "complete_camera_entries": complete_camera_entries,
        "complete_lidar_entries": complete_lidar_entries,
        "frame_consistent_entries": frame_consistent_entries,
        "warnings": warnings,
        "errors": errors,
    }
