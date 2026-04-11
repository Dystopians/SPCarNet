from __future__ import annotations

from pathlib import Path

from ss3dm_prior.data.discovery import (
    EXPECTED_CAMERA_NAMES,
    EXPECTED_LIDAR_NAMES,
    discover_ss3dm_sequences,
    parse_num_frames_from_sequence_name,
)
from ss3dm_prior.data.manifest import build_manifest, validate_manifest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_fake_sequence(
    root: Path,
    *,
    town_id: str = "Town01",
    sequence_name: str = "3_streetsurf",
    break_depth_count: bool = False,
) -> Path:
    sequence_root = root / "DATA" / town_id / sequence_name
    mesh_path = root / "meshes" / "mesh" / f"{town_id}_obj.obj"

    _touch(mesh_path)
    _touch(sequence_root / "scenario.pt")
    _touch(sequence_root / "scenario.txt")

    for frame_idx in range(3):
        for camera_name in EXPECTED_CAMERA_NAMES:
            _touch(sequence_root / "images" / camera_name / f"{frame_idx:08d}.jpg")
            if not (break_depth_count and camera_name == "camera_FRONT" and frame_idx == 2):
                _touch(sequence_root / "depth_gts" / camera_name / f"{frame_idx:08d}.png")
            _touch(sequence_root / "masks" / camera_name / f"{frame_idx:08d}.npz")
        for lidar_name in EXPECTED_LIDAR_NAMES:
            _touch(sequence_root / "lidars" / lidar_name / f"{frame_idx:08d}.npz")

    return sequence_root


def test_parse_num_frames_from_sequence_name() -> None:
    assert parse_num_frames_from_sequence_name("550_streetsurf") == 550


def test_build_manifest_sequence_entry(tmp_path: Path) -> None:
    _write_fake_sequence(tmp_path)

    manifest = build_manifest(tmp_path)
    assert manifest["num_sequences"] == 1
    entry = manifest["entries"][0]

    assert entry["town_id"] == "Town01"
    assert entry["sequence_id"] == "Town01__3_streetsurf"
    assert entry["num_frames_from_name"] == 3
    assert sorted(entry["camera_names"]) == sorted(EXPECTED_CAMERA_NAMES)
    assert sorted(entry["lidar_names"]) == sorted(EXPECTED_LIDAR_NAMES)
    assert entry["frame_count_summary"]["images_jpg"]["camera_FRONT"] == 3
    assert entry["frame_count_summary"]["depth_png"]["camera_FRONT"] == 3
    assert entry["frame_count_summary"]["masks_npz"]["camera_FRONT"] == 3
    assert entry["frame_count_summary"]["lidars_npz"]["lidar_TOP"] == 3
    assert entry["all_required_paths_exist"] is True
    assert entry["notes"] == []


def test_validate_manifest_reports_frame_mismatch(tmp_path: Path) -> None:
    _write_fake_sequence(tmp_path, break_depth_count=True)
    manifest = build_manifest(tmp_path)
    report = validate_manifest(manifest)

    assert report["errors"] == []
    assert report["frame_consistent_entries"] == 0
    assert any("frame mismatch for camera_FRONT" in warning for warning in report["warnings"])


def test_discovery_ignores_non_expected_town(tmp_path: Path) -> None:
    _write_fake_sequence(tmp_path, town_id="Town01")
    _write_fake_sequence(tmp_path, town_id="Town99")

    records = discover_ss3dm_sequences(tmp_path)
    assert len(records) == 1
    assert records[0].town_id == "Town01"
