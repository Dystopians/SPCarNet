"""Dataset discovery for the raw SS3DM directory layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_TOWN_IDS = (
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10",
)

EXPECTED_CAMERA_NAMES = (
    "camera_BACK",
    "camera_BACK_LEFT",
    "camera_BACK_RIGHT",
    "camera_FRONT",
    "camera_FRONT_LEFT",
    "camera_FRONT_RIGHT",
)

EXPECTED_LIDAR_NAMES = (
    "lidar_FRONT",
    "lidar_LEFT",
    "lidar_REAR",
    "lidar_RIGHT",
    "lidar_TOP",
)


def parse_num_frames_from_sequence_name(sequence_name: str) -> int:
    prefix, _, _ = sequence_name.partition("_")
    if not prefix.isdigit():
        raise ValueError(f"Cannot parse num_frames from sequence name: {sequence_name}")
    return int(prefix)


def _sorted_child_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_dir())


def _count_suffix_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix == suffix)


@dataclass(frozen=True)
class SequenceDiscoveryRecord:
    town_id: str
    sequence_id: str
    sequence_root: str
    town_mesh_obj_path: str
    scenario_pt_path: str
    scenario_txt_path: str
    num_frames_from_name: int
    camera_names: list[str]
    lidar_names: list[str]
    image_dir_map: dict[str, str]
    mask_dir_map: dict[str, str]
    depth_dir_map: dict[str, str]
    lidar_dir_map: dict[str, str]
    frame_count_summary: dict[str, dict[str, int]]
    all_required_paths_exist: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_ss3dm_sequences(root: str | Path) -> list[SequenceDiscoveryRecord]:
    root_path = Path(root).expanduser().resolve()
    data_root = root_path / "DATA"
    mesh_root = root_path / "meshes" / "mesh"

    if not data_root.exists():
        raise FileNotFoundError(f"Missing DATA directory under {root_path}")
    if not mesh_root.exists():
        raise FileNotFoundError(f"Missing meshes/mesh directory under {root_path}")

    records: list[SequenceDiscoveryRecord] = []
    for town_dir in _sorted_child_dirs(data_root):
        town_id = town_dir.name
        if town_id not in EXPECTED_TOWN_IDS:
            continue

        mesh_path = (mesh_root / f"{town_id}_obj.obj").resolve()
        for seq_dir in _sorted_child_dirs(town_dir):
            if not seq_dir.name.endswith("_streetsurf"):
                continue

            sequence_name = seq_dir.name
            sequence_id = f"{town_id}__{sequence_name}"
            num_frames_from_name = parse_num_frames_from_sequence_name(sequence_name)
            scenario_pt = (seq_dir / "scenario.pt").resolve()
            scenario_txt = (seq_dir / "scenario.txt").resolve()

            images_root = seq_dir / "images"
            masks_root = seq_dir / "masks"
            depth_root = seq_dir / "depth_gts"
            lidars_root = seq_dir / "lidars"

            camera_names = sorted(
                {
                    path.name
                    for root_dir in (images_root, masks_root, depth_root)
                    for path in _sorted_child_dirs(root_dir)
                }
            )
            lidar_names = [path.name for path in _sorted_child_dirs(lidars_root)]

            image_dir_map = {name: str((images_root / name).resolve()) for name in camera_names}
            mask_dir_map = {name: str((masks_root / name).resolve()) for name in camera_names}
            depth_dir_map = {name: str((depth_root / name).resolve()) for name in camera_names}
            lidar_dir_map = {name: str((lidars_root / name).resolve()) for name in lidar_names}

            frame_count_summary = {
                "images_jpg": {
                    name: _count_suffix_files(Path(path), ".jpg")
                    for name, path in image_dir_map.items()
                },
                "depth_png": {
                    name: _count_suffix_files(Path(path), ".png")
                    for name, path in depth_dir_map.items()
                },
                "masks_npz": {
                    name: _count_suffix_files(Path(path), ".npz")
                    for name, path in mask_dir_map.items()
                },
                "lidars_npz": {
                    name: _count_suffix_files(Path(path), ".npz")
                    for name, path in lidar_dir_map.items()
                },
            }

            required_paths = [
                seq_dir,
                scenario_pt,
                scenario_txt,
                mesh_path,
                images_root,
                masks_root,
                depth_root,
                lidars_root,
            ]
            required_paths.extend(Path(path) for path in image_dir_map.values())
            required_paths.extend(Path(path) for path in mask_dir_map.values())
            required_paths.extend(Path(path) for path in depth_dir_map.values())
            required_paths.extend(Path(path) for path in lidar_dir_map.values())
            all_required_paths_exist = all(path.exists() for path in required_paths)

            notes: list[str] = []
            missing_cameras = sorted(set(EXPECTED_CAMERA_NAMES) - set(camera_names))
            extra_cameras = sorted(set(camera_names) - set(EXPECTED_CAMERA_NAMES))
            missing_lidars = sorted(set(EXPECTED_LIDAR_NAMES) - set(lidar_names))
            extra_lidars = sorted(set(lidar_names) - set(EXPECTED_LIDAR_NAMES))

            if missing_cameras:
                notes.append(f"missing_cameras={','.join(missing_cameras)}")
            if extra_cameras:
                notes.append(f"extra_cameras={','.join(extra_cameras)}")
            if missing_lidars:
                notes.append(f"missing_lidars={','.join(missing_lidars)}")
            if extra_lidars:
                notes.append(f"extra_lidars={','.join(extra_lidars)}")

            for camera_name in camera_names:
                image_count = frame_count_summary["images_jpg"].get(camera_name, 0)
                depth_count = frame_count_summary["depth_png"].get(camera_name, 0)
                mask_count = frame_count_summary["masks_npz"].get(camera_name, 0)
                if len({image_count, depth_count, mask_count}) != 1:
                    notes.append(
                        "camera_count_mismatch="
                        f"{camera_name}:jpg={image_count},png={depth_count},npz={mask_count}"
                    )
                if image_count != num_frames_from_name:
                    notes.append(
                        f"camera_frame_count_vs_name={camera_name}:"
                        f"count={image_count},name={num_frames_from_name}"
                    )

            for lidar_name in lidar_names:
                lidar_count = frame_count_summary["lidars_npz"].get(lidar_name, 0)
                if lidar_count != num_frames_from_name:
                    notes.append(
                        f"lidar_frame_count_vs_name={lidar_name}:"
                        f"count={lidar_count},name={num_frames_from_name}"
                    )

            records.append(
                SequenceDiscoveryRecord(
                    town_id=town_id,
                    sequence_id=sequence_id,
                    sequence_root=str(seq_dir.resolve()),
                    town_mesh_obj_path=str(mesh_path),
                    scenario_pt_path=str(scenario_pt),
                    scenario_txt_path=str(scenario_txt),
                    num_frames_from_name=num_frames_from_name,
                    camera_names=camera_names,
                    lidar_names=lidar_names,
                    image_dir_map=image_dir_map,
                    mask_dir_map=mask_dir_map,
                    depth_dir_map=depth_dir_map,
                    lidar_dir_map=lidar_dir_map,
                    frame_count_summary=frame_count_summary,
                    all_required_paths_exist=all_required_paths_exist,
                    notes=notes,
                )
            )

    return records
