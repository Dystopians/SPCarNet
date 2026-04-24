"""Training dataset for online-corrupted teacher patch samples."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np


def zlib_crc32_bytes(data: bytes) -> int:
    return zlib.crc32(data)

from ss3dm_prior.data.corruptions import apply_patch_corruptions, resolve_corruption_severity_scale
from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.patch_types import load_patch_npz
from ss3dm_prior.utils.io import load_yaml

import torch
from torch.utils.data import Dataset


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _random_rotation_matrix(rng: np.random.Generator, axis: str = "z") -> np.ndarray:
    """Return a 3x3 rotation matrix.

    ``axis='z'`` — rotate about vertical Z (random azimuth, preserves up).
    ``axis='so3'`` — full SO(3) via uniformly-random quaternion (Shoemake '92).
    """
    if axis == "z":
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        c, s = np.cos(theta), np.sin(theta)
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    # Shoemake uniform quaternion.
    u1, u2, u3 = rng.uniform(size=3)
    q = np.asarray(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        dtype=np.float32,
    )
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _apply_rotation_inplace(sample: dict[str, Any], rotation: np.ndarray) -> None:
    """Rotate every 3-D point/normal field of ``sample`` by ``rotation``.

    Applies the same rotation to clean, corrupted, observed, visible/hidden
    splits, queries, and hard-negative fields so the per-sample geometry
    stays self-consistent. Scalar statistics and non-3D fields are ignored.
    """
    fields_xyz = [
        "clean_points", "corrupted_points", "observed_points",
        "visible_clean_points", "hidden_clean_points",
        "surface_query_points", "free_query_points", "unknown_query_points",
        "query_points_all", "free_space_query_hard_negatives",
    ]
    fields_normals = [
        "clean_normals", "corrupted_normals",
        "visible_clean_normals", "hidden_clean_normals",
    ]
    for key in fields_xyz + fields_normals:
        arr = sample.get(key)
        if arr is None:
            continue
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[-1] == 3 and arr.size > 0:
            sample[key] = arr @ rotation.T
    # Symmetry plane normal is a direction, rotate it.
    n = sample.get("symmetry_plane_normal")
    if isinstance(n, np.ndarray) and n.shape[-1] == 3:
        sample["symmetry_plane_normal"] = (n.reshape(1, 3) @ rotation.T).reshape(3)
    # patch_center_world is a 3-vector — rotate too.
    c = sample.get("patch_center_world")
    if isinstance(c, np.ndarray) and c.shape[-1] == 3:
        sample["patch_center_world"] = (c.reshape(1, 3) @ rotation.T).reshape(3)


class TeacherPatchTrainDataset(Dataset):
    def __init__(
        self,
        *,
        patch_index_path: str | Path,
        records: list[dict[str, Any]] | None = None,
        split_config: str | Path | dict[str, Any] | None = None,
        subsets: tuple[str, ...] = ("train",),
        corruption_config: dict[str, Any] | None = None,
        seed: int = 0,
        dynamic_corruption: bool = True,
    ) -> None:
        self.patch_index_path = Path(patch_index_path).expanduser().resolve()
        self.records = list(records) if records is not None else read_patch_index_jsonl(self.patch_index_path)
        self.seed = int(seed)
        self.corruption_config = corruption_config or {}
        self._sample_visit_counts: dict[str, int] = defaultdict(int)
        self.dynamic_corruption = bool(dynamic_corruption)
        self.current_epoch = 0

        if split_config is not None:
            if isinstance(split_config, (str, Path)):
                split_data = load_yaml(split_config)
            else:
                split_data = split_config
            selected_towns: set[str] = set()
            for subset in subsets:
                selected_towns.update(split_data.get(f"{subset}_towns", []))
            self.records = [
                record for record in self.records if str(record["town_id"]) in selected_towns
            ]

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def current_corruption_severity_scale(self) -> float:
        if not self.dynamic_corruption:
            return 1.0
        return resolve_corruption_severity_scale(self.corruption_config, epoch=self.current_epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        patch = load_patch_npz(record["patch_file"])
        clean_points = np.asarray(patch["clean_points"], dtype=np.float32)
        clean_normals = np.asarray(patch["clean_normals"], dtype=np.float32)
        observed_points = np.asarray(patch["observed_points"], dtype=np.float32)
        surface_query_points = np.asarray(patch.get("surface_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        surface_query_labels = np.asarray(patch.get("surface_query_labels", np.zeros((0,), dtype=np.int8)))
        free_query_points = np.asarray(patch.get("free_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        free_query_labels = np.asarray(patch.get("free_query_labels", np.zeros((0,), dtype=np.int8)))
        unknown_query_points = np.asarray(patch.get("unknown_query_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        query_points_all = np.asarray(patch.get("query_points_all", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        query_labels_all = np.asarray(patch.get("query_labels_all", np.zeros((0,), dtype=np.int8)))
        query_ignore_mask = np.asarray(patch.get("query_ignore_mask", np.zeros((0,), dtype=bool)), dtype=bool)
        patch_center_world = np.asarray(patch["patch_center_world"], dtype=np.float32)
        patch_id = str(_scalar(patch["patch_id"]))
        town_id = str(_scalar(patch["town_id"]))
        sequence_id = str(_scalar(patch["sequence_id"]))
        scale_id = int(_scalar(patch.get("scale_id", np.asarray(0, dtype=np.int32))))
        patch_radius_m = float(_scalar(patch.get("patch_radius_m", np.asarray(1.0, dtype=np.float32))))
        patch_cache_format_version = int(_scalar(patch.get("patch_cache_format_version", np.asarray(1, dtype=np.int32))))
        difficulty_components_json = str(_scalar(patch.get("difficulty_components_json", np.asarray("{}"))))
        patch_metadata_json = str(_scalar(patch.get("patch_metadata_json", np.asarray("{}"))))
        if self.dynamic_corruption:
            visit_count = self._sample_visit_counts[patch_id]
            self._sample_visit_counts[patch_id] += 1
            sample_key = f"{patch_id}__visit_{visit_count}"
        else:
            sample_key = patch_id

        # Symmetry targets (rotated alongside geometry below).
        symmetry_plane_normal = np.asarray(
            patch.get("symmetry_plane_normal", np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        symmetry_plane_offset = float(_scalar(patch.get("symmetry_plane_offset", np.asarray(0.0, dtype=np.float32))))
        visible_clean_points_np = np.asarray(
            patch.get("visible_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32
        )
        visible_clean_normals_np = np.asarray(
            patch.get("visible_clean_normals", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32
        )
        hidden_clean_points_np = np.asarray(
            patch.get("hidden_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32
        )
        hidden_clean_normals_np = np.asarray(
            patch.get("hidden_clean_normals", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32
        )

        # Random SO(3) rotation on TRAIN only (dynamic_corruption=True). Applied
        # BEFORE the corruption pipeline so the corrupted output lives in the
        # same rotated frame as clean / observed / queries / symmetry. This is
        # pure data augmentation — 37M params on 1300 samples was memorising
        # the training set's specific orientations; with random rotations the
        # effective dataset becomes ~360× larger without adding parameters.
        if self.dynamic_corruption:
            rot_rng = np.random.default_rng(
                (self.seed + zlib_crc32_bytes(sample_key.encode("utf-8"))) & 0xFFFFFFFF
            )
            rotation = _random_rotation_matrix(rot_rng, axis="so3")
            clean_points = clean_points @ rotation.T
            clean_normals = clean_normals @ rotation.T
            observed_points = observed_points @ rotation.T
            if len(surface_query_points) > 0:
                surface_query_points = surface_query_points @ rotation.T
            if len(free_query_points) > 0:
                free_query_points = free_query_points @ rotation.T
            if len(unknown_query_points) > 0:
                unknown_query_points = unknown_query_points @ rotation.T
            if len(query_points_all) > 0:
                query_points_all = query_points_all @ rotation.T
            if len(visible_clean_points_np) > 0:
                visible_clean_points_np = visible_clean_points_np @ rotation.T
                visible_clean_normals_np = visible_clean_normals_np @ rotation.T
            if len(hidden_clean_points_np) > 0:
                hidden_clean_points_np = hidden_clean_points_np @ rotation.T
                hidden_clean_normals_np = hidden_clean_normals_np @ rotation.T
            # symmetry plane rotates with the object.
            symmetry_plane_normal = (symmetry_plane_normal.reshape(1, 3) @ rotation.T).reshape(3)

        severity_scale = self.current_corruption_severity_scale() if self.dynamic_corruption else 1.0
        corruption = apply_patch_corruptions(
            clean_points=clean_points,
            clean_normals=clean_normals,
            observed_points=observed_points,
            config=self.corruption_config,
            seed=self.seed,
            sample_key=sample_key,
            severity_scale=severity_scale,
        )
        sample = {
            "clean_points": torch.from_numpy(clean_points),
            "clean_normals": torch.from_numpy(clean_normals),
            "observed_points": torch.from_numpy(observed_points),
            "corrupted_points": torch.from_numpy(corruption.corrupted_points),
            "corrupted_normals": torch.from_numpy(corruption.corrupted_normals),
            "point_defect_target": torch.from_numpy(corruption.point_defect_target),
            "corruption_score_target": torch.tensor(
                corruption.corruption_score_target,
                dtype=torch.float32,
            ),
            "surface_query_points": torch.from_numpy(surface_query_points),
            "surface_query_labels": torch.from_numpy(surface_query_labels),
            "free_query_points": torch.from_numpy(free_query_points),
            "free_query_labels": torch.from_numpy(free_query_labels),
            "free_space_query_hard_negatives": torch.from_numpy(
                np.asarray(patch.get("free_space_query_hard_negatives", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            ),
            "unknown_query_points": torch.from_numpy(unknown_query_points),
            "query_points_all": torch.from_numpy(query_points_all),
            "query_labels_all": torch.from_numpy(query_labels_all),
            "query_ignore_mask": torch.from_numpy(query_ignore_mask),
            "visible_clean_points": torch.from_numpy(visible_clean_points_np),
            "visible_clean_normals": torch.from_numpy(visible_clean_normals_np),
            "hidden_clean_points": torch.from_numpy(hidden_clean_points_np),
            "hidden_clean_normals": torch.from_numpy(hidden_clean_normals_np),
            "surface_support_mask": torch.from_numpy(
                np.asarray(patch.get("surface_support_mask", np.zeros((0,), dtype=bool)), dtype=bool)
            ),
            "camera_support_count": torch.tensor(
                float(_scalar(patch.get("camera_support_count", np.asarray(0, dtype=np.int32)))),
                dtype=torch.float32,
            ),
            "lidar_support_count": torch.tensor(
                float(_scalar(patch.get("lidar_support_count", np.asarray(0, dtype=np.int32)))),
                dtype=torch.float32,
            ),
            "visible_surface_fraction": torch.tensor(
                float(_scalar(patch.get("visible_surface_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "visible_support_fraction": torch.tensor(
                float(_scalar(patch.get("visible_support_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "hidden_surface_fraction": torch.tensor(
                float(_scalar(patch.get("hidden_surface_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "free_space_fraction": torch.tensor(
                float(_scalar(patch.get("free_space_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "unknown_fraction": torch.tensor(
                float(_scalar(patch.get("unknown_fraction", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "free_space_hard_negative_count": torch.tensor(
                float(_scalar(patch.get("free_space_hard_negative_count", np.asarray(0, dtype=np.int32)))),
                dtype=torch.float32,
            ),
            "intrinsic_patch_difficulty_target": torch.tensor(
                float(
                    _scalar(
                        patch.get(
                            "intrinsic_patch_difficulty_target",
                            np.asarray(0.0, dtype=np.float32),
                        )
                    )
                ),
                dtype=torch.float32,
            ),
            "symmetry_plane_normal": torch.from_numpy(symmetry_plane_normal.astype(np.float32)),
            "symmetry_plane_offset": torch.tensor(symmetry_plane_offset, dtype=torch.float32),
            "symmetry_target_confidence": torch.tensor(
                float(_scalar(patch.get("symmetry_target_confidence", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "symmetry_chamfer_residual": torch.tensor(
                float(_scalar(patch.get("symmetry_chamfer_residual", np.asarray(0.0, dtype=np.float32)))),
                dtype=torch.float32,
            ),
            "patch_center_world": torch.from_numpy(patch_center_world),
            "patch_radius_m": torch.tensor(patch_radius_m, dtype=torch.float32),
            "scale_id": torch.tensor(float(scale_id), dtype=torch.float32),
            "town_id": town_id,
            "sequence_id": sequence_id,
            "patch_id": patch_id,
            "patch_cache_format_version": patch_cache_format_version,
            "difficulty_components": json.loads(difficulty_components_json),
            "patch_metadata": json.loads(patch_metadata_json),
            "corruption_metadata": corruption.metadata,
        }
        return sample
