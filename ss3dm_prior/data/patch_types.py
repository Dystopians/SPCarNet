"""Patch sample and index record types for teacher patch caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PatchIndexRecord:
    patch_id: str
    town_id: str
    sequence_id: str
    tile_id: int
    patch_file: str
    num_local_faces: int
    num_observed_points_raw: int
    num_clean_points: int
    num_observed_points: int
    teacher_area_local: float
    planarity_hint: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class TeacherPatchSample:
    clean_points: np.ndarray
    clean_normals: np.ndarray
    observed_points: np.ndarray
    patch_center_world: np.ndarray
    patch_radius_m: float
    town_id: str
    sequence_id: str
    tile_id: int
    patch_id: str
    num_local_faces: int
    num_observed_points_raw: int
    teacher_area_local: float
    source_town_mesh_cache_dir: str
    source_sequence_observed_cache: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_npz_payload(self) -> dict[str, Any]:
        return {
            "clean_points": np.asarray(self.clean_points, dtype=np.float32),
            "clean_normals": np.asarray(self.clean_normals, dtype=np.float32),
            "observed_points": np.asarray(self.observed_points, dtype=np.float32),
            "patch_center_world": np.asarray(self.patch_center_world, dtype=np.float32),
            "patch_radius_m": np.asarray(float(self.patch_radius_m), dtype=np.float32),
            "town_id": np.asarray(self.town_id),
            "sequence_id": np.asarray(self.sequence_id),
            "tile_id": np.asarray(int(self.tile_id), dtype=np.int32),
            "patch_id": np.asarray(self.patch_id),
            "num_local_faces": np.asarray(int(self.num_local_faces), dtype=np.int32),
            "num_observed_points_raw": np.asarray(int(self.num_observed_points_raw), dtype=np.int32),
            "teacher_area_local": np.asarray(float(self.teacher_area_local), dtype=np.float32),
            "source_town_mesh_cache_dir": np.asarray(self.source_town_mesh_cache_dir),
            "source_sequence_observed_cache": np.asarray(self.source_sequence_observed_cache),
            "patch_metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }

    def save(self, patch_path: str | Path) -> Path:
        patch_path = Path(patch_path).expanduser().resolve()
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(patch_path, **self.to_npz_payload())
        return patch_path


def load_patch_npz(patch_path: str | Path) -> dict[str, Any]:
    patch_path = Path(patch_path).expanduser().resolve()
    with np.load(patch_path) as payload:
        data = {key: payload[key] for key in payload.files}
    return data
