"""Patch sample and index record types for teacher patch caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_labels() -> np.ndarray:
    return np.zeros((0,), dtype=np.int8)


def _empty_ignore_mask() -> np.ndarray:
    return np.zeros((0,), dtype=bool)


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
    scale_id: int = 0
    patch_radius_m: float = 0.0
    patch_cache_format_version: int = 1
    num_surface_query_points: int = 0
    num_free_query_points: int = 0
    num_unknown_query_points: int = 0
    camera_support_count: int = 0
    lidar_support_count: int = 0
    visible_surface_fraction: float = 0.0
    free_space_fraction: float = 0.0
    unknown_fraction: float = 0.0
    intrinsic_patch_difficulty_target: float = 0.0
    num_visible_clean_points: int = 0
    num_hidden_clean_points: int = 0
    visible_support_fraction: float = 0.0
    hidden_surface_fraction: float = 0.0
    free_space_hard_negative_count: int = 0
    symmetry_target_confidence: float = 0.0
    symmetry_chamfer_residual: float = 0.0
    difficulty_components_json: dict[str, Any] = field(default_factory=dict)

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
    patch_cache_format_version: int = 1
    surface_query_points: np.ndarray = field(default_factory=_empty_points)
    surface_query_labels: np.ndarray = field(default_factory=_empty_labels)
    free_query_points: np.ndarray = field(default_factory=_empty_points)
    free_query_labels: np.ndarray = field(default_factory=_empty_labels)
    free_space_query_hard_negatives: np.ndarray = field(default_factory=_empty_points)
    unknown_query_points: np.ndarray = field(default_factory=_empty_points)
    query_points_all: np.ndarray = field(default_factory=_empty_points)
    query_labels_all: np.ndarray = field(default_factory=_empty_labels)
    query_ignore_mask: np.ndarray = field(default_factory=_empty_ignore_mask)
    visible_clean_points: np.ndarray = field(default_factory=_empty_points)
    visible_clean_normals: np.ndarray = field(default_factory=_empty_points)
    hidden_clean_points: np.ndarray = field(default_factory=_empty_points)
    hidden_clean_normals: np.ndarray = field(default_factory=_empty_points)
    surface_support_mask: np.ndarray = field(default_factory=_empty_ignore_mask)
    camera_support_count: int = 0
    lidar_support_count: int = 0
    visible_surface_fraction: float = 0.0
    visible_support_fraction: float = 0.0
    hidden_surface_fraction: float = 0.0
    free_space_fraction: float = 0.0
    unknown_fraction: float = 0.0
    free_space_hard_negative_count: int = 0
    intrinsic_patch_difficulty_target: float = 0.0
    difficulty_components_json: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_id: int = 0
    # Symmetry targets (CarNet_v0 / Phase 2 / A2). Populated by the cache
    # builder via `ss3dm_prior.data.symmetry_targets.estimate_symmetry_plane`.
    symmetry_plane_normal: np.ndarray = field(default_factory=lambda: np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    symmetry_plane_offset: float = 0.0
    symmetry_target_confidence: float = 0.0
    symmetry_chamfer_residual: float = 0.0

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
            "scale_id": np.asarray(int(self.scale_id), dtype=np.int32),
            "patch_id": np.asarray(self.patch_id),
            "num_local_faces": np.asarray(int(self.num_local_faces), dtype=np.int32),
            "num_observed_points_raw": np.asarray(int(self.num_observed_points_raw), dtype=np.int32),
            "teacher_area_local": np.asarray(float(self.teacher_area_local), dtype=np.float32),
            "source_town_mesh_cache_dir": np.asarray(self.source_town_mesh_cache_dir),
            "source_sequence_observed_cache": np.asarray(self.source_sequence_observed_cache),
            "patch_cache_format_version": np.asarray(int(self.patch_cache_format_version), dtype=np.int32),
            "surface_query_points": np.asarray(self.surface_query_points, dtype=np.float32),
            "surface_query_labels": np.asarray(self.surface_query_labels, dtype=np.int8),
            "free_query_points": np.asarray(self.free_query_points, dtype=np.float32),
            "free_query_labels": np.asarray(self.free_query_labels, dtype=np.int8),
            "free_space_query_hard_negatives": np.asarray(
                self.free_space_query_hard_negatives,
                dtype=np.float32,
            ),
            "unknown_query_points": np.asarray(self.unknown_query_points, dtype=np.float32),
            "query_points_all": np.asarray(self.query_points_all, dtype=np.float32),
            "query_labels_all": np.asarray(self.query_labels_all, dtype=np.int8),
            "query_ignore_mask": np.asarray(self.query_ignore_mask, dtype=bool),
            "visible_clean_points": np.asarray(self.visible_clean_points, dtype=np.float32),
            "visible_clean_normals": np.asarray(self.visible_clean_normals, dtype=np.float32),
            "hidden_clean_points": np.asarray(self.hidden_clean_points, dtype=np.float32),
            "hidden_clean_normals": np.asarray(self.hidden_clean_normals, dtype=np.float32),
            "surface_support_mask": np.asarray(self.surface_support_mask, dtype=bool),
            "camera_support_count": np.asarray(int(self.camera_support_count), dtype=np.int32),
            "lidar_support_count": np.asarray(int(self.lidar_support_count), dtype=np.int32),
            "visible_surface_fraction": np.asarray(float(self.visible_surface_fraction), dtype=np.float32),
            "visible_support_fraction": np.asarray(float(self.visible_support_fraction), dtype=np.float32),
            "hidden_surface_fraction": np.asarray(float(self.hidden_surface_fraction), dtype=np.float32),
            "free_space_fraction": np.asarray(float(self.free_space_fraction), dtype=np.float32),
            "unknown_fraction": np.asarray(float(self.unknown_fraction), dtype=np.float32),
            "free_space_hard_negative_count": np.asarray(int(self.free_space_hard_negative_count), dtype=np.int32),
            "intrinsic_patch_difficulty_target": np.asarray(
                float(self.intrinsic_patch_difficulty_target),
                dtype=np.float32,
            ),
            "difficulty_components_json": np.asarray(
                json.dumps(self.difficulty_components_json, sort_keys=True)
            ),
            "patch_metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
            "symmetry_plane_normal": np.asarray(self.symmetry_plane_normal, dtype=np.float32),
            "symmetry_plane_offset": np.asarray(float(self.symmetry_plane_offset), dtype=np.float32),
            "symmetry_target_confidence": np.asarray(float(self.symmetry_target_confidence), dtype=np.float32),
            "symmetry_chamfer_residual": np.asarray(float(self.symmetry_chamfer_residual), dtype=np.float32),
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
    defaults: dict[str, Any] = {
        "patch_cache_format_version": np.asarray(1, dtype=np.int32),
        "surface_query_points": _empty_points(),
        "surface_query_labels": _empty_labels(),
        "free_query_points": _empty_points(),
        "free_query_labels": _empty_labels(),
        "free_space_query_hard_negatives": _empty_points(),
        "unknown_query_points": _empty_points(),
        "query_points_all": _empty_points(),
        "query_labels_all": _empty_labels(),
        "query_ignore_mask": _empty_ignore_mask(),
        "visible_clean_points": _empty_points(),
        "visible_clean_normals": _empty_points(),
        "hidden_clean_points": _empty_points(),
        "hidden_clean_normals": _empty_points(),
        "surface_support_mask": _empty_ignore_mask(),
        "camera_support_count": np.asarray(0, dtype=np.int32),
        "lidar_support_count": np.asarray(0, dtype=np.int32),
        "visible_surface_fraction": np.asarray(0.0, dtype=np.float32),
        "visible_support_fraction": np.asarray(0.0, dtype=np.float32),
        "hidden_surface_fraction": np.asarray(0.0, dtype=np.float32),
        "free_space_fraction": np.asarray(0.0, dtype=np.float32),
        "unknown_fraction": np.asarray(0.0, dtype=np.float32),
        "free_space_hard_negative_count": np.asarray(0, dtype=np.int32),
        "scale_id": np.asarray(0, dtype=np.int32),
        "intrinsic_patch_difficulty_target": np.asarray(0.0, dtype=np.float32),
        "difficulty_components_json": np.asarray("{}"),
        "symmetry_plane_normal": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "symmetry_plane_offset": np.asarray(0.0, dtype=np.float32),
        "symmetry_target_confidence": np.asarray(0.0, dtype=np.float32),
        "symmetry_chamfer_residual": np.asarray(0.0, dtype=np.float32),
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    return data
