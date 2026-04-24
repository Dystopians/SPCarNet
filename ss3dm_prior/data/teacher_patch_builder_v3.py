"""Build teacher patch cache v3 with multi-scale visible/hidden semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.data.teacher_patch_builder import (
    SequenceObservedCache,
    _build_patch_metadata,
    build_patch_from_tile,
)
from ss3dm_prior.data.visibility_queries import build_patch_visibility_queries
from ss3dm_prior.utils.io import dump_json


def _radius_token(radius_m: float) -> str:
    return f"r{float(radius_m):.2f}m".replace(".", "p")


def make_multiscale_patch_id(sequence_id: str, tile_id: int, scale_id: int, patch_radius_m: float) -> str:
    return f"{sequence_id}__tile_{tile_id:06d}__scale_{scale_id:02d}__{_radius_token(patch_radius_m)}"


def _min_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if len(points_a) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(points_b) == 0:
        return np.full((len(points_a),), np.inf, dtype=np.float32)
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.linalg.norm(diff, axis=-1).min(axis=1).astype(np.float32)


def _visible_hidden_clean_split(
    *,
    clean_points_local: np.ndarray,
    clean_normals_local: np.ndarray,
    observed_points_local: np.ndarray,
    support_radius_local: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_mask = _min_distances(clean_points_local, observed_points_local) <= float(support_radius_local)
    return (
        clean_points_local[support_mask],
        clean_normals_local[support_mask],
        clean_points_local[~support_mask],
        clean_normals_local[~support_mask],
        support_mask.astype(bool),
    )


def _free_space_hard_negatives(
    *,
    free_query_points_local: np.ndarray,
    clean_points_local: np.ndarray,
    observed_points_local: np.ndarray,
    hard_negative_radius_local: float,
) -> np.ndarray:
    if len(free_query_points_local) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    clean_dist = _min_distances(free_query_points_local, clean_points_local)
    observed_dist = _min_distances(free_query_points_local, observed_points_local)
    hard_mask = np.minimum(clean_dist, observed_dist) <= float(hard_negative_radius_local)
    return np.asarray(free_query_points_local[hard_mask], dtype=np.float32)


def build_patch_from_tile_v3(
    *,
    raw_sequence: Any,
    town_mesh_cache: Any,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    tile_id: int,
    scale_id: int,
    patch_center_world: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> TeacherPatchSample | None:
    per_scale_config = dict(config)
    patch_radius_m = float(config["patch_radius_m_list"][scale_id])
    per_scale_config["patch_radius_m"] = patch_radius_m

    base_sample = build_patch_from_tile(
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_center_world=patch_center_world,
        config=per_scale_config,
        seed=seed,
    )
    if base_sample is None:
        return None

    face_query_margin_m = float(per_scale_config["face_query_margin_m"])
    town_mesh_unit_scale = float(per_scale_config.get("town_mesh_unit_scale", 1.0))
    query_config = dict(per_scale_config.get("visibility_queries", {}))
    support_radius_local = float(
        query_config.get(
            "visible_clean_support_radius_local",
            query_config.get("surface_visibility_radius_local", 0.08),
        )
    )
    hard_negative_radius_local = float(query_config.get("free_space_hard_negative_radius_local", 0.10))
    patch_center_world = np.asarray(patch_center_world, dtype=np.float32).reshape(3)

    local_face_mask = town_mesh_cache.query_faces_in_radius(
        patch_center_world,
        patch_radius_m,
        margin=face_query_margin_m,
        coordinate_scale=town_mesh_unit_scale,
    )
    local_mesh_dict = town_mesh_cache.build_local_mesh_from_face_mask(
        local_face_mask,
        coordinate_scale=town_mesh_unit_scale,
    )
    local_mesh = trimesh.Trimesh(
        vertices=np.asarray(local_mesh_dict["vertices"], dtype=np.float32),
        faces=np.asarray(local_mesh_dict["faces"], dtype=np.int64),
        process=False,
    )
    visibility_bundle = build_patch_visibility_queries(
        raw_sequence=raw_sequence,
        local_mesh=local_mesh,
        patch_center_world=patch_center_world,
        patch_radius_m=patch_radius_m,
        clean_points_local=np.asarray(base_sample.clean_points, dtype=np.float32),
        clean_normals_local=np.asarray(base_sample.clean_normals, dtype=np.float32),
        observed_points_local=np.asarray(base_sample.observed_points, dtype=np.float32),
        planarity_hint=float(base_sample.metadata.get("planarity_hint", 0.0)),
        query_config=query_config,
        seed=seed,
        patch_id=make_multiscale_patch_id(sequence_id, tile_id, scale_id, patch_radius_m),
    )
    visible_clean_points, visible_clean_normals, hidden_clean_points, hidden_clean_normals, support_mask = (
        _visible_hidden_clean_split(
            clean_points_local=np.asarray(base_sample.clean_points, dtype=np.float32),
            clean_normals_local=np.asarray(base_sample.clean_normals, dtype=np.float32),
            observed_points_local=np.asarray(base_sample.observed_points, dtype=np.float32),
            support_radius_local=support_radius_local,
        )
    )
    free_space_query_hard_negatives = _free_space_hard_negatives(
        free_query_points_local=visibility_bundle.free_query_points,
        clean_points_local=np.asarray(base_sample.clean_points, dtype=np.float32),
        observed_points_local=np.asarray(base_sample.observed_points, dtype=np.float32),
        hard_negative_radius_local=hard_negative_radius_local,
    )
    clean_count = max(int(len(base_sample.clean_points)), 1)
    visible_support_fraction = float(len(visible_clean_points) / clean_count)
    hidden_surface_fraction = float(len(hidden_clean_points) / clean_count)

    metadata = _build_patch_metadata(
        clean_points_local=np.asarray(base_sample.clean_points, dtype=np.float32),
        clean_normals_local=np.asarray(base_sample.clean_normals, dtype=np.float32),
        observed_points_local=np.asarray(base_sample.observed_points, dtype=np.float32),
        local_face_areas=np.asarray(local_mesh_dict["face_areas"], dtype=np.float32),
        patch_center_world=patch_center_world,
        teacher_area_local=base_sample.teacher_area_local,
    )
    metadata.update(
        {
            "patch_cache_format_version": 3,
            "scale_id": int(scale_id),
            "patch_radius_m": patch_radius_m,
            "patch_id_rule": "{sequence_id}__tile_{tile_id:06d}__scale_{scale_id:02d}__r{patch_radius_m:.2f}m".replace(
                ".",
                "p",
            ),
            "num_surface_query_points": int(len(visibility_bundle.surface_query_points)),
            "num_free_query_points": int(len(visibility_bundle.free_query_points)),
            "num_unknown_query_points": int(len(visibility_bundle.unknown_query_points)),
            "num_visible_clean_points": int(len(visible_clean_points)),
            "num_hidden_clean_points": int(len(hidden_clean_points)),
            "free_space_hard_negative_count": int(len(free_space_query_hard_negatives)),
            "camera_support_count": int(visibility_bundle.camera_support_count),
            "lidar_support_count": int(visibility_bundle.lidar_support_count),
            "visible_surface_fraction": float(visibility_bundle.visible_surface_fraction),
            "visible_support_fraction": visible_support_fraction,
            "hidden_surface_fraction": hidden_surface_fraction,
            "free_space_fraction": float(visibility_bundle.free_space_fraction),
            "unknown_fraction": float(visibility_bundle.unknown_fraction),
            "intrinsic_patch_difficulty_target": float(visibility_bundle.intrinsic_patch_difficulty_target),
        }
    )
    return TeacherPatchSample(
        clean_points=np.asarray(base_sample.clean_points, dtype=np.float32),
        clean_normals=np.asarray(base_sample.clean_normals, dtype=np.float32),
        observed_points=np.asarray(base_sample.observed_points, dtype=np.float32),
        patch_center_world=np.asarray(base_sample.patch_center_world, dtype=np.float32),
        patch_radius_m=patch_radius_m,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_id=make_multiscale_patch_id(sequence_id, tile_id, scale_id, patch_radius_m),
        scale_id=int(scale_id),
        num_local_faces=base_sample.num_local_faces,
        num_observed_points_raw=base_sample.num_observed_points_raw,
        teacher_area_local=base_sample.teacher_area_local,
        source_town_mesh_cache_dir=base_sample.source_town_mesh_cache_dir,
        source_sequence_observed_cache=base_sample.source_sequence_observed_cache,
        patch_cache_format_version=3,
        surface_query_points=visibility_bundle.surface_query_points,
        surface_query_labels=visibility_bundle.surface_query_labels,
        free_query_points=visibility_bundle.free_query_points,
        free_query_labels=visibility_bundle.free_query_labels,
        free_space_query_hard_negatives=free_space_query_hard_negatives,
        unknown_query_points=visibility_bundle.unknown_query_points,
        query_points_all=visibility_bundle.query_points_all,
        query_labels_all=visibility_bundle.query_labels_all,
        query_ignore_mask=visibility_bundle.query_ignore_mask,
        visible_clean_points=visible_clean_points,
        visible_clean_normals=visible_clean_normals,
        hidden_clean_points=hidden_clean_points,
        hidden_clean_normals=hidden_clean_normals,
        surface_support_mask=support_mask,
        camera_support_count=visibility_bundle.camera_support_count,
        lidar_support_count=visibility_bundle.lidar_support_count,
        visible_surface_fraction=visibility_bundle.visible_surface_fraction,
        visible_support_fraction=visible_support_fraction,
        hidden_surface_fraction=hidden_surface_fraction,
        free_space_fraction=visibility_bundle.free_space_fraction,
        unknown_fraction=visibility_bundle.unknown_fraction,
        free_space_hard_negative_count=int(len(free_space_query_hard_negatives)),
        intrinsic_patch_difficulty_target=visibility_bundle.intrinsic_patch_difficulty_target,
        difficulty_components_json=visibility_bundle.difficulty_components_json,
        metadata=metadata,
    )


def build_teacher_patches_for_sequence_v3(
    *,
    raw_sequence: Any,
    town_mesh_cache: Any,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    output_dir: str | Path,
    config: dict[str, Any],
    seed: int,
) -> list[PatchIndexRecord]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_centers = observed_cache.tile_centers
    debug_max_tiles = config.get("debug_max_tiles_per_sequence")
    if debug_max_tiles is not None:
        tile_centers = tile_centers[: int(debug_max_tiles)]
    patch_radius_m_list = [float(value) for value in config.get("patch_radius_m_list", [config.get("patch_radius_m", 3.0)])]

    sequence_records: list[PatchIndexRecord] = []
    sequence_stats = {
        "town_id": town_id,
        "sequence_id": sequence_id,
        "patch_cache_format_version": 3,
        "patch_radius_m_list": patch_radius_m_list,
        "num_input_tiles": int(len(observed_cache.tile_centers)),
        "num_processed_tiles": int(len(tile_centers)),
        "num_written_patches": 0,
        "patch_ids_preview": [],
    }

    for tile_id, patch_center_world in enumerate(tile_centers):
        for scale_id, patch_radius_m in enumerate(patch_radius_m_list):
            sample = build_patch_from_tile_v3(
                raw_sequence=raw_sequence,
                town_mesh_cache=town_mesh_cache,
                observed_cache=observed_cache,
                town_id=town_id,
                sequence_id=sequence_id,
                tile_id=tile_id,
                scale_id=scale_id,
                patch_center_world=patch_center_world,
                config=config,
                seed=seed,
            )
            if sample is None:
                continue
            patch_path = output_dir / f"{sample.patch_id}.npz"
            sample.save(patch_path)
            sequence_records.append(
                PatchIndexRecord(
                    patch_id=sample.patch_id,
                    town_id=town_id,
                    sequence_id=sequence_id,
                    tile_id=tile_id,
                    patch_file=str(patch_path),
                    num_local_faces=sample.num_local_faces,
                    num_observed_points_raw=sample.num_observed_points_raw,
                    num_clean_points=int(len(sample.clean_points)),
                    num_observed_points=int(len(sample.observed_points)),
                    scale_id=int(scale_id),
                    patch_radius_m=float(patch_radius_m),
                    teacher_area_local=float(sample.teacher_area_local),
                    planarity_hint=float(sample.metadata.get("planarity_hint", 0.0)),
                    patch_cache_format_version=3,
                    num_surface_query_points=int(len(sample.surface_query_points)),
                    num_free_query_points=int(len(sample.free_query_points)),
                    num_unknown_query_points=int(len(sample.unknown_query_points)),
                    camera_support_count=int(sample.camera_support_count),
                    lidar_support_count=int(sample.lidar_support_count),
                    visible_surface_fraction=float(sample.visible_surface_fraction),
                    free_space_fraction=float(sample.free_space_fraction),
                    unknown_fraction=float(sample.unknown_fraction),
                    intrinsic_patch_difficulty_target=float(sample.intrinsic_patch_difficulty_target),
                    num_visible_clean_points=int(len(sample.visible_clean_points)),
                    num_hidden_clean_points=int(len(sample.hidden_clean_points)),
                    visible_support_fraction=float(sample.visible_support_fraction),
                    hidden_surface_fraction=float(sample.hidden_surface_fraction),
                    free_space_hard_negative_count=int(sample.free_space_hard_negative_count),
                    difficulty_components_json=dict(sample.difficulty_components_json),
                )
            )

    sequence_records = sorted(sequence_records, key=lambda record: (record.tile_id, record.scale_id))
    sequence_stats["num_written_patches"] = len(sequence_records)
    sequence_stats["patch_ids_preview"] = [record.patch_id for record in sequence_records[:10]]
    dump_json(output_dir / "sequence_patch_stats.json", sequence_stats, indent=2)
    return sequence_records
