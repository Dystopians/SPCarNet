"""Build clean teacher and observed local patch caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import trimesh

from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.data.town_mesh_cache import TownMeshCache
from ss3dm_prior.data.visibility_queries import build_patch_visibility_queries
from ss3dm_prior.utils.io import dump_json


def _patch_rng(seed: int, patch_id: str) -> np.random.Generator:
    derived_seed = (int(seed) + zlib.crc32(patch_id.encode("utf-8"))) % (2**32)
    return np.random.default_rng(derived_seed)


def _normalize_points(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return ((points - center[None, :]) / float(radius)).astype(np.float32)


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    return (normals / norm).astype(np.float32)


def _sample_indices(rng: np.random.Generator, size: int, sample_count: int) -> np.ndarray:
    replace = size < sample_count
    return rng.choice(size, size=sample_count, replace=replace)


def _mean_normal(normals: np.ndarray) -> list[float]:
    if len(normals) == 0:
        return [0.0, 0.0, 0.0]
    mean_vec = np.mean(normals.astype(np.float32), axis=0, keepdims=True)
    return _normalize_normals(mean_vec)[0].astype(float).tolist()


def _bbox_dict(points: np.ndarray) -> dict[str, list[float]]:
    if len(points) == 0:
        zero = [0.0, 0.0, 0.0]
        return {"min": zero, "max": zero}
    return {
        "min": points.min(axis=0).astype(float).tolist(),
        "max": points.max(axis=0).astype(float).tolist(),
    }


def _planarity_hint(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    denom = float(np.sum(eigvals))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - eigvals[0] / denom)


def _build_patch_metadata(
    *,
    clean_points_local: np.ndarray,
    clean_normals_local: np.ndarray,
    observed_points_local: np.ndarray,
    local_face_areas: np.ndarray,
    patch_center_world: np.ndarray,
    teacher_area_local: float,
) -> dict[str, Any]:
    return {
        "num_clean_points": int(len(clean_points_local)),
        "num_observed_points": int(len(observed_points_local)),
        "clean_bbox_local": _bbox_dict(clean_points_local),
        "observed_bbox_local": _bbox_dict(observed_points_local),
        "approx_area_world": float(np.sum(local_face_areas, dtype=np.float64)),
        "teacher_area_local": teacher_area_local,
        "mean_normal": _mean_normal(clean_normals_local),
        "planarity_hint": _planarity_hint(clean_points_local),
        "patch_center_world": patch_center_world.astype(float).tolist(),
    }


@dataclass
class SequenceObservedCache:
    observed_points: np.ndarray
    tile_centers: np.ndarray
    camera_centers: np.ndarray
    sequence_stats: dict[str, Any]
    cache_path: Path


def load_sequence_observed_cache(observed_cache_path: str | Path) -> SequenceObservedCache:
    observed_cache_path = Path(observed_cache_path).expanduser().resolve()
    with np.load(observed_cache_path, mmap_mode="r") as payload:
        observed_points = np.asarray(payload["observed_points"], dtype=np.float32)
        tile_centers = np.asarray(payload["tile_centers"], dtype=np.float32)
        camera_centers = np.asarray(payload["camera_centers"], dtype=np.float32)
        sequence_stats = json.loads(str(payload["sequence_stats_json"].item()))
    return SequenceObservedCache(
        observed_points=observed_points,
        tile_centers=tile_centers,
        camera_centers=camera_centers,
        sequence_stats=sequence_stats,
        cache_path=observed_cache_path,
    )


def build_patch_from_tile(
    *,
    town_mesh_cache: TownMeshCache,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    tile_id: int,
    patch_center_world: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> TeacherPatchSample | None:
    patch_radius_m = float(config["patch_radius_m"])
    observed_min_points = int(config["observed_min_points"])
    clean_min_faces = int(config["clean_min_faces"])
    clean_sample_count = int(config["clean_sample_count"])
    observed_sample_count = int(config["observed_sample_count"])
    face_query_margin_m = float(config["face_query_margin_m"])
    town_mesh_unit_scale = float(config.get("town_mesh_unit_scale", 1.0))

    patch_center_world = np.asarray(patch_center_world, dtype=np.float32).reshape(3)
    patch_id = f"{sequence_id}__tile_{tile_id:06d}"
    rng = _patch_rng(seed, patch_id)

    observed_dist_sq = np.sum(
        (observed_cache.observed_points - patch_center_world[None, :]) ** 2,
        axis=1,
    )
    observed_mask = observed_dist_sq <= patch_radius_m**2
    observed_points_world = observed_cache.observed_points[observed_mask]
    num_observed_points_raw = int(len(observed_points_world))
    if num_observed_points_raw < observed_min_points:
        return None

    observed_indices = _sample_indices(rng, num_observed_points_raw, observed_sample_count)
    observed_points_local = _normalize_points(
        observed_points_world[observed_indices],
        patch_center_world,
        patch_radius_m,
    )

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
    num_local_faces = int(len(local_mesh_dict["faces"]))
    if num_local_faces < clean_min_faces:
        return None

    local_vertices_world = np.asarray(local_mesh_dict["vertices"], dtype=np.float32)
    local_faces = np.asarray(local_mesh_dict["faces"], dtype=np.int64)
    local_mesh = trimesh.Trimesh(vertices=local_vertices_world, faces=local_faces, process=False)
    if len(local_mesh.faces) == 0 or float(local_mesh.area) <= 0.0:
        return None

    clean_points_world, sampled_face_indices = trimesh.sample.sample_surface(
        local_mesh,
        clean_sample_count,
        seed=rng,
    )
    sampled_face_indices = np.asarray(sampled_face_indices, dtype=np.int64)
    clean_normals_world = np.asarray(local_mesh.face_normals[sampled_face_indices], dtype=np.float32)
    clean_points_local = _normalize_points(
        np.asarray(clean_points_world, dtype=np.float32),
        patch_center_world,
        patch_radius_m,
    )
    clean_normals_local = _normalize_normals(clean_normals_world)

    teacher_area_local = float(np.sum(local_mesh_dict["face_areas"], dtype=np.float64) / (patch_radius_m**2))
    metadata = _build_patch_metadata(
        clean_points_local=clean_points_local,
        clean_normals_local=clean_normals_local,
        observed_points_local=observed_points_local,
        local_face_areas=np.asarray(local_mesh_dict["face_areas"], dtype=np.float32),
        patch_center_world=patch_center_world,
        teacher_area_local=teacher_area_local,
    )

    return TeacherPatchSample(
        clean_points=clean_points_local,
        clean_normals=clean_normals_local,
        observed_points=observed_points_local,
        patch_center_world=patch_center_world.astype(np.float32),
        patch_radius_m=patch_radius_m,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_id=patch_id,
        num_local_faces=num_local_faces,
        num_observed_points_raw=num_observed_points_raw,
        teacher_area_local=teacher_area_local,
        source_town_mesh_cache_dir=str(town_mesh_cache.cache_dir),
        source_sequence_observed_cache=str(observed_cache.cache_path),
        metadata=metadata,
        scale_id=0,
    )


def build_patch_from_tile_v2(
    *,
    raw_sequence: Any,
    town_mesh_cache: TownMeshCache,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    tile_id: int,
    patch_center_world: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> TeacherPatchSample | None:
    sample = build_patch_from_tile(
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_center_world=patch_center_world,
        config=config,
        seed=seed,
    )
    if sample is None:
        return None

    patch_radius_m = float(config["patch_radius_m"])
    face_query_margin_m = float(config["face_query_margin_m"])
    town_mesh_unit_scale = float(config.get("town_mesh_unit_scale", 1.0))
    query_config = dict(config.get("visibility_queries", {}))
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
        clean_points_local=np.asarray(sample.clean_points, dtype=np.float32),
        clean_normals_local=np.asarray(sample.clean_normals, dtype=np.float32),
        observed_points_local=np.asarray(sample.observed_points, dtype=np.float32),
        planarity_hint=float(sample.metadata.get("planarity_hint", 0.0)),
        query_config=query_config,
        seed=seed,
        patch_id=sample.patch_id,
    )
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "patch_cache_format_version": 2,
            "num_surface_query_points": int(len(visibility_bundle.surface_query_points)),
            "num_free_query_points": int(len(visibility_bundle.free_query_points)),
            "num_unknown_query_points": int(len(visibility_bundle.unknown_query_points)),
            "camera_support_count": int(visibility_bundle.camera_support_count),
            "lidar_support_count": int(visibility_bundle.lidar_support_count),
            "visible_surface_fraction": float(visibility_bundle.visible_surface_fraction),
            "free_space_fraction": float(visibility_bundle.free_space_fraction),
            "unknown_fraction": float(visibility_bundle.unknown_fraction),
            "intrinsic_patch_difficulty_target": float(
                visibility_bundle.intrinsic_patch_difficulty_target
            ),
        }
    )
    return TeacherPatchSample(
        clean_points=sample.clean_points,
        clean_normals=sample.clean_normals,
        observed_points=sample.observed_points,
        patch_center_world=sample.patch_center_world,
        patch_radius_m=sample.patch_radius_m,
        town_id=sample.town_id,
        sequence_id=sample.sequence_id,
        tile_id=sample.tile_id,
        patch_id=sample.patch_id,
        num_local_faces=sample.num_local_faces,
        num_observed_points_raw=sample.num_observed_points_raw,
        teacher_area_local=sample.teacher_area_local,
        source_town_mesh_cache_dir=sample.source_town_mesh_cache_dir,
        source_sequence_observed_cache=sample.source_sequence_observed_cache,
        patch_cache_format_version=2,
        surface_query_points=visibility_bundle.surface_query_points,
        surface_query_labels=visibility_bundle.surface_query_labels,
        free_query_points=visibility_bundle.free_query_points,
        free_query_labels=visibility_bundle.free_query_labels,
        unknown_query_points=visibility_bundle.unknown_query_points,
        query_points_all=visibility_bundle.query_points_all,
        query_labels_all=visibility_bundle.query_labels_all,
        query_ignore_mask=visibility_bundle.query_ignore_mask,
        camera_support_count=visibility_bundle.camera_support_count,
        lidar_support_count=visibility_bundle.lidar_support_count,
        visible_surface_fraction=visibility_bundle.visible_surface_fraction,
        free_space_fraction=visibility_bundle.free_space_fraction,
        unknown_fraction=visibility_bundle.unknown_fraction,
        intrinsic_patch_difficulty_target=visibility_bundle.intrinsic_patch_difficulty_target,
        difficulty_components_json=visibility_bundle.difficulty_components_json,
        metadata=metadata,
        scale_id=0,
    )


def build_teacher_patches_for_sequence(
    *,
    town_mesh_cache: TownMeshCache,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    output_dir: str | Path,
    config: dict[str, Any],
    seed: int,
) -> list[PatchIndexRecord]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_max_tiles = config.get("debug_max_tiles_per_sequence")
    tile_centers = observed_cache.tile_centers
    if debug_max_tiles is not None:
        tile_centers = tile_centers[: int(debug_max_tiles)]

    sequence_records: list[PatchIndexRecord] = []
    sequence_stats = {
        "town_id": town_id,
        "sequence_id": sequence_id,
        "num_input_tiles": int(len(observed_cache.tile_centers)),
        "num_processed_tiles": int(len(tile_centers)),
        "num_written_patches": 0,
        "patch_ids_preview": [],
    }

    for tile_id, patch_center_world in enumerate(tile_centers):
        sample = build_patch_from_tile(
            town_mesh_cache=town_mesh_cache,
            observed_cache=observed_cache,
            town_id=town_id,
            sequence_id=sequence_id,
            tile_id=tile_id,
            patch_center_world=patch_center_world,
            config=config,
            seed=seed,
        )
        if sample is None:
            continue

        patch_path = output_dir / f"{sample.patch_id}.npz"
        sample.save(patch_path)
        record = PatchIndexRecord(
            patch_id=sample.patch_id,
            town_id=town_id,
            sequence_id=sequence_id,
            tile_id=tile_id,
            patch_file=str(patch_path),
            num_local_faces=sample.num_local_faces,
            num_observed_points_raw=sample.num_observed_points_raw,
            num_clean_points=int(len(sample.clean_points)),
            num_observed_points=int(len(sample.observed_points)),
            scale_id=0,
            patch_radius_m=float(sample.patch_radius_m),
            teacher_area_local=float(sample.teacher_area_local),
            planarity_hint=float(sample.metadata.get("planarity_hint", 0.0)),
        )
        sequence_records.append(record)

    sequence_stats["num_written_patches"] = len(sequence_records)
    sequence_stats["patch_ids_preview"] = [record.patch_id for record in sequence_records[:10]]
    dump_json(output_dir / "sequence_patch_stats.json", sequence_stats, indent=2)
    return sequence_records


def build_teacher_patches_for_sequence_v2(
    *,
    raw_sequence: Any,
    town_mesh_cache: TownMeshCache,
    observed_cache: SequenceObservedCache,
    town_id: str,
    sequence_id: str,
    output_dir: str | Path,
    config: dict[str, Any],
    seed: int,
) -> list[PatchIndexRecord]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_max_tiles = config.get("debug_max_tiles_per_sequence")
    tile_centers = observed_cache.tile_centers
    if debug_max_tiles is not None:
        tile_centers = tile_centers[: int(debug_max_tiles)]

    sequence_records: list[PatchIndexRecord] = []
    sequence_stats = {
        "town_id": town_id,
        "sequence_id": sequence_id,
        "patch_cache_format_version": 2,
        "num_input_tiles": int(len(observed_cache.tile_centers)),
        "num_processed_tiles": int(len(tile_centers)),
        "num_written_patches": 0,
        "patch_ids_preview": [],
    }

    for tile_id, patch_center_world in enumerate(tile_centers):
        sample = build_patch_from_tile_v2(
            raw_sequence=raw_sequence,
            town_mesh_cache=town_mesh_cache,
            observed_cache=observed_cache,
            town_id=town_id,
            sequence_id=sequence_id,
            tile_id=tile_id,
            patch_center_world=patch_center_world,
            config=config,
            seed=seed,
        )
        if sample is None:
            continue

        patch_path = output_dir / f"{sample.patch_id}.npz"
        sample.save(patch_path)
        record = PatchIndexRecord(
            patch_id=sample.patch_id,
            town_id=town_id,
            sequence_id=sequence_id,
            tile_id=tile_id,
            patch_file=str(patch_path),
            num_local_faces=sample.num_local_faces,
            num_observed_points_raw=sample.num_observed_points_raw,
            num_clean_points=int(len(sample.clean_points)),
            num_observed_points=int(len(sample.observed_points)),
            scale_id=0,
            patch_radius_m=float(sample.patch_radius_m),
            teacher_area_local=float(sample.teacher_area_local),
            planarity_hint=float(sample.metadata.get("planarity_hint", 0.0)),
            patch_cache_format_version=2,
            num_surface_query_points=int(len(sample.surface_query_points)),
            num_free_query_points=int(len(sample.free_query_points)),
            num_unknown_query_points=int(len(sample.unknown_query_points)),
            camera_support_count=int(sample.camera_support_count),
            lidar_support_count=int(sample.lidar_support_count),
            visible_surface_fraction=float(sample.visible_surface_fraction),
            free_space_fraction=float(sample.free_space_fraction),
            unknown_fraction=float(sample.unknown_fraction),
            intrinsic_patch_difficulty_target=float(sample.intrinsic_patch_difficulty_target),
            difficulty_components_json=dict(sample.difficulty_components_json),
        )
        sequence_records.append(record)

    sequence_stats["num_written_patches"] = len(sequence_records)
    sequence_stats["patch_ids_preview"] = [record.patch_id for record in sequence_records[:10]]
    dump_json(output_dir / "sequence_patch_stats.json", sequence_stats, indent=2)
    return sequence_records
