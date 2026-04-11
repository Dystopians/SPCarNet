from __future__ import annotations

from pathlib import Path

import numpy as np

from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.data.teacher_patch_builder import (
    SequenceObservedCache,
    build_patch_from_tile,
    build_teacher_patches_for_sequence,
)
from ss3dm_prior.tools.build_teacher_patch_cache import _collect_completed_patch_records
from ss3dm_prior.data.town_mesh_cache import load_town_mesh_cache


def _write_plane_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 2 0 0",
                "v 0 2 0",
                "v 2 2 0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_teacher_patch_builder_outputs_local_patch(tmp_path: Path) -> None:
    obj_path = tmp_path / "TownUnit_obj.obj"
    mesh_cache_dir = tmp_path / "mesh_cache" / "TownUnit"
    _write_plane_obj(obj_path)
    convert_obj_to_cache(
        obj_path=obj_path,
        out_dir=mesh_cache_dir,
        town_id="TownUnit",
        conversion_command="unit-test",
    )
    town_mesh_cache = load_town_mesh_cache(mesh_cache_dir, mmap=False)

    observed_points = np.asarray(
        [
            [0.9, 0.9, 0.05],
            [1.0, 1.0, 0.0],
            [1.1, 0.95, 0.02],
            [1.2, 1.05, 0.01],
            [0.8, 1.1, 0.0],
        ],
        dtype=np.float32,
    )
    observed_cache = SequenceObservedCache(
        observed_points=observed_points,
        tile_centers=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        camera_centers=np.zeros((0, 3), dtype=np.float32),
        sequence_stats={},
        cache_path=tmp_path / "observed_cache.npz",
    )

    sample = build_patch_from_tile(
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        tile_id=0,
        patch_center_world=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        config={
            "patch_radius_m": 2.0,
            "observed_min_points": 4,
            "clean_min_faces": 1,
            "clean_sample_count": 16,
            "observed_sample_count": 8,
            "face_query_margin_m": 0.5,
        },
        seed=0,
    )

    assert sample is not None
    assert sample.clean_points.shape == (16, 3)
    assert sample.clean_normals.shape == (16, 3)
    assert sample.observed_points.shape == (8, 3)
    assert np.max(np.linalg.norm(sample.observed_points, axis=1)) <= 1.0
    assert sample.num_local_faces >= 1
    assert sample.teacher_area_local > 0.0


def test_build_teacher_patches_for_sequence_writes_patch(tmp_path: Path) -> None:
    obj_path = tmp_path / "TownUnit_obj.obj"
    mesh_cache_dir = tmp_path / "mesh_cache" / "TownUnit"
    out_dir = tmp_path / "patch_cache" / "TownUnit" / "TownUnit__seq"
    _write_plane_obj(obj_path)
    convert_obj_to_cache(
        obj_path=obj_path,
        out_dir=mesh_cache_dir,
        town_id="TownUnit",
        conversion_command="unit-test",
    )
    town_mesh_cache = load_town_mesh_cache(mesh_cache_dir, mmap=False)

    observed_cache = SequenceObservedCache(
        observed_points=np.asarray(
            [[1.0, 1.0, 0.0], [0.9, 1.0, 0.0], [1.1, 1.0, 0.0], [1.0, 0.9, 0.0]],
            dtype=np.float32,
        ),
        tile_centers=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        camera_centers=np.zeros((0, 3), dtype=np.float32),
        sequence_stats={},
        cache_path=tmp_path / "observed_cache.npz",
    )

    records = build_teacher_patches_for_sequence(
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        output_dir=out_dir,
        config={
            "patch_radius_m": 2.0,
            "observed_min_points": 4,
            "clean_min_faces": 1,
            "clean_sample_count": 8,
            "observed_sample_count": 4,
            "face_query_margin_m": 0.5,
            "debug_max_tiles_per_sequence": 1,
        },
        seed=0,
    )

    assert len(records) == 1
    assert Path(records[0].patch_file).exists()
    assert (out_dir / "sequence_patch_stats.json").exists()


def test_collect_completed_patch_records_reads_finished_sequence(tmp_path: Path) -> None:
    obj_path = tmp_path / "TownUnit_obj.obj"
    mesh_cache_dir = tmp_path / "mesh_cache" / "TownUnit"
    out_dir = tmp_path / "patch_cache" / "TownUnit" / "TownUnit__seq"
    _write_plane_obj(obj_path)
    convert_obj_to_cache(
        obj_path=obj_path,
        out_dir=mesh_cache_dir,
        town_id="TownUnit",
        conversion_command="unit-test",
    )
    town_mesh_cache = load_town_mesh_cache(mesh_cache_dir, mmap=False)
    observed_cache = SequenceObservedCache(
        observed_points=np.asarray(
            [[1.0, 1.0, 0.0], [0.9, 1.0, 0.0], [1.1, 1.0, 0.0], [1.0, 0.9, 0.0]],
            dtype=np.float32,
        ),
        tile_centers=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        camera_centers=np.zeros((0, 3), dtype=np.float32),
        sequence_stats={},
        cache_path=tmp_path / "observed_cache.npz",
    )
    build_teacher_patches_for_sequence(
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        output_dir=out_dir,
        config={
            "patch_radius_m": 2.0,
            "observed_min_points": 4,
            "clean_min_faces": 1,
            "clean_sample_count": 8,
            "observed_sample_count": 4,
            "face_query_margin_m": 0.5,
            "debug_max_tiles_per_sequence": 1,
        },
        seed=0,
    )

    records = _collect_completed_patch_records(tmp_path / "patch_cache")
    assert len(records) == 1
    assert records[0].patch_id == "TownUnit__seq__tile_000000"
