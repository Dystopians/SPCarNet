from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.data.patch_types import load_patch_npz
from ss3dm_prior.data.teacher_patch_builder import (
    SequenceObservedCache,
    build_patch_from_tile_v2,
)
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


class _FakeRawSequence:
    def __init__(self, lidar_frame_path: Path) -> None:
        self._lidar_frame_path = lidar_frame_path
        c2w = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 1, axis=0)
        c2w[0, :3, 3] = np.asarray([1.0, 1.0, 4.0], dtype=np.float32)
        self.scenario = SimpleNamespace(
            cameras={"camera_FRONT": SimpleNamespace(c2w=c2w)},
        )

    def iter_frame_indices(self, frame_stride: int) -> list[int]:
        del frame_stride
        return [0]

    def lidar_names(self) -> list[str]:
        return ["lidar_TOP"]

    def camera_names(self) -> list[str]:
        return ["camera_FRONT"]

    def lidar_frame_path(self, lidar_name: str, frame_idx: int) -> Path:
        assert lidar_name == "lidar_TOP"
        assert frame_idx == 0
        return self._lidar_frame_path


def _write_lidar_frame(path: Path) -> None:
    origin = np.asarray([1.0, 1.0, 2.0], dtype=np.float32)
    targets = np.asarray(
        [
            [0.7, 0.8, 0.0],
            [0.9, 1.0, 0.0],
            [1.1, 1.0, 0.0],
            [1.3, 1.2, 0.0],
            [1.0, 0.7, 0.0],
            [1.0, 1.3, 0.0],
        ],
        dtype=np.float32,
    )
    directions = targets - origin[None, :]
    ranges = np.linalg.norm(directions, axis=1).astype(np.float32)
    rays_d = (directions / ranges[:, None]).astype(np.float32)
    rays_o = np.repeat(origin[None, :], len(targets), axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rays_o=rays_o, rays_d=rays_d, ranges=ranges)


def test_build_patch_from_tile_v2_outputs_visibility_queries(tmp_path: Path) -> None:
    obj_path = tmp_path / "TownUnit_obj.obj"
    mesh_cache_dir = tmp_path / "mesh_cache" / "TownUnit"
    lidar_frame_path = tmp_path / "lidars" / "lidar_TOP" / "00000000.npz"
    _write_plane_obj(obj_path)
    _write_lidar_frame(lidar_frame_path)
    convert_obj_to_cache(
        obj_path=obj_path,
        out_dir=mesh_cache_dir,
        town_id="TownUnit",
        conversion_command="unit-test",
    )
    town_mesh_cache = load_town_mesh_cache(mesh_cache_dir, mmap=False)
    observed_cache = SequenceObservedCache(
        observed_points=np.asarray(
            [
                [0.9, 0.9, 0.03],
                [1.0, 1.0, 0.01],
                [1.1, 1.0, 0.02],
                [1.0, 1.1, 0.01],
                [0.8, 1.0, 0.02],
            ],
            dtype=np.float32,
        ),
        tile_centers=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        camera_centers=np.asarray([[1.0, 1.0, 4.0]], dtype=np.float32),
        sequence_stats={},
        cache_path=tmp_path / "observed_cache.npz",
    )
    raw_sequence = _FakeRawSequence(lidar_frame_path)

    sample = build_patch_from_tile_v2(
        raw_sequence=raw_sequence,
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
            "clean_sample_count": 32,
            "observed_sample_count": 8,
            "face_query_margin_m": 0.5,
            "town_mesh_unit_scale": 1.0,
            "visibility_queries": {
                "surface_query_count": 12,
                "free_query_count": 12,
                "unknown_query_count": 8,
                "frame_stride": 1,
                "min_range": 0.1,
                "max_range": 10.0,
                "max_points_per_frame": 32,
                "max_free_rays_per_patch": 12,
                "camera_support_radius_m": 10.0,
            },
        },
        seed=0,
    )

    assert sample is not None
    assert sample.patch_cache_format_version == 2
    assert sample.surface_query_points.shape[0] == 12
    assert sample.free_query_points.shape[0] > 0
    assert sample.unknown_query_points.shape[0] == 8
    assert sample.query_points_all.shape[0] == (
        sample.surface_query_points.shape[0]
        + sample.free_query_points.shape[0]
        + sample.unknown_query_points.shape[0]
    )
    assert sample.query_labels_all.shape[0] == sample.query_points_all.shape[0]
    assert sample.query_ignore_mask.shape[0] == sample.query_points_all.shape[0]
    assert sample.camera_support_count > 0
    assert sample.lidar_support_count > 0
    assert 0.0 <= sample.visible_surface_fraction <= 1.0
    assert 0.0 <= sample.free_space_fraction <= 1.0
    assert 0.0 <= sample.unknown_fraction <= 1.0
    assert 0.0 <= sample.intrinsic_patch_difficulty_target <= 1.0
    assert "observed_to_clean_nn_error" in sample.difficulty_components_json
    patch_path = sample.save(tmp_path / "patch_v2.npz")
    payload = load_patch_npz(patch_path)
    assert int(payload["patch_cache_format_version"].item()) == 2
    assert payload["surface_query_points"].shape[0] == 12


def test_load_patch_npz_backward_compatible_defaults(tmp_path: Path) -> None:
    patch_path = tmp_path / "legacy_patch.npz"
    np.savez_compressed(
        patch_path,
        clean_points=np.zeros((4, 3), dtype=np.float32),
        clean_normals=np.zeros((4, 3), dtype=np.float32),
        observed_points=np.zeros((2, 3), dtype=np.float32),
        patch_center_world=np.zeros((3,), dtype=np.float32),
        patch_radius_m=np.asarray(2.0, dtype=np.float32),
        town_id=np.asarray("TownUnit"),
        sequence_id=np.asarray("TownUnit__seq"),
        tile_id=np.asarray(0, dtype=np.int32),
        patch_id=np.asarray("TownUnit__seq__tile_000000"),
        num_local_faces=np.asarray(1, dtype=np.int32),
        num_observed_points_raw=np.asarray(2, dtype=np.int32),
        teacher_area_local=np.asarray(1.0, dtype=np.float32),
        source_town_mesh_cache_dir=np.asarray("mesh_cache"),
        source_sequence_observed_cache=np.asarray("observed_cache"),
        patch_metadata_json=np.asarray("{}"),
    )

    payload = load_patch_npz(patch_path)

    assert int(payload["patch_cache_format_version"].item()) == 1
    assert payload["surface_query_points"].shape == (0, 3)
    assert payload["free_query_points"].shape == (0, 3)
    assert payload["query_ignore_mask"].shape == (0,)
