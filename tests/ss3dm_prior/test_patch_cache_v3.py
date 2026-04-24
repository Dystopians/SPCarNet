from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.data.patch_types import load_patch_npz
from ss3dm_prior.data.teacher_patch_builder import SequenceObservedCache
from ss3dm_prior.data.teacher_patch_builder_v3 import (
    build_patch_from_tile_v3,
    build_teacher_patches_for_sequence_v3,
    make_multiscale_patch_id,
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
        c2w = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 1, axis=0)
        c2w[0, :3, 3] = np.asarray([1.0, 1.0, 4.0], dtype=np.float32)
        self._lidar_frame_path = lidar_frame_path
        self.scenario = SimpleNamespace(cameras={"camera_FRONT": SimpleNamespace(c2w=c2w)})

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


def _make_fixture(tmp_path: Path):
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
    config = {
        "patch_radius_m_list": [2.0, 4.0],
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
            "visible_clean_support_radius_local": 0.10,
            "free_space_hard_negative_radius_local": 0.20,
        },
        "debug_max_tiles_per_sequence": 1,
    }
    return town_mesh_cache, observed_cache, raw_sequence, config


def test_make_multiscale_patch_id_is_deterministic() -> None:
    patch_id = make_multiscale_patch_id("TownUnit__seq", 7, 2, 6.0)
    assert patch_id == "TownUnit__seq__tile_000007__scale_02__r6p00m"


def test_build_patch_from_tile_v3_outputs_semantic_fields(tmp_path: Path) -> None:
    town_mesh_cache, observed_cache, raw_sequence, config = _make_fixture(tmp_path)
    sample = build_patch_from_tile_v3(
        raw_sequence=raw_sequence,
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        tile_id=0,
        scale_id=0,
        patch_center_world=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        config=config,
        seed=0,
    )

    assert sample is not None
    assert sample.patch_cache_format_version == 3
    assert sample.scale_id == 0
    assert sample.patch_id == "TownUnit__seq__tile_000000__scale_00__r2p00m"
    assert sample.visible_clean_points.shape[1] == 3
    assert sample.hidden_clean_points.shape[1] == 3
    assert sample.surface_support_mask.shape[0] == sample.clean_points.shape[0]
    assert 0.0 <= sample.visible_support_fraction <= 1.0
    assert 0.0 <= sample.hidden_surface_fraction <= 1.0
    assert sample.free_space_hard_negative_count == sample.free_space_query_hard_negatives.shape[0]

    patch_path = sample.save(tmp_path / "patch_v3.npz")
    payload = load_patch_npz(patch_path)
    assert int(payload["patch_cache_format_version"].item()) == 3
    assert int(payload["scale_id"].item()) == 0
    assert payload["visible_clean_points"].shape[1] == 3
    assert payload["hidden_clean_points"].shape[1] == 3


def test_build_teacher_patches_for_sequence_v3_writes_multiscale_records(tmp_path: Path) -> None:
    town_mesh_cache, observed_cache, raw_sequence, config = _make_fixture(tmp_path)
    out_dir = tmp_path / "patch_cache" / "TownUnit" / "TownUnit__seq"
    records = build_teacher_patches_for_sequence_v3(
        raw_sequence=raw_sequence,
        town_mesh_cache=town_mesh_cache,
        observed_cache=observed_cache,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        output_dir=out_dir,
        config=config,
        seed=0,
    )

    assert len(records) == 2
    assert records[0].scale_id == 0
    assert records[1].scale_id == 1
    assert records[0].patch_radius_m == 2.0
    assert records[1].patch_radius_m == 4.0
    assert Path(records[0].patch_file).exists()
    assert Path(records[1].patch_file).exists()
    assert records[0].patch_id.endswith("__scale_00__r2p00m")
    assert records[1].patch_id.endswith("__scale_01__r4p00m")
