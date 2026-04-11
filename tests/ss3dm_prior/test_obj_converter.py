from __future__ import annotations

from pathlib import Path

import numpy as np

from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.data.town_mesh_cache import load_town_mesh_cache


def _write_toy_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_convert_obj_to_cache_and_query(tmp_path: Path) -> None:
    obj_path = tmp_path / "TownUnit_obj.obj"
    out_dir = tmp_path / "cache" / "TownUnit"
    _write_toy_obj(obj_path)

    converted = convert_obj_to_cache(
        obj_path=obj_path,
        out_dir=out_dir,
        town_id="TownUnit",
        conversion_command="unit-test",
    )

    assert converted.vertices.shape == (4, 3)
    assert converted.faces.shape == (2, 3)
    assert converted.face_centroids.shape == (2, 3)
    assert converted.face_normals.shape == (2, 3)
    assert converted.face_areas.shape == (2,)

    cache = load_town_mesh_cache(out_dir, mmap=True)
    assert isinstance(cache.vertices, np.memmap)
    mask = cache.query_faces_in_radius(center=[0.3, 0.3, 0.1], radius=0.5)
    assert mask.shape == (2,)
    assert int(mask.sum()) >= 1

    local_mesh = cache.build_local_mesh_from_face_mask(mask)
    assert local_mesh["vertices"].shape[1] == 3
    assert local_mesh["faces"].shape[1] == 3
    assert len(local_mesh["face_indices"]) == int(mask.sum())
