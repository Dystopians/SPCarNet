from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ss3dm_prior.data.scenario_loader import load_scenario


def _write_scenario(sequence_root: Path, payload: dict) -> None:
    sequence_root.mkdir(parents=True, exist_ok=True)
    with (sequence_root / "scenario.pt").open("wb") as handle:
        pickle.dump(payload, handle)
    (sequence_root / "scenario.txt").write_text(str({"scene_id": payload.get("scene_id", "dummy")}))


def _make_scenario_payload(include_c2w: bool = True) -> dict:
    camera_data = {
        "hw": np.asarray([[1080.0, 1920.0], [1080.0, 1920.0]], dtype=np.float32),
        "intr": np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)], axis=0),
    }
    if include_c2w:
        camera_data["c2w"] = np.stack(
            [
                np.eye(4, dtype=np.float32),
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

    lidar_pose = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)], axis=0)
    return {
        "scene_id": "town01_2",
        "metas": {"num_frames": 2},
        "objects": {},
        "observers": {
            "camera_FRONT": {
                "id": "camera_FRONT",
                "class_name": "Camera",
                "num_frames": 2,
                "data": camera_data,
            },
            "lidar_TOP": {
                "id": "lidar_TOP",
                "class_name": "RaysLidar",
                "num_frames": 2,
                "data": {
                    "l2v": lidar_pose,
                    "sensor_v2w": lidar_pose,
                    "timestamp": np.asarray([0, 1], dtype=np.int64),
                },
            },
        },
    }


def test_load_scenario_from_pickle_pt(tmp_path: Path) -> None:
    sequence_root = tmp_path / "Town01" / "2_streetsurf"
    _write_scenario(sequence_root, _make_scenario_payload())

    scenario = load_scenario(sequence_root)

    assert scenario.scene_id == "town01_2"
    assert scenario.num_frames == 2
    assert "camera_FRONT" in scenario.cameras
    assert "lidar_TOP" in scenario.lidars
    assert scenario.cameras["camera_FRONT"].intr is not None
    assert scenario.cameras["camera_FRONT"].c2w is not None
    assert scenario.cameras["camera_FRONT"].center_array().shape == (2, 3)
    assert scenario.lidars["lidar_TOP"].sensor_v2w is not None


def test_load_scenario_warns_on_missing_camera_fields(tmp_path: Path) -> None:
    sequence_root = tmp_path / "Town01" / "2_streetsurf"
    _write_scenario(sequence_root, _make_scenario_payload(include_c2w=False))

    scenario = load_scenario(sequence_root)

    assert any("torch.load unavailable" in warning for warning in scenario.warnings)
    assert any("missing c2w" in warning for warning in scenario.warnings)
    assert scenario.cameras["camera_FRONT"].c2w is None
