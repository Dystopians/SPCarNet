from __future__ import annotations

import numpy as np

from ss3dm_prior.data.corruptions import resolve_corruption_severity_scale
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset


def test_resolve_corruption_severity_scale_linear() -> None:
    config = {
        "corruptions": {
            "severity_schedule": {
                "type": "linear",
                "start_scale": 0.2,
                "end_scale": 1.0,
                "warmup_epochs": 4,
            }
        }
    }

    assert np.isclose(resolve_corruption_severity_scale(config, epoch=0), 0.2)
    assert np.isclose(resolve_corruption_severity_scale(config, epoch=2), 0.6)
    assert np.isclose(resolve_corruption_severity_scale(config, epoch=4), 1.0)
    assert np.isclose(resolve_corruption_severity_scale(config, epoch=10), 1.0)


def test_resolve_corruption_severity_scale_cosine_and_dataset_epoch() -> None:
    config = {
        "corruptions": {
            "severity_schedule": {
                "type": "cosine",
                "start_scale": 0.5,
                "end_scale": 1.0,
                "warmup_epochs": 4,
            }
        }
    }

    dataset = TeacherPatchTrainDataset(
        patch_index_path="/tmp/nonexistent_patch_index.jsonl",
        records=[],
        corruption_config=config,
        dynamic_corruption=True,
    )
    dataset.set_epoch(0)
    start_scale = dataset.current_corruption_severity_scale()
    dataset.set_epoch(2)
    mid_scale = dataset.current_corruption_severity_scale()
    dataset.set_epoch(4)
    end_scale = dataset.current_corruption_severity_scale()

    assert np.isclose(start_scale, 0.5)
    assert 0.5 < mid_scale < 1.0
    assert np.isclose(end_scale, 1.0)
