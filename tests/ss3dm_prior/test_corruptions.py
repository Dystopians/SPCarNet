from __future__ import annotations

import numpy as np

from ss3dm_prior.data.corruptions import apply_patch_corruptions


def test_apply_patch_corruptions_outputs_targets() -> None:
    clean_points = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    observed_points = clean_points.copy()

    result = apply_patch_corruptions(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        config={
            "target_corrupted_count": 8,
            "point_dropout": {"enabled": True, "dropout_ratio": 0.25},
            "gaussian_jitter": {"enabled": True, "sigma": 0.05, "anisotropic": False},
            "normal_noise": {"enabled": True, "sigma": 0.1, "flip_prob": 0.0},
            "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.15},
            "outlier_cluster": {"enabled": True, "cluster_size": 2, "cluster_offset_sigma": 0.1, "cluster_spread_sigma": 0.01},
            "density_imbalance": {"enabled": True, "region_radius": 0.2, "thin_probability": 1.0, "thin_ratio": 0.5, "duplicate_count": 2},
        },
        seed=0,
        sample_key="unit_patch",
    )

    assert result.corrupted_points.shape == (8, 3)
    assert result.corrupted_normals.shape == (8, 3)
    assert result.point_defect_target.shape == (8,)
    assert float(result.corruption_score_target) > 0.0
    assert np.all(np.isfinite(result.point_defect_target))
    assert "enabled_corruptions" in result.metadata
