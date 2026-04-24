from __future__ import annotations

import torch

from ss3dm_prior.losses import compute_hard_example_sample_weights


def test_hard_weight_source_switches_between_corruption_and_intrinsic() -> None:
    batch = {
        "clean_points": torch.zeros((2, 4, 3), dtype=torch.float32),
        "corruption_score_target": torch.tensor([0.0, 9.0], dtype=torch.float32),
        "intrinsic_patch_difficulty_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }

    corruption_weights = compute_hard_example_sample_weights(
        batch,
        {"hard_example_reweight": 1.0, "hard_weight_source": "corruption"},
    )
    intrinsic_weights = compute_hard_example_sample_weights(
        batch,
        {"hard_example_reweight": 1.0, "hard_weight_source": "intrinsic"},
    )

    assert corruption_weights.shape == (2,)
    assert intrinsic_weights.shape == (2,)
    assert corruption_weights[1] > corruption_weights[0]
    assert intrinsic_weights[0] > intrinsic_weights[1]


def test_hard_weight_blend_alpha_combines_corruption_and_intrinsic() -> None:
    batch = {
        "clean_points": torch.zeros((2, 4, 3), dtype=torch.float32),
        "corruption_score_target": torch.tensor([0.0, 9.0], dtype=torch.float32),
        "intrinsic_patch_difficulty_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }

    blended_weights = compute_hard_example_sample_weights(
        batch,
        {
            "hard_example_reweight": 1.0,
            "hard_weight_source": "blend",
            "hard_weight_blend_alpha": 0.75,
        },
    )

    expected = torch.tensor([1.25, 0.75], dtype=torch.float32)
    assert torch.allclose(blended_weights, expected, atol=1e-5)
