from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser


def _count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _make_random_batch(batch_size: int = 1, num_points: int = 96, num_queries: int = 80) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(0)
    clean_points = torch.randn(batch_size, num_points, 3, generator=rng) * 0.1
    clean_normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3, generator=rng), dim=-1)
    observed_points = clean_points[:, : num_points // 2]
    corrupted_points = clean_points + 0.01 * torch.randn(batch_size, num_points, 3, generator=rng)
    corrupted_normals = torch.nn.functional.normalize(
        clean_normals + 0.01 * torch.randn(batch_size, num_points, 3, generator=rng),
        dim=-1,
    )
    visible_clean_points = clean_points[:, : num_points // 2]
    visible_clean_normals = clean_normals[:, : num_points // 2]
    hidden_clean_points = clean_points[:, num_points // 2 :]
    hidden_clean_normals = clean_normals[:, num_points // 2 :]
    query_points_all = torch.randn(batch_size, num_queries, 3, generator=rng) * 0.5
    return {
        "clean_points": clean_points,
        "clean_normals": clean_normals,
        "observed_points": observed_points,
        "corrupted_points": corrupted_points,
        "corrupted_normals": corrupted_normals,
        "visible_clean_points": visible_clean_points,
        "visible_clean_normals": visible_clean_normals,
        "hidden_clean_points": hidden_clean_points,
        "hidden_clean_normals": hidden_clean_normals,
        "query_points_all": query_points_all,
    }


def test_model_v10_crossattn_random_forward() -> None:
    batch = _make_random_batch()
    model = LocalPatchDenoiser(model_type="v10_cross_attention_hybrid")
    wide_model = LocalPatchDenoiser(model_type="hybrid_v2_wide")
    outputs = model(
        corrupted_points=batch["corrupted_points"].float(),
        corrupted_normals=batch["corrupted_normals"].float(),
        observed_points=batch["observed_points"].float(),
        clean_points=batch["clean_points"].float(),
        clean_normals=batch["clean_normals"].float(),
        query_points_all=batch["query_points_all"].float(),
        visible_clean_points=batch["visible_clean_points"].float(),
        visible_clean_normals=batch["visible_clean_normals"].float(),
        hidden_clean_points=batch["hidden_clean_points"].float(),
        hidden_clean_normals=batch["hidden_clean_normals"].float(),
    )

    assert outputs["recon_points"].shape == (1, 3072, 3)
    assert outputs["recon_normals"].shape == (1, 3072, 3)
    assert outputs["query_occupancy_logits"].shape == (1, 80)
    assert outputs["intrinsic_difficulty_pred"].shape == (1,)
    assert outputs["retrieval_embedding"].shape == (1, 256)
    assert outputs["code_indices"].shape == (1,)
    assert _count_params(model) > _count_params(wide_model)


def test_model_v10_crossattn_forward_on_real_v3_patch_batch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    patch_index_path = repo_root / "outputs" / "ss3dm_prior" / "teacher_patch_cache_v3_debug" / "patch_index.jsonl"
    if not patch_index_path.exists():
        pytest.skip("real v3 patch cache debug output not available")

    dataset = TeacherPatchTrainDataset(
        patch_index_path=patch_index_path,
        records=None,
        split_config=None,
        corruption_config={"target_corrupted_count": 2048},
        seed=0,
        dynamic_corruption=False,
    )
    assert len(dataset) > 0
    sample = dataset[0]
    model = LocalPatchDenoiser(model_type="v10_cross_attention_hybrid")
    outputs = model(
        corrupted_points=sample["corrupted_points"].unsqueeze(0).float(),
        corrupted_normals=sample["corrupted_normals"].unsqueeze(0).float(),
        observed_points=sample["observed_points"].unsqueeze(0).float(),
        clean_points=sample["clean_points"].unsqueeze(0).float(),
        clean_normals=sample["clean_normals"].unsqueeze(0).float(),
        query_points_all=sample["query_points_all"].unsqueeze(0).float(),
        visible_clean_points=sample["visible_clean_points"].unsqueeze(0).float(),
        visible_clean_normals=sample["visible_clean_normals"].unsqueeze(0).float(),
        hidden_clean_points=sample["hidden_clean_points"].unsqueeze(0).float(),
        hidden_clean_normals=sample["hidden_clean_normals"].unsqueeze(0).float(),
    )

    assert int(sample["patch_cache_format_version"]) == 3
    assert outputs["recon_points"].shape == (1, 3072, 3)
    assert outputs["query_occupancy_logits"] is not None
    assert outputs["query_occupancy_logits"].shape[1] == int(sample["query_points_all"].shape[0])
    assert outputs["intrinsic_difficulty_pred"] is not None
