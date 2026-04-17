from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.losses import compute_patch_losses
from ss3dm_prior.metrics import (
    free_space_violation_rate,
    intrinsic_difficulty_mae,
    intrinsic_difficulty_spearman,
    occupancy_iou_visible,
    prototype_usage_entropy,
)
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser


def _corruption_config() -> dict:
    return {
        "target_corrupted_count": 32,
        "point_dropout": {"enabled": True, "dropout_ratio": 0.1},
        "gaussian_jitter": {"enabled": True, "sigma": 0.01, "anisotropic": False},
        "normal_noise": {"enabled": True, "sigma": 0.05, "flip_prob": 0.0},
        "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.1},
        "outlier_cluster": {
            "enabled": True,
            "cluster_size": 4,
            "cluster_offset_sigma": 0.1,
            "cluster_spread_sigma": 0.01,
        },
        "density_imbalance": {
            "enabled": True,
            "region_radius": 0.15,
            "thin_probability": 0.5,
            "thin_ratio": 0.5,
            "duplicate_count": 4,
        },
    }


def _make_random_batch(batch_size: int = 2, num_points: int = 32, num_queries: int = 48) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(0)
    clean_points = torch.randn(batch_size, num_points, 3, generator=rng) * 0.1
    clean_normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3, generator=rng), dim=-1)
    observed_points = clean_points[:, : num_points // 2]
    corrupted_points = clean_points + 0.01 * torch.randn(batch_size, num_points, 3, generator=rng)
    corrupted_normals = torch.nn.functional.normalize(
        clean_normals + 0.01 * torch.randn(batch_size, num_points, 3, generator=rng),
        dim=-1,
    )
    point_defect_target = torch.abs(torch.randn(batch_size, num_points, generator=rng)) * 0.1
    corruption_score_target = torch.rand(batch_size, generator=rng) * 0.5
    intrinsic_target = torch.rand(batch_size, generator=rng)
    query_points_all = torch.randn(batch_size, num_queries, 3, generator=rng) * 0.5
    query_labels_all = torch.randint(0, 2, (batch_size, num_queries), generator=rng)
    query_ignore_mask = torch.zeros(batch_size, num_queries, dtype=torch.bool)
    query_ignore_mask[:, -8:] = True
    return {
        "clean_points": clean_points,
        "clean_normals": clean_normals,
        "observed_points": observed_points,
        "corrupted_points": corrupted_points,
        "corrupted_normals": corrupted_normals,
        "point_defect_target": point_defect_target,
        "corruption_score_target": corruption_score_target,
        "intrinsic_patch_difficulty_target": intrinsic_target,
        "query_points_all": query_points_all,
        "query_labels_all": query_labels_all,
        "query_ignore_mask": query_ignore_mask,
    }


def _write_v2_patch(patch_path: Path, patch_id: str) -> None:
    rng = np.random.default_rng(0)
    clean_points = (rng.normal(size=(32, 3)) * 0.1).astype(np.float32)
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (32, 1))
    observed_points = clean_points[:16]
    surface_query_points = clean_points[:12]
    free_query_points = np.clip(clean_points[12:24] + np.asarray([0.0, 0.0, 0.2], dtype=np.float32), -1.0, 1.0)
    unknown_query_points = rng.uniform(-0.5, 0.5, size=(8, 3)).astype(np.float32)
    query_points_all = np.concatenate([surface_query_points, free_query_points, unknown_query_points], axis=0)
    query_labels_all = np.concatenate(
        [
            np.ones((len(surface_query_points),), dtype=np.int8),
            np.zeros((len(free_query_points),), dtype=np.int8),
            np.zeros((len(unknown_query_points),), dtype=np.int8),
        ]
    )
    query_ignore_mask = np.concatenate(
        [
            np.zeros((len(surface_query_points),), dtype=bool),
            np.zeros((len(free_query_points),), dtype=bool),
            np.ones((len(unknown_query_points),), dtype=bool),
        ]
    )
    sample = TeacherPatchSample(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        patch_center_world=np.zeros((3,), dtype=np.float32),
        patch_radius_m=3.0,
        town_id="Town02",
        sequence_id="Town02__150_streetsurf",
        tile_id=0,
        patch_id=patch_id,
        num_local_faces=24,
        num_observed_points_raw=16,
        teacher_area_local=1.0,
        source_town_mesh_cache_dir="mesh_cache",
        source_sequence_observed_cache="observed_cache",
        patch_cache_format_version=2,
        surface_query_points=surface_query_points,
        surface_query_labels=np.ones((len(surface_query_points),), dtype=np.int8),
        free_query_points=free_query_points,
        free_query_labels=np.zeros((len(free_query_points),), dtype=np.int8),
        unknown_query_points=unknown_query_points,
        query_points_all=query_points_all,
        query_labels_all=query_labels_all,
        query_ignore_mask=query_ignore_mask,
        camera_support_count=3,
        lidar_support_count=2,
        visible_surface_fraction=0.5,
        free_space_fraction=0.375,
        unknown_fraction=0.25,
        intrinsic_patch_difficulty_target=0.4,
        difficulty_components_json={"observed_to_clean_nn_error": 0.2},
        metadata={"planarity_hint": 0.5},
    )
    sample.save(patch_path)


def test_hybrid_v2_random_forward_and_losses() -> None:
    batch = _make_random_batch()
    model = LocalPatchDenoiser(
        model_type="hybrid_v2",
        latent_dim=48,
        retrieval_dim=16,
        recon_point_count=32,
        codebook_size=32,
        occupancy_hidden_dim=32,
        use_observed_condition=True,
    )
    outputs = model(
        corrupted_points=batch["corrupted_points"].float(),
        corrupted_normals=batch["corrupted_normals"].float(),
        observed_points=batch["observed_points"].float(),
        clean_points=batch["clean_points"].float(),
        clean_normals=batch["clean_normals"].float(),
        query_points_all=batch["query_points_all"].float(),
    )
    losses = compute_patch_losses(
        outputs,
        batch,
        weights={
            "recon_chamfer_loss": 1.0,
            "recon_normal_loss": 0.2,
            "point_defect_loss": 0.1,
            "corruption_score_loss": 0.1,
            "intrinsic_difficulty_loss": 0.2,
            "occupancy_bce_loss": 0.2,
            "free_space_violation_loss": 0.2,
            "vq_commitment_loss": 0.1,
            "prototype_diversity_loss": 0.05,
            "latent_align_loss": 0.05,
            "retrieval_align_loss": 0.05,
        },
    )

    assert outputs["recon_points"].shape == (2, 32, 3)
    assert outputs["recon_normals"].shape == (2, 32, 3)
    assert outputs["query_occupancy_logits"].shape == (2, 48)
    assert outputs["intrinsic_difficulty_pred"].shape == (2,)
    assert outputs["quantized_latent"].shape == (2, 48)
    assert outputs["code_indices"].shape == (2,)
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["occupancy_bce_loss"])
    assert torch.isfinite(losses["vq_commitment_loss"])
    assert torch.isfinite(losses["intrinsic_difficulty_loss"])
    assert np.isfinite(
        occupancy_iou_visible(
            outputs["query_occupancy_logits"],
            batch["query_labels_all"],
            batch["query_ignore_mask"],
        )
    )
    assert np.isfinite(
        free_space_violation_rate(
            outputs["query_occupancy_logits"],
            batch["query_labels_all"],
            batch["query_ignore_mask"],
        )
    )
    assert np.isfinite(
        intrinsic_difficulty_mae(
            outputs["intrinsic_difficulty_pred"],
            batch["intrinsic_patch_difficulty_target"],
        )
    )
    spearman_value = intrinsic_difficulty_spearman(
        outputs["intrinsic_difficulty_pred"],
        batch["intrinsic_patch_difficulty_target"],
    )
    assert np.isfinite(spearman_value) or np.isnan(spearman_value)
    assert np.isfinite(prototype_usage_entropy(outputs["code_indices"], codebook_size=32))


def test_hybrid_v2_forward_on_v2_dataset_batch(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patch_cache" / "Town02" / "Town02__150_streetsurf"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "patch_000.npz"
    _write_v2_patch(patch_path, "patch_000")
    index_path = tmp_path / "patch_cache" / "patch_index.jsonl"
    write_patch_index_jsonl(
        index_path,
        [
            PatchIndexRecord(
                patch_id="patch_000",
                town_id="Town02",
                sequence_id="Town02__150_streetsurf",
                tile_id=0,
                patch_file=str(patch_path),
                num_local_faces=24,
                num_observed_points_raw=16,
                num_clean_points=32,
                num_observed_points=16,
                teacher_area_local=1.0,
                planarity_hint=0.5,
                patch_cache_format_version=2,
                num_surface_query_points=12,
                num_free_query_points=12,
                num_unknown_query_points=8,
                camera_support_count=3,
                lidar_support_count=2,
                visible_surface_fraction=0.5,
                free_space_fraction=0.375,
                unknown_fraction=0.25,
                intrinsic_patch_difficulty_target=0.4,
                difficulty_components_json={"observed_to_clean_nn_error": 0.2},
            )
        ],
    )
    dataset = TeacherPatchTrainDataset(
        patch_index_path=index_path,
        split_config={"train_towns": ["Town02"]},
        subsets=("train",),
        corruption_config=_corruption_config(),
        seed=0,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    model = LocalPatchDenoiser(
        model_type="hybrid_v2",
        latent_dim=32,
        retrieval_dim=16,
        recon_point_count=32,
        codebook_size=16,
        occupancy_hidden_dim=32,
        use_observed_condition=True,
    )
    outputs = model(
        corrupted_points=batch["corrupted_points"].float(),
        corrupted_normals=batch["corrupted_normals"].float(),
        observed_points=batch["observed_points"].float(),
        clean_points=batch["clean_points"].float(),
        clean_normals=batch["clean_normals"].float(),
        query_points_all=batch["query_points_all"].float(),
    )
    losses = compute_patch_losses(
        outputs,
        batch,
        weights={
            "intrinsic_difficulty_loss": 0.2,
            "occupancy_bce_loss": 0.2,
            "free_space_violation_loss": 0.2,
            "vq_commitment_loss": 0.1,
        },
    )

    assert int(batch["patch_cache_format_version"][0]) == 2
    assert batch["query_points_all"].shape[1] == 32
    assert outputs["query_occupancy_logits"].shape == (1, 32)
    assert torch.isfinite(losses["total_loss"])
