from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.data.train_dataset import TeacherPatchTrainDataset
from ss3dm_prior.losses import compute_patch_losses
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser


def _write_patch(patch_path: Path, patch_id: str) -> None:
    clean_points = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32) * 0.1
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (32, 1))
    observed_points = clean_points[:16]
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
        metadata={"planarity_hint": 0.5},
    )
    sample.save(patch_path)


def test_dataset_dataloader_and_model_forward(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patch_cache" / "Town02" / "Town02__150_streetsurf"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "patch_000.npz"
    _write_patch(patch_path, "patch_000")

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
            )
        ],
    )

    dataset = TeacherPatchTrainDataset(
        patch_index_path=index_path,
        split_config={"train_towns": ["Town02"]},
        subsets=("train",),
        corruption_config={
            "target_corrupted_count": 32,
            "point_dropout": {"enabled": True, "dropout_ratio": 0.1},
            "gaussian_jitter": {"enabled": True, "sigma": 0.01, "anisotropic": False},
            "normal_noise": {"enabled": True, "sigma": 0.05, "flip_prob": 0.0},
            "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.1},
            "outlier_cluster": {"enabled": True, "cluster_size": 4, "cluster_offset_sigma": 0.1, "cluster_spread_sigma": 0.01},
            "density_imbalance": {"enabled": True, "region_radius": 0.15, "thin_probability": 0.5, "thin_ratio": 0.5, "duplicate_count": 4},
        },
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    model = LocalPatchDenoiser(latent_dim=32, retrieval_dim=16, recon_point_count=32, use_observed_condition=True)
    outputs = model(
        corrupted_points=batch["corrupted_points"].float(),
        corrupted_normals=batch["corrupted_normals"].float(),
        observed_points=batch["observed_points"].float(),
        clean_points=batch["clean_points"].float(),
        clean_normals=batch["clean_normals"].float(),
    )
    losses = compute_patch_losses(outputs, batch)

    assert outputs["recon_points"].shape == (1, 32, 3)
    assert outputs["recon_normals"].shape == (1, 32, 3)
    assert outputs["point_defect_pred"].shape == (1, 32)
    assert outputs["patch_score_pred"].shape == (1,)
    assert outputs["retrieval_embedding"].shape == (1, 16)
    assert outputs["clean_retrieval_embedding"].shape == (1, 16)
    assert outputs["fused_latent"].shape == (1, 64)
    assert torch.isfinite(losses["total_loss"])
    assert torch.isfinite(losses["retrieval_align_loss"])
