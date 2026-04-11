from __future__ import annotations

from pathlib import Path

import numpy as np

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.train import main as train_main
from ss3dm_prior.utils.io import dump_yaml


def _write_patch(patch_path: Path, patch_id: str, rng_seed: int) -> PatchIndexRecord:
    rng = np.random.default_rng(rng_seed)
    clean_points = (rng.normal(size=(64, 3)) * 0.1).astype(np.float32)
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (64, 1))
    observed_points = clean_points[:32]
    sample = TeacherPatchSample(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        patch_center_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        patch_radius_m=3.0,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        tile_id=rng_seed,
        patch_id=patch_id,
        num_local_faces=32,
        num_observed_points_raw=32,
        teacher_area_local=1.0,
        source_town_mesh_cache_dir="mesh_cache",
        source_sequence_observed_cache="observed_cache",
        metadata={"planarity_hint": 0.5},
    )
    sample.save(patch_path)
    return PatchIndexRecord(
        patch_id=patch_id,
        town_id="TownUnit",
        sequence_id="TownUnit__seq",
        tile_id=rng_seed,
        patch_file=str(patch_path),
        num_local_faces=32,
        num_observed_points_raw=32,
        num_clean_points=64,
        num_observed_points=32,
        teacher_area_local=1.0,
        planarity_hint=0.5,
    )


def test_train_smoke(tmp_path: Path) -> None:
    patch_cache_dir = tmp_path / "patch_cache"
    patch_dir = patch_cache_dir / "TownUnit" / "TownUnit__seq"
    patch_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _write_patch(patch_dir / f"patch_{idx:03d}.npz", f"patch_{idx:03d}", idx)
        for idx in range(4)
    ]
    write_patch_index_jsonl(patch_cache_dir / "patch_index.jsonl", records)

    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    train_config_path = tmp_path / "train.yaml"
    split_config_path = tmp_path / "split.yaml"
    output_dir = tmp_path / "train_output"

    dump_yaml(data_config_path, {"dataset": {"name": "smoke"}})
    dump_yaml(
        model_config_path,
        {
            "corruptions": {
                "target_corrupted_count": 64,
                "point_dropout": {"enabled": True, "dropout_ratio": 0.1},
                "gaussian_jitter": {"enabled": True, "sigma": 0.01, "anisotropic": False},
                "normal_noise": {"enabled": True, "sigma": 0.05, "flip_prob": 0.0},
                "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.1},
                "outlier_cluster": {"enabled": True, "cluster_size": 4, "cluster_offset_sigma": 0.1, "cluster_spread_sigma": 0.01},
                "density_imbalance": {"enabled": True, "region_radius": 0.15, "thin_probability": 0.5, "thin_ratio": 0.5, "duplicate_count": 4},
            },
            "model": {
                "latent_dim": 32,
                "retrieval_dim": 16,
                "recon_point_count": 64,
                "use_observed_condition": True,
                "use_local_frame": True,
                "use_residual_reconstruction": True,
            },
            "loss_weights": {
                "recon_chamfer_loss": 1.0,
                "recon_normal_loss": 0.5,
                "point_defect_loss": 1.0,
                "patch_score_loss": 0.5,
                "latent_align_loss": 0.05,
                "retrieval_align_loss": 0.1,
            },
        },
    )
    dump_yaml(
        train_config_path,
        {
            "train": {
                "seed": 0,
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "amp": False,
                "log_interval": 1,
                "val_interval": 1,
                "save_interval": 1,
                "wandb_enable": False,
                "wandb_mode": "disabled",
                "max_visualization_examples": 1,
                "debug_use_all_patches_for_train_val": True,
                "allow_debug_split_override": True,
                "allow_split_fallback": True,
                "debug_val_fraction": 0.5,
            }
        },
    )
    dump_yaml(split_config_path, {"train_towns": ["TownUnit"], "val_towns": ["TownUnit"]})

    exit_code = train_main(
        [
            "--data_config",
            str(data_config_path),
            "--model_config",
            str(model_config_path),
            "--train_config",
            str(train_config_path),
            "--patch_cache_dir",
            str(patch_cache_dir),
            "--split_config",
            str(split_config_path),
            "--output_dir",
            str(output_dir),
            "--run_name",
            "smoke_train",
            "--wandb_mode",
            "disabled",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "checkpoints" / "last.pt").exists()
    assert (output_dir / "checkpoints" / "best_recon.pt").exists()
    assert (output_dir / "checkpoints" / "best_gain.pt").exists()
    assert list((output_dir / "visualizations" / "epoch_000").glob("*_panel.png"))
