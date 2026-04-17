from __future__ import annotations

import json
from pathlib import Path

from ss3dm_prior.eval import main as eval_main
from ss3dm_prior.train import main as train_main
from ss3dm_prior.utils.io import dump_yaml


def test_eval_smoke(tmp_path: Path) -> None:
    from tests.ss3dm_prior.test_train_smoke import _write_patch
    from ss3dm_prior.data.patch_index import write_patch_index_jsonl

    patch_cache_dir = tmp_path / "patch_cache"
    patch_dir = patch_cache_dir / "TownUnit" / "TownUnit__seq"
    patch_dir.mkdir(parents=True, exist_ok=True)
    records = [_write_patch(patch_dir / f"patch_{idx:03d}.npz", f"patch_{idx:03d}", idx) for idx in range(4)]
    write_patch_index_jsonl(patch_cache_dir / "patch_index.jsonl", records)

    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    train_config_path = tmp_path / "train.yaml"
    split_config_path = tmp_path / "split.yaml"
    eval_split_config_path = tmp_path / "eval_split.yaml"
    train_output_dir = tmp_path / "train_output"
    eval_output_root = tmp_path / "eval_output"

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
    dump_yaml(eval_split_config_path, {"test_towns": ["TownUnit"]})

    train_exit = train_main(
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
            str(train_output_dir),
            "--run_name",
            "smoke_train",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert train_exit == 0

    eval_exit = eval_main(
        [
            "--checkpoint",
            str(train_output_dir / "checkpoints" / "best_recon.pt"),
            "--manifest_path",
            str(tmp_path / "manifest.json"),
            "--patch_cache_dir",
            str(patch_cache_dir),
            "--split_config",
            str(eval_split_config_path),
            "--output_dir",
            str(eval_output_root),
            "--eval_name",
            "smoke_eval",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert eval_exit == 0
    eval_dir = eval_output_root / "smoke_eval"
    assert (eval_dir / "metrics_summary.json").exists()
    assert (eval_dir / "metrics_per_town.csv").exists()
    assert (eval_dir / "metrics_per_sequence.csv").exists()
    assert (eval_dir / "patch_predictions.csv").exists()
    assert (eval_dir / "report.md").exists()
    assert list((eval_dir / "patch_panels").glob("*.png"))
    summary = json.loads((eval_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert "protocol_valid" in summary
    assert "protocol_warnings" in summary
    assert "protocol_summary" in summary
    report_text = (eval_dir / "report.md").read_text(encoding="utf-8")
    assert "## Protocol Audit" in report_text
