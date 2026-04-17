from __future__ import annotations

import csv
import json
from pathlib import Path

import trimesh

from ss3dm_prior.tools.build_car_mesh_patch_cache import main as build_car_cache_main
from ss3dm_prior.train import main as train_main
from ss3dm_prior.utils.io import dump_yaml


def _write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sha256",
        "file_identifier",
        "captions",
        "local_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_mesh(path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "box":
        mesh = trimesh.creation.box(extents=(1.4, 0.8, 0.5))
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    else:
        mesh = trimesh.creation.cylinder(radius=0.35, height=1.4, sections=32)
    mesh.export(path)


def test_car_mesh_patch_cache_and_train_smoke(tmp_path: Path) -> None:
    dataset_root = tmp_path / "meshfleet_like"
    train_raw = dataset_root / "train" / "raw"
    test_raw = dataset_root / "test" / "raw"
    train_rows = [
        {
            "sha256": "car_train_a",
            "file_identifier": "synthetic://car_train_a",
            "captions": "synthetic train car A",
            "local_path": "./raw/car_train_a.obj",
        },
        {
            "sha256": "car_train_b",
            "file_identifier": "synthetic://car_train_b",
            "captions": "synthetic train car B",
            "local_path": "./raw/car_train_b.obj",
        },
    ]
    test_rows = [
        {
            "sha256": "car_test_a",
            "file_identifier": "synthetic://car_test_a",
            "captions": "synthetic test car A",
            "local_path": "./raw/car_test_a.obj",
        }
    ]
    _write_mesh(train_raw / "car_train_a.obj", "box")
    _write_mesh(train_raw / "car_train_b.obj", "sphere")
    _write_mesh(test_raw / "car_test_a.obj", "cylinder")
    _write_metadata_csv(dataset_root / "train" / "metadata.csv", train_rows)
    _write_metadata_csv(dataset_root / "test" / "metadata.csv", test_rows)

    patch_cache_dir = tmp_path / "car_patch_cache"
    build_exit = build_car_cache_main(
        [
            "--dataset_root",
            str(dataset_root),
            "--out_dir",
            str(patch_cache_dir),
            "--val_fraction",
            "0.5",
            "--clean_sample_count",
            "128",
            "--observed_sample_count",
            "64",
            "--num_workers",
            "1",
            "--seed",
            "0",
        ]
    )
    assert build_exit == 0
    assert (patch_cache_dir / "patch_index.jsonl").exists()
    assert (patch_cache_dir / "split_meshfleet_car.yaml").exists()
    assert (patch_cache_dir / "source_mesh_manifest.json").exists()

    manifest_payload = json.loads((patch_cache_dir / "source_mesh_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["dataset_name"] == "meshfleet_car_whole_mesh"
    assert len(manifest_payload["records"]) == 3

    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    train_config_path = tmp_path / "train.yaml"
    output_dir = tmp_path / "train_output"
    dump_yaml(data_config_path, {"dataset": {"name": "meshfleet_car_smoke"}})
    dump_yaml(
        model_config_path,
        {
            "corruptions": {
                "target_corrupted_count": 128,
                "point_dropout": {"enabled": True, "dropout_ratio": 0.1},
                "gaussian_jitter": {"enabled": True, "sigma": 0.01, "anisotropic": False},
                "normal_noise": {"enabled": True, "sigma": 0.05, "flip_prob": 0.0},
                "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.1},
                "outlier_cluster": {"enabled": True, "cluster_size": 4, "cluster_offset_sigma": 0.1, "cluster_spread_sigma": 0.01},
                "density_imbalance": {"enabled": True, "region_radius": 0.15, "thin_probability": 0.5, "thin_ratio": 0.5, "duplicate_count": 4},
            },
            "model": {
                "model_type": "hybrid_v2",
                "latent_dim": 32,
                "retrieval_dim": 16,
                "recon_point_count": 128,
                "use_observed_condition": True,
                "use_local_frame": True,
                "use_residual_reconstruction": True,
                "use_vector_quantization": False,
                "codebook_size": 16,
                "occupancy_hidden_dim": 32,
            },
            "loss_weights": {
                "recon_chamfer_loss": 1.0,
                "recon_normal_loss": 0.2,
                "point_defect_loss": 0.5,
                "corruption_score_loss": 0.2,
                "intrinsic_difficulty_loss": 0.0,
                "intrinsic_difficulty_pairwise_weight": 0.0,
                "occupancy_bce_loss": 0.0,
                "free_space_violation_loss": 0.0,
                "vq_commitment_loss": 0.0,
                "prototype_diversity_loss": 0.0,
                "latent_align_loss": 0.05,
                "retrieval_align_loss": 0.05,
            },
        },
    )
    dump_yaml(
        train_config_path,
        {
            "train": {
                "seed": 0,
                "epochs": 1,
                "batch_size": 1,
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
                "step_visualization_interval_steps": 1,
                "step_visualization_num_examples": 1,
                "debug_use_all_patches_for_train_val": False,
                "allow_debug_split_override": False,
                "allow_split_fallback": False,
                "curriculum": {
                    "warmup_epochs": 0,
                    "main_start_epoch": 0,
                    "occupancy_start_epoch": 999,
                    "intrinsic_start_epoch": 999,
                    "vq_start_epoch": 999,
                    "prototype_start_epoch": 999,
                },
                "hard_example_sampling": {
                    "enable": False,
                    "alpha": 1.0,
                    "floor": 1.0,
                    "power": 1.0,
                },
                "checkpoint_selection": {
                    "best_composite_weights": {
                        "val_denoise_gain_chamfer": 1.0,
                        "val_recon_chamfer_l1": -1.0,
                    },
                    "best_visibility_weights": {},
                },
            }
        },
    )

    train_exit = train_main(
        [
            "--data_config",
            str(data_config_path),
            "--model_config",
            str(model_config_path),
            "--train_config",
            str(train_config_path),
            "--manifest_path",
            str(patch_cache_dir / "source_mesh_manifest.json"),
            "--patch_cache_dir",
            str(patch_cache_dir),
            "--split_config",
            str(patch_cache_dir / "split_meshfleet_car.yaml"),
            "--output_dir",
            str(output_dir),
            "--run_name",
            "meshfleet_car_smoke",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert train_exit == 0
    assert (output_dir / "checkpoints" / "last.pt").exists()
    assert (output_dir / "checkpoints" / "best_recon.pt").exists()
    assert (output_dir / "checkpoints" / "best_gain.pt").exists()
    assert (output_dir / "checkpoints" / "best_composite.pt").exists()
