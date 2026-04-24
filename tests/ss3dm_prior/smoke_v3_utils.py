from __future__ import annotations

from pathlib import Path

import numpy as np

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.utils.io import dump_yaml


def _write_v3_patch(
    patch_path: Path,
    *,
    patch_id: str,
    town_id: str,
    sequence_id: str,
    tile_id: int,
    center_x: float,
    center_y: float,
    intrinsic_target: float,
    scale_id: int,
    patch_radius_m: float,
    rng_seed: int,
) -> PatchIndexRecord:
    rng = np.random.default_rng(rng_seed)
    clean_points = (rng.normal(size=(96, 3)) * (0.06 + 0.01 * scale_id)).astype(np.float32)
    clean_normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (96, 1))
    visible_clean_points = clean_points[:48]
    visible_clean_normals = clean_normals[:48]
    hidden_clean_points = clean_points[48:]
    hidden_clean_normals = clean_normals[48:]
    observed_points = visible_clean_points[:32]
    surface_query_points = clean_points[:24]
    free_query_points = np.clip(clean_points[24:48] + np.asarray([0.0, 0.0, 0.18], dtype=np.float32), -1.0, 1.0)
    hard_negatives = free_query_points[:12]
    unknown_query_points = rng.uniform(-0.35, 0.35, size=(16, 3)).astype(np.float32)
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
    visible_surface_fraction = float(len(visible_clean_points) / len(clean_points))
    visible_support_fraction = float(len(observed_points) / max(len(visible_clean_points), 1))
    hidden_surface_fraction = float(len(hidden_clean_points) / len(clean_points))
    free_space_fraction = float(len(free_query_points) / len(query_points_all))
    unknown_fraction = float(len(unknown_query_points) / len(query_points_all))
    patch_center = np.asarray([center_x, center_y, 0.0], dtype=np.float32)
    sample = TeacherPatchSample(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        patch_center_world=patch_center,
        patch_radius_m=patch_radius_m,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_id=patch_id,
        num_local_faces=48,
        num_observed_points_raw=len(observed_points),
        teacher_area_local=1.0,
        source_town_mesh_cache_dir="mesh_cache",
        source_sequence_observed_cache="observed_cache",
        patch_cache_format_version=3,
        surface_query_points=surface_query_points,
        surface_query_labels=np.ones((len(surface_query_points),), dtype=np.int8),
        free_query_points=free_query_points,
        free_query_labels=np.zeros((len(free_query_points),), dtype=np.int8),
        free_space_query_hard_negatives=hard_negatives,
        unknown_query_points=unknown_query_points,
        query_points_all=query_points_all,
        query_labels_all=query_labels_all,
        query_ignore_mask=query_ignore_mask,
        visible_clean_points=visible_clean_points,
        visible_clean_normals=visible_clean_normals,
        hidden_clean_points=hidden_clean_points,
        hidden_clean_normals=hidden_clean_normals,
        surface_support_mask=np.concatenate(
            [np.ones((len(visible_clean_points),), dtype=bool), np.zeros((len(hidden_clean_points),), dtype=bool)]
        ),
        camera_support_count=2 + scale_id,
        lidar_support_count=3 + scale_id,
        visible_surface_fraction=visible_surface_fraction,
        visible_support_fraction=visible_support_fraction,
        hidden_surface_fraction=hidden_surface_fraction,
        free_space_fraction=free_space_fraction,
        unknown_fraction=unknown_fraction,
        free_space_hard_negative_count=len(hard_negatives),
        intrinsic_patch_difficulty_target=intrinsic_target,
        difficulty_components_json={"observed_to_clean_nn_error": round(0.15 + 0.03 * rng_seed, 4)},
        metadata={"planarity_hint": 0.5},
        scale_id=scale_id,
    )
    sample.save(patch_path)
    return PatchIndexRecord(
        patch_id=patch_id,
        town_id=town_id,
        sequence_id=sequence_id,
        tile_id=tile_id,
        patch_file=str(patch_path),
        num_local_faces=48,
        num_observed_points_raw=len(observed_points),
        num_clean_points=len(clean_points),
        num_observed_points=len(observed_points),
        teacher_area_local=1.0,
        planarity_hint=0.5,
        scale_id=scale_id,
        patch_radius_m=patch_radius_m,
        patch_cache_format_version=3,
        num_surface_query_points=len(surface_query_points),
        num_free_query_points=len(free_query_points),
        num_unknown_query_points=len(unknown_query_points),
        camera_support_count=2 + scale_id,
        lidar_support_count=3 + scale_id,
        visible_surface_fraction=visible_surface_fraction,
        free_space_fraction=free_space_fraction,
        unknown_fraction=unknown_fraction,
        intrinsic_patch_difficulty_target=intrinsic_target,
        num_visible_clean_points=len(visible_clean_points),
        num_hidden_clean_points=len(hidden_clean_points),
        visible_support_fraction=visible_support_fraction,
        hidden_surface_fraction=hidden_surface_fraction,
        free_space_hard_negative_count=len(hard_negatives),
        difficulty_components_json={"observed_to_clean_nn_error": round(0.15 + 0.03 * rng_seed, 4)},
    )


def build_v3_smoke_fixture(tmp_path: Path) -> dict[str, Path]:
    patch_cache_dir = tmp_path / "patch_cache_v3"
    patch_specs = [
        ("TownTrain", "TownTrain__seq0", 0.0, 0.0, 0.20, 0, 2.0),
        ("TownTrain", "TownTrain__seq0", 1.0, 0.0, 0.28, 1, 4.0),
        ("TownVal", "TownVal__seq0", 0.0, 1.0, 0.36, 0, 2.0),
        ("TownVal", "TownVal__seq0", 1.0, 1.0, 0.44, 1, 4.0),
        ("TownEval", "TownEval__seqA", 0.0, 2.0, 0.56, 0, 2.0),
        ("TownEval", "TownEval__seqA", 1.0, 2.0, 0.62, 1, 4.0),
    ]
    records: list[PatchIndexRecord] = []
    for idx, (town_id, sequence_id, cx, cy, intrinsic_target, scale_id, patch_radius_m) in enumerate(patch_specs):
        patch_dir = patch_cache_dir / town_id / sequence_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        radius_token = str(f"{patch_radius_m:.2f}").replace(".", "p")
        patch_id = f"{sequence_id}__tile_{idx:06d}__scale_{scale_id:02d}__r{radius_token}m"
        records.append(
            _write_v3_patch(
                patch_dir / f"patch_{idx:03d}.npz",
                patch_id=patch_id,
                town_id=town_id,
                sequence_id=sequence_id,
                tile_id=idx,
                center_x=cx,
                center_y=cy,
                intrinsic_target=intrinsic_target,
                scale_id=scale_id,
                patch_radius_m=patch_radius_m,
                rng_seed=idx,
            )
        )
    write_patch_index_jsonl(patch_cache_dir / "patch_index.jsonl", records)

    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    train_config_path = tmp_path / "train.yaml"
    split_config_path = tmp_path / "split.yaml"
    eval_split_config_path = tmp_path / "eval_split.yaml"
    output_dir = tmp_path / "train_output_v3"
    eval_output_root = tmp_path / "eval_output_v3"

    dump_yaml(data_config_path, {"dataset": {"name": "smoke_v3"}})
    dump_yaml(
        model_config_path,
        {
            "corruptions": {
                "target_corrupted_count": 96,
                "point_dropout": {"enabled": True, "dropout_ratio": 0.1},
                "gaussian_jitter": {"enabled": True, "sigma": 0.01, "anisotropic": False},
                "normal_noise": {"enabled": True, "sigma": 0.05, "flip_prob": 0.0},
                "local_hole_mask": {"enabled": True, "max_holes": 1, "hole_radius": 0.08},
                "outlier_cluster": {"enabled": True, "cluster_size": 4, "cluster_offset_sigma": 0.1, "cluster_spread_sigma": 0.01},
                "density_imbalance": {"enabled": True, "region_radius": 0.15, "thin_probability": 0.5, "thin_ratio": 0.5, "duplicate_count": 4},
                "severity_schedule": {"type": "linear", "start_scale": 0.8, "end_scale": 1.0, "warmup_epochs": 1},
            },
            "model": {
                "model_type": "v10_cross_attention_hybrid",
                "latent_dim": 32,
                "retrieval_dim": 16,
                "recon_point_count": 96,
                "use_observed_condition": True,
                "use_local_frame": True,
                "use_residual_reconstruction": True,
                "codebook_size": 16,
                "vq_commitment_cost": 0.25,
                "use_vector_quantization": True,
                "teacher_encoder_enabled": True,
                "occupancy_hidden_dim": 32,
                "num_latent_queries": 8,
                "num_cross_attention_layers": 2,
                "num_latent_self_attention_layers": 1,
                "attention_heads": 4,
                "ffn_dim": 128,
                "dropout": 0.0,
                "token_hidden_dims": [32, 32],
                "decoder_hidden_dims": [64, 32],
                "point_defect_hidden_dims": [32, 16],
                "score_head_hidden_dims": [32, 16],
                "intrinsic_head_hidden_dims": [32, 16],
                "retrieval_head_hidden_dims": [32, 16],
                "occupancy_hidden_dims": [32, 32],
            },
            "loss_weights": {
                "recon_chamfer_loss": 1.0,
                "recon_normal_loss": 0.2,
                "point_defect_loss": 0.4,
                "corruption_score_loss": 0.2,
                "intrinsic_difficulty_loss": 0.2,
                "intrinsic_difficulty_pairwise_weight": 0.2,
                "occupancy_bce_loss": 0.2,
                "free_space_violation_loss": 0.2,
                "vq_commitment_loss": 0.05,
                "prototype_diversity_loss": 0.01,
                "latent_align_loss": 0.01,
                "retrieval_align_loss": 0.01,
                "hard_example_reweight": 0.5,
                "hard_weight_source": "blend",
                "hard_weight_blend_alpha": 0.5,
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
                "grad_accum_steps": 2,
                "num_workers": 0,
                "lr": 1e-3,
                "min_lr": 1e-5,
                "lr_scheduler": "cosine",
                "weight_decay": 1e-4,
                "grad_clip_norm": 1.0,
                "amp": False,
                "ema": {"enable": True, "decay": 0.99, "use_for_eval": True},
                "log_interval": 1,
                "val_interval": 1,
                "save_interval": 1,
                "wandb_enable": False,
                "wandb_mode": "disabled",
                "max_visualization_examples": 1,
                "fixed_visualization_patch_ids": [],
                "step_visualization_interval_steps": 1,
                "step_visualization_num_examples": 1,
                "step_visualization_patch_ids": [],
                "debug_use_all_patches_for_train_val": False,
                "allow_debug_split_override": False,
                "allow_split_fallback": False,
                "debug_val_fraction": 0.5,
                "curriculum": {
                    "recon_warmup_epochs": 0,
                    "warmup_epochs": 0,
                    "main_start_epoch": 0,
                    "occupancy_start_epoch": 0,
                    "intrinsic_start_epoch": 0,
                    "vq_start_epoch": 0,
                    "prototype_start_epoch": 0,
                },
                "hard_example_sampling": {
                    "enable": True,
                    "alpha": 1.0,
                    "floor": 1.0,
                    "power": 1.0,
                },
                "checkpoint_selection": {
                    "best_composite_weights": {
                        "val_denoise_gain_chamfer": 1.0,
                        "val_recon_chamfer_l1": -1.0,
                        "val_visible_recon_chamfer_l1": -0.5,
                        "val_hidden_completion_chamfer_l1": -0.5,
                        "val_occupancy_iou_visible": 0.5,
                        "val_free_space_violation_rate": -0.5,
                        "val_intrinsic_difficulty_spearman": 0.2,
                    },
                    "best_visibility_weights": {
                        "val_occupancy_iou_visible": 1.0,
                        "val_free_space_violation_rate": -1.0,
                    },
                    "best_paper_weights": {
                        "val_denoise_gain_chamfer": 1.0,
                        "val_recon_chamfer_l1": -1.0,
                        "val_intrinsic_difficulty_spearman": 0.25,
                        "val_occupancy_iou_visible": 0.5,
                        "val_free_space_violation_rate": -0.5,
                        "val_retrieval_top1_nonself": 0.25,
                    },
                },
            }
        },
    )
    dump_yaml(split_config_path, {"train_towns": ["TownTrain"], "val_towns": ["TownVal"], "test_towns": ["TownEval"]})
    dump_yaml(eval_split_config_path, {"test_towns": ["TownEval"]})
    return {
        "patch_cache_dir": patch_cache_dir,
        "data_config_path": data_config_path,
        "model_config_path": model_config_path,
        "train_config_path": train_config_path,
        "split_config_path": split_config_path,
        "eval_split_config_path": eval_split_config_path,
        "output_dir": output_dir,
        "eval_output_root": eval_output_root,
    }
