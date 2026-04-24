from __future__ import annotations

import json
from pathlib import Path

import torch

from ss3dm_prior.eval import main as eval_main
from ss3dm_prior.models.patch_denoiser import LocalPatchDenoiser
from ss3dm_prior.train import main as train_main
from ss3dm_prior.utils.io import dump_yaml, load_yaml
from tests.ss3dm_prior.smoke_v3_utils import build_v3_smoke_fixture


def _rewrite_fixture_for_v11(tmp_path: Path) -> dict[str, Path]:
    fixture = build_v3_smoke_fixture(tmp_path)
    model_cfg = load_yaml(fixture["model_config_path"])
    train_cfg = load_yaml(fixture["train_config_path"])
    model_cfg["model"].update(
        {
            "model_type": "v11_latent_flow_hybrid",
            "latent_flow_hidden_dims": [64, 64],
            "latent_flow_time_embed_dim": 32,
            "latent_flow_dropout": 0.0,
            "latent_flow_target": "hidden_residual",
            "stochastic_eval_k_list": [1, 4, 8],
            "stochastic_flow_steps": 4,
            "stochastic_free_space_safe_threshold": 0.15,
            "stochastic_rerank_weights": {
                "observed_consistency": 0.20,
                "visible_consistency": -0.35,
                "free_space_penalty": -0.30,
                "prototype_consistency": 0.15,
            },
        }
    )
    model_cfg["loss_weights"]["latent_flow_matching_loss"] = 0.10
    train_cfg["train"]["batch_size"] = 1
    train_cfg["train"]["grad_accum_steps"] = 1
    dump_yaml(fixture["model_config_path"], model_cfg)
    dump_yaml(fixture["train_config_path"], train_cfg)
    return fixture


def test_model_v11_latent_flow_random_forward() -> None:
    model = LocalPatchDenoiser(
        model_type="v11_latent_flow_hybrid",
        latent_dim=32,
        retrieval_dim=16,
        recon_point_count=96,
        use_observed_condition=True,
        use_local_frame=True,
        use_residual_reconstruction=True,
        codebook_size=16,
        vq_commitment_cost=0.25,
        use_vector_quantization=True,
        teacher_encoder_enabled=True,
        occupancy_hidden_dim=32,
        num_latent_queries=8,
        num_cross_attention_layers=2,
        num_latent_self_attention_layers=1,
        attention_heads=4,
        ffn_dim=128,
        dropout=0.0,
        token_hidden_dims=[32, 32],
        decoder_hidden_dims=[64, 32],
        point_defect_hidden_dims=[32, 16],
        score_head_hidden_dims=[32, 16],
        intrinsic_head_hidden_dims=[32, 16],
        retrieval_head_hidden_dims=[32, 16],
        occupancy_hidden_dims=[32, 32],
        latent_flow_hidden_dims=[64, 64],
        latent_flow_time_embed_dim=32,
        latent_flow_dropout=0.0,
        stochastic_eval_k_list=[1, 4, 8],
        stochastic_flow_steps=4,
    )
    batch_size = 2
    outputs = model(
        corrupted_points=torch.randn(batch_size, 96, 3),
        corrupted_normals=torch.randn(batch_size, 96, 3),
        observed_points=torch.randn(batch_size, 32, 3),
        clean_points=torch.randn(batch_size, 96, 3),
        clean_normals=torch.randn(batch_size, 96, 3),
        query_points_all=torch.randn(batch_size, 64, 3),
        visible_clean_points=torch.randn(batch_size, 48, 3),
        visible_clean_normals=torch.randn(batch_size, 48, 3),
        hidden_clean_points=torch.randn(batch_size, 48, 3),
        hidden_clean_normals=torch.randn(batch_size, 48, 3),
        sample_latent_candidates_k=4,
        stochastic_flow_steps=4,
    )
    assert outputs["recon_points"] is not None
    assert outputs["latent_flow_matching_loss"] is not None
    assert outputs["stochastic_candidate_recon_points"] is not None
    assert tuple(outputs["stochastic_candidate_recon_points"].shape[:3]) == (batch_size, 4, 96)


def test_model_v11_latent_flow_train_smoke(tmp_path: Path) -> None:
    fixture = _rewrite_fixture_for_v11(tmp_path)
    exit_code = train_main(
        [
            "--data_config",
            str(fixture["data_config_path"]),
            "--model_config",
            str(fixture["model_config_path"]),
            "--train_config",
            str(fixture["train_config_path"]),
            "--patch_cache_dir",
            str(fixture["patch_cache_dir"]),
            "--split_config",
            str(fixture["split_config_path"]),
            "--output_dir",
            str(fixture["output_dir"]),
            "--run_name",
            "smoke_train_v11",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert exit_code == 0
    assert (fixture["output_dir"] / "checkpoints" / "best_paper.pt").exists()


def test_model_v11_latent_flow_eval_smoke(tmp_path: Path) -> None:
    fixture = _rewrite_fixture_for_v11(tmp_path)
    train_exit = train_main(
        [
            "--data_config",
            str(fixture["data_config_path"]),
            "--model_config",
            str(fixture["model_config_path"]),
            "--train_config",
            str(fixture["train_config_path"]),
            "--patch_cache_dir",
            str(fixture["patch_cache_dir"]),
            "--split_config",
            str(fixture["split_config_path"]),
            "--output_dir",
            str(fixture["output_dir"]),
            "--run_name",
            "smoke_train_v11_eval",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert train_exit == 0
    eval_exit = eval_main(
        [
            "--checkpoint",
            str(fixture["output_dir"] / "checkpoints" / "best_paper.pt"),
            "--manifest_path",
            str(tmp_path / "manifest.json"),
            "--patch_cache_dir",
            str(fixture["patch_cache_dir"]),
            "--split_config",
            str(fixture["eval_split_config_path"]),
            "--output_dir",
            str(fixture["eval_output_root"]),
            "--eval_name",
            "smoke_eval_v11",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert eval_exit == 0
    eval_dir = fixture["eval_output_root"] / "smoke_eval_v11"
    summary = json.loads((eval_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    for key in [
        "best_of_k_hidden_completion",
        "mean_of_k_hidden_completion",
        "sample_diversity",
        "free_space_safe_best_of_k",
        "stochastic_comparison",
    ]:
        assert key in summary
    assert list((eval_dir / "patch_panels").glob("*__stochastic_candidates.png"))
