from __future__ import annotations

from pathlib import Path

from ss3dm_prior.train import main as train_main

from tests.ss3dm_prior.smoke_v3_utils import build_v3_smoke_fixture


def test_trainer_v3_smoke(tmp_path: Path) -> None:
    fixture = build_v3_smoke_fixture(tmp_path)
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
            "smoke_train_v3",
            "--wandb_mode",
            "disabled",
        ]
    )

    assert exit_code == 0
    ckpt_dir = fixture["output_dir"] / "checkpoints"
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best_recon.pt").exists()
    assert (ckpt_dir / "best_gain.pt").exists()
    assert (ckpt_dir / "best_composite.pt").exists()
    assert (ckpt_dir / "best_visibility.pt").exists()
    assert (ckpt_dir / "best_paper.pt").exists()

    epoch_dir = fixture["output_dir"] / "visualizations" / "epoch_000"
    assert list(epoch_dir.glob("*_visibility_panel.png"))
    assert list(epoch_dir.glob("*_visible_vs_hidden_panel.png"))
    assert list(epoch_dir.glob("*_free_space_error_panel.png"))
    assert list(epoch_dir.glob("*_difficulty_calibration_panel.png"))
    assert (epoch_dir / "prototype_usage_gallery.png").exists()
