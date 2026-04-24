from __future__ import annotations

import json
from pathlib import Path

from ss3dm_prior.eval import main as eval_main
from ss3dm_prior.train import main as train_main

from tests.ss3dm_prior.smoke_v3_utils import build_v3_smoke_fixture


def test_eval_v3_smoke(tmp_path: Path) -> None:
    fixture = build_v3_smoke_fixture(tmp_path)
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
            "smoke_train_v3_eval",
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
            "smoke_eval_v3",
            "--wandb_mode",
            "disabled",
        ]
    )
    assert eval_exit == 0

    eval_dir = fixture["eval_output_root"] / "smoke_eval_v3"
    summary = json.loads((eval_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    for key in [
        "recon_chamfer_l1",
        "visible_recon_chamfer_l1",
        "hidden_completion_chamfer_l1",
        "visible_recon_normal_cosine",
        "hidden_completion_gain",
        "intrinsic_difficulty_calibration_mae",
        "free_space_fp_rate",
        "retrieval_top1_nonself",
        "retrieval_top5_nonself",
        "protocol_valid",
    ]:
        assert key in summary
    assert summary["protocol_valid"] is True

    assert (eval_dir / "metrics_per_town.csv").exists()
    assert (eval_dir / "metrics_per_sequence.csv").exists()
    assert (eval_dir / "patch_predictions.csv").exists()
    assert (eval_dir / "report.md").exists()
    assert list((eval_dir / "patch_panels").glob("*__visible_vs_hidden_panel.png"))
    assert list((eval_dir / "patch_panels").glob("*__free_space_error_panel.png"))
    assert list((eval_dir / "patch_panels").glob("*__difficulty_calibration_panel.png"))
    assert (eval_dir / "prototype_gallery" / "prototype_gallery.png").exists()
    assert (eval_dir / "difficulty_calibration_panel.png").exists()
    assert len(list((eval_dir / "sequence_maps").glob("*.png"))) >= 2
