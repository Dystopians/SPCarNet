from __future__ import annotations

from pathlib import Path

from ss3dm_prior import train as train_module


def _make_load_yaml(train_config_path: Path, model_config_path: Path) -> callable:
    def _load_yaml(path: str | Path):
        resolved = Path(path)
        if resolved == train_config_path:
            return {
                "train": {
                    "wandb_enable": False,
                    "wandb_mode": "disabled",
                    "wandb_project": "cfg_project",
                }
            }
        if resolved == model_config_path:
            return {"corruptions": {}, "model": {}, "loss_weights": {}}
        return {}

    return _load_yaml


def test_train_entrypoint_keeps_config_wandb_when_cli_mode_absent(tmp_path: Path, monkeypatch) -> None:
    train_config_path = tmp_path / "train.yaml"
    model_config_path = tmp_path / "model.yaml"
    patch_cache_dir = tmp_path / "patch_cache"
    output_dir = tmp_path / "output"
    patch_cache_dir.mkdir()

    captured: dict[str, object] = {}

    monkeypatch.setattr(train_module, "load_yaml", _make_load_yaml(train_config_path, model_config_path))
    monkeypatch.setattr(train_module, "dump_yaml", lambda *_args, **_kwargs: None)

    class _FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            return {"history_path": str(output_dir / "history.json"), "best_metrics": {"best_recon": 0.0, "best_gain": 0.0}}

    monkeypatch.setattr(train_module, "SS3DMPriorTrainer", _FakeTrainer)

    exit_code = train_module.main(
        [
            "--model_config",
            str(model_config_path),
            "--train_config",
            str(train_config_path),
            "--patch_cache_dir",
            str(patch_cache_dir),
            "--split_config",
            str(tmp_path / "split.yaml"),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    train_config = captured["train_config"]
    assert isinstance(train_config, dict)
    assert train_config["wandb_enable"] is False
    assert train_config["wandb_mode"] == "disabled"
    assert train_config["wandb_project"] == "cfg_project"


def test_train_entrypoint_cli_wandb_mode_overrides_enable_and_mode(tmp_path: Path, monkeypatch) -> None:
    train_config_path = tmp_path / "train.yaml"
    model_config_path = tmp_path / "model.yaml"
    patch_cache_dir = tmp_path / "patch_cache"
    output_dir = tmp_path / "output"
    patch_cache_dir.mkdir()

    captured: dict[str, object] = {}

    monkeypatch.setattr(train_module, "load_yaml", _make_load_yaml(train_config_path, model_config_path))
    monkeypatch.setattr(train_module, "dump_yaml", lambda *_args, **_kwargs: None)

    class _FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            return {"history_path": str(output_dir / "history.json"), "best_metrics": {"best_recon": 0.0, "best_gain": 0.0}}

    monkeypatch.setattr(train_module, "SS3DMPriorTrainer", _FakeTrainer)

    exit_code = train_module.main(
        [
            "--model_config",
            str(model_config_path),
            "--train_config",
            str(train_config_path),
            "--patch_cache_dir",
            str(patch_cache_dir),
            "--split_config",
            str(tmp_path / "split.yaml"),
            "--output_dir",
            str(output_dir),
            "--wandb_project",
            "cli_project",
            "--wandb_mode",
            "offline",
        ]
    )

    assert exit_code == 0
    train_config = captured["train_config"]
    assert isinstance(train_config, dict)
    assert train_config["wandb_enable"] is True
    assert train_config["wandb_mode"] == "offline"
    assert train_config["wandb_project"] == "cli_project"
