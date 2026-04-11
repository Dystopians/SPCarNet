"""CLI entrypoint for SS3DM prior training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ss3dm_prior.engine.trainer import SS3DMPriorTrainer
from ss3dm_prior.utils.io import dump_yaml, load_yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SS3DM prior local patch denoiser with logging and checkpointing."
    )
    parser.add_argument("--data_config", default=None, help="Optional YAML data config.")
    parser.add_argument("--model_config", required=True, help="YAML model/corruption/loss config.")
    parser.add_argument("--train_config", required=True, help="YAML training config.")
    parser.add_argument("--manifest_path", default=None, help="Manifest path metadata.")
    parser.add_argument("--observed_cache_dir", default=None, help="Observed cache root metadata.")
    parser.add_argument("--town_mesh_cache_dir", default=None, help="Town mesh cache root metadata.")
    parser.add_argument("--patch_cache_dir", required=True, help="Teacher patch cache root.")
    parser.add_argument("--split_config", required=True, help="Split YAML for train/val.")
    parser.add_argument("--run_name", default="ss3dm_prior_run", help="Run name.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--wandb_project", default=None, help="wandb project override.")
    parser.add_argument("--wandb_mode", default=None, help="wandb mode override.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)

    data_config = load_yaml(args.data_config) if args.data_config else {}
    model_config = load_yaml(args.model_config)
    train_cfg_raw = load_yaml(args.train_config)
    train_config = train_cfg_raw.get("train", train_cfg_raw)
    if args.wandb_project is not None:
        train_config["wandb_project"] = args.wandb_project
    if args.wandb_mode is not None:
        train_config["wandb_mode"] = args.wandb_mode

    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    patch_index_path = patch_cache_dir / "patch_index.jsonl"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "data_config": data_config,
        "model_config": model_config,
        "train_config": train_config,
        "cli": {
            "manifest_path": args.manifest_path,
            "observed_cache_dir": args.observed_cache_dir,
            "town_mesh_cache_dir": args.town_mesh_cache_dir,
            "patch_cache_dir": str(patch_cache_dir),
            "patch_index_path": str(patch_index_path),
            "split_config": args.split_config,
            "run_name": args.run_name,
            "output_dir": str(output_dir),
            "resume": args.resume,
        },
    }
    dump_yaml(output_dir / "configs" / "resolved_data_config.yaml", data_config or {})
    dump_yaml(output_dir / "configs" / "resolved_model_config.yaml", model_config)
    dump_yaml(output_dir / "configs" / "resolved_train_config.yaml", train_config)
    (output_dir / "configs" / "resolved_run_metadata.json").write_text(
        json.dumps(resolved["cli"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    trainer = SS3DMPriorTrainer(
        model_config=model_config,
        train_config=train_config,
        patch_index_path=patch_index_path,
        split_config=args.split_config,
        output_dir=output_dir,
        run_name=args.run_name,
        run_metadata=resolved,
        resume_path=args.resume,
    )
    result = trainer.fit()
    print(f"history_path: {result['history_path']}")
    print(f"best_recon: {result['best_metrics']['best_recon']}")
    print(f"best_gain: {result['best_metrics']['best_gain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
