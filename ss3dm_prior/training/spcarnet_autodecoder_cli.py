"""SP-CarNet Stage 2 auto-decoder training CLI.

Thin wrapper around :class:`ShapeFieldAutoDecoderTrainer`. Use:

  python -m ss3dm_prior.training.spcarnet_autodecoder_cli \
      --model_config configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml \
      --train_config configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml

The full training run is intentionally not launched from the smoke test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ss3dm_prior.training.spcarnet_autodecoder import (
    ShapeFieldAutoDecoderTrainer,
    load_configs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--object_index", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=0, help="Cap step count for ad-hoc runs (0 = unlimited).")
    args = parser.parse_args(argv)

    model_cfg, loss_cfg, train_cfg = load_configs(args.model_config, args.train_config)
    if args.object_index:
        train_cfg.object_index_path = args.object_index
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.run_name:
        train_cfg.run_name = args.run_name
    if args.device:
        train_cfg.device = args.device

    out = Path(train_cfg.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolved_config.json").open("w") as f:
        json.dump(
            {
                "model": asdict(model_cfg),
                "loss": asdict(loss_cfg),
                "train": asdict(train_cfg),
            },
            f,
            indent=2,
        )

    trainer = ShapeFieldAutoDecoderTrainer(
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        train_cfg=train_cfg,
    )
    summary = trainer.fit(max_steps=args.max_steps if args.max_steps > 0 else None)
    with (out / "fit_summary.json").open("w") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
