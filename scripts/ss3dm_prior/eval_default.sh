#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/ss3dm_prior/train_default/checkpoints/best_recon.pt}"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-$ROOT/outputs/ss3dm_prior/teacher_patch_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior_eval}"
EVAL_NAME="${EVAL_NAME:-default_eval}"
WANDB_MODE="${WANDB_MODE:-offline}"

python -m ss3dm_prior.eval \
  --checkpoint "$CHECKPOINT" \
  --manifest_path "$ROOT/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" \
  --patch_cache_dir "$PATCH_CACHE_DIR" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/default_town_split.yaml" \
  --output_dir "$OUTPUT_DIR" \
  --eval_name "$EVAL_NAME" \
  --wandb_project "ss3dm_prior_eval" \
  --wandb_mode "$WANDB_MODE"
