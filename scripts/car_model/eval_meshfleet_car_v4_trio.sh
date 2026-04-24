#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT/outputs/ss3dm_prior_car/v4_ablations}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/outputs/ss3dm_prior_car/v4_eval}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-car_model_eval_v4}"
DEVICE="${DEVICE:-cuda}"

python -m ss3dm_prior.eval \
  --checkpoint "$TRAIN_ROOT/v3_5_baseline_v4/checkpoints/best_composite.pt" \
  --manifest_path "$MANIFEST_PATH" \
  --patch_cache_dir "$CACHE_DIR" \
  --split_config "$SPLIT_CONFIG" \
  --output_dir "$EVAL_ROOT" \
  --eval_name "v3_5_baseline_v4_eval" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE" \
  --device "$DEVICE"

python -m ss3dm_prior.eval \
  --checkpoint "$TRAIN_ROOT/v3_5_occ_v4/checkpoints/best_composite.pt" \
  --manifest_path "$MANIFEST_PATH" \
  --patch_cache_dir "$CACHE_DIR" \
  --split_config "$SPLIT_CONFIG" \
  --output_dir "$EVAL_ROOT" \
  --eval_name "v3_5_occ_v4_eval" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE" \
  --device "$DEVICE"

python -m ss3dm_prior.eval \
  --checkpoint "$TRAIN_ROOT/v3_5_occ_vq_v4/checkpoints/best_composite.pt" \
  --manifest_path "$MANIFEST_PATH" \
  --patch_cache_dir "$CACHE_DIR" \
  --split_config "$SPLIT_CONFIG" \
  --output_dir "$EVAL_ROOT" \
  --eval_name "v3_5_occ_vq_v4_eval" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE" \
  --device "$DEVICE"
