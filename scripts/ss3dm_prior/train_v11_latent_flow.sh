#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-$ROOT/outputs/ss3dm_prior/teacher_patch_cache_v3_debug}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior/train_v11_latent_flow}"
WANDB_MODE="${WANDB_MODE:-offline}"

python -m ss3dm_prior.train \
  --data_config "$ROOT/configs/ss3dm_prior/data_default.yaml" \
  --model_config "$ROOT/configs/ss3dm_prior/model_v11_latent_flow.yaml" \
  --train_config "$ROOT/configs/ss3dm_prior/train_v11_latent_flow.yaml" \
  --manifest_path "$ROOT/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" \
  --observed_cache_dir "$ROOT/outputs/ss3dm_prior/observed_cache" \
  --town_mesh_cache_dir "$ROOT/outputs/ss3dm_prior/town_mesh_cache" \
  --patch_cache_dir "$PATCH_CACHE_DIR" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/default_town_split.yaml" \
  --run_name "ss3dm_prior_v11_latent_flow" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "ss3dm_prior" \
  --wandb_mode "$WANDB_MODE"
