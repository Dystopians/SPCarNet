#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior/ablations}"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-$ROOT/outputs/ss3dm_prior/teacher_patch_cache_v3_debug}"
MULTISCALE_PATCH_CACHE_DIR="${MULTISCALE_PATCH_CACHE_DIR:-$PATCH_CACHE_DIR}"
WANDB_MODE="${WANDB_MODE:-disabled}"

python -m ss3dm_prior.tools.run_ablation_suite \
  --output_dir "$OUTPUT_DIR" \
  --suite_name "v3_ablation_suite" \
  --data_config "$ROOT/configs/ss3dm_prior/data_default.yaml" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/default_town_split.yaml" \
  --patch_cache_dir "$PATCH_CACHE_DIR" \
  --multiscale_patch_cache_dir "$MULTISCALE_PATCH_CACHE_DIR" \
  --manifest_path "$ROOT/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" \
  --observed_cache_dir "$ROOT/outputs/ss3dm_prior/observed_cache" \
  --town_mesh_cache_dir "$ROOT/outputs/ss3dm_prior/town_mesh_cache" \
  --strict_model_config "$ROOT/configs/ss3dm_prior/model_v7_gain.yaml" \
  --strict_train_config "$ROOT/configs/ss3dm_prior/train_v9_strict_control.yaml" \
  --wide_model_config "$ROOT/configs/ss3dm_prior/model_v9_wide.yaml" \
  --wide_train_config "$ROOT/configs/ss3dm_prior/train_v9_wide.yaml" \
  --crossattn_model_config "$ROOT/configs/ss3dm_prior/model_v10_crossattn.yaml" \
  --crossattn_train_config "$ROOT/configs/ss3dm_prior/train_v10_crossattn.yaml" \
  --wandb_mode "$WANDB_MODE"
