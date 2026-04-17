#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior_ablations}"
SUITE_NAME="${SUITE_NAME:-v2_ablation_suite}"
WANDB_MODE="${WANDB_MODE:-disabled}"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-$ROOT/outputs/ss3dm_prior/teacher_patch_cache_v2_debug}"
EXTRA_ARGS=()

if [[ "${DEBUG_SYNTHETIC:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--debug_synthetic)
else
  EXTRA_ARGS+=(--patch_cache_dir "$PATCH_CACHE_DIR")
fi

if [[ -n "${LIDAR_ONLY_PATCH_CACHE_DIR:-}" ]]; then
  EXTRA_ARGS+=(--include_optional_lidar_only --lidar_only_patch_cache_dir "$LIDAR_ONLY_PATCH_CACHE_DIR")
fi

python -m ss3dm_prior.tools.run_ablation_suite \
  --output_dir "$OUTPUT_DIR" \
  --suite_name "$SUITE_NAME" \
  --wandb_mode "$WANDB_MODE" \
  "${EXTRA_ARGS[@]}"
