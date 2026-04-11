#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-$ROOT/outputs/ss3dm_prior/teacher_patch_cache_debug_val}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior/train_debug}"
TMP_TRAIN_CONFIG="$(mktemp)"
trap 'rm -f "$TMP_TRAIN_CONFIG"' EXIT

cat > "$TMP_TRAIN_CONFIG" <<'EOF'
train:
  seed: 0
  epochs: 2
  batch_size: 2
  num_workers: 0
  lr: 0.001
  weight_decay: 0.0001
  amp: false
  log_interval: 1
  val_interval: 1
  save_interval: 1
  wandb_enable: true
  wandb_project: ss3dm_prior
  wandb_mode: disabled
  max_visualization_examples: 1
  fixed_visualization_patch_ids: []
  debug_use_all_patches_for_train_val: true
  allow_split_fallback: true
  debug_val_fraction: 0.25
EOF

python -m ss3dm_prior.train \
  --data_config "$ROOT/configs/ss3dm_prior/data_default.yaml" \
  --model_config "$ROOT/configs/ss3dm_prior/model_default.yaml" \
  --train_config "$TMP_TRAIN_CONFIG" \
  --manifest_path "$ROOT/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" \
  --observed_cache_dir "$ROOT/outputs/ss3dm_prior/observed_cache_debug" \
  --town_mesh_cache_dir "$ROOT/outputs/ss3dm_prior/town_mesh_cache_smoke" \
  --patch_cache_dir "$PATCH_CACHE_DIR" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/debug_town_split.yaml" \
  --run_name "ss3dm_prior_debug" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "ss3dm_prior" \
  --wandb_mode "disabled"
