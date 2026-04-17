#!/usr/bin/env bash
set -euo pipefail
ROOT="/data2/peilincai/mesh-splatting"
CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/ss3dm_prior_car/train_v8_car_novis}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-ss3dm_prior_car}"

CUDA_VISIBLE_DEVICES=9 python -m ss3dm_prior.train \
  --data_config "$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml" \
  --model_config "$ROOT/configs/ss3dm_prior/model_v8_car_novis.yaml" \
  --train_config "$ROOT/configs/ss3dm_prior/train_v8_car.yaml" \
  --manifest_path "$MANIFEST_PATH" \
  --patch_cache_dir "$CACHE_DIR" \
  --split_config "$SPLIT_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "meshfleet_car_v2_novis" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "$WANDB_MODE"
