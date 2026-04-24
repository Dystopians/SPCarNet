#!/usr/bin/env bash
# CarNet_v0.6 evaluation — mirror v0.4 pattern against v5 cache.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/carnet/v0_6/full/checkpoints/best_composite.pt}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/outputs/carnet/v0_6/eval}"
EVAL_NAME="${EVAL_NAME:-carnet_v0_6_eval}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_2}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-7}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"

mkdir -p "$EVAL_ROOT"

CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -m ss3dm_prior.eval \
    --checkpoint "$CHECKPOINT" \
    --manifest_path "$MANIFEST_PATH" \
    --patch_cache_dir "$CACHE_DIR" \
    --split_config "$SPLIT_CONFIG" \
    --output_dir "$EVAL_ROOT" \
    --eval_name "$EVAL_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --device "$DEVICE"
