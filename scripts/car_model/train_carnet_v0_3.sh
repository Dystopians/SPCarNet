#!/usr/bin/env bash
# CarNet_v0.3 launcher — capacity probe (~85M params, 2.3× v0.2).
# Uses the same meshfleet_car_cache_v5 data stack as v0.2.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_3/model_carnet_v0_3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_3/train_carnet_v0_3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0_3}"
RUN_NAME="${RUN_NAME:-carnet_v0_3}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/full}"
WANDB_MODE="${WANDB_MODE:-online}"
# Share the v0.2 wandb project so v0.2 and v0.3 runs stack in one workspace.
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_2}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_PATH="${LOG_PATH:-$OUTPUT_ROOT/logs/carnet_v0_3.log}"

echo "[launch] gpu=$GPU output_dir=$OUTPUT_DIR cache=$CACHE_DIR" >&2

CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -m ss3dm_prior.train \
    --data_config "$DATA_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --train_config "$TRAIN_CONFIG" \
    --manifest_path "$MANIFEST_PATH" \
    --patch_cache_dir "$CACHE_DIR" \
    --split_config "$SPLIT_CONFIG" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --device "$DEVICE"
