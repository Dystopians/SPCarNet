#!/usr/bin/env bash
# CarNet_v0.8.1 — v0.8 + corrupted warm-start + EdgeConv + K=8.
# See configs/ss3dm_prior/carnet_v0_8_1/model_carnet_v0_8_1.yaml for rationale.

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_8_1/model_carnet_v0_8_1.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_8_1/train_carnet_v0_8_1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0_8_1}"
RUN_NAME="${RUN_NAME:-carnet_v0_8_1}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/full}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_2}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"

mkdir -p "$OUTPUT_ROOT/logs"
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
