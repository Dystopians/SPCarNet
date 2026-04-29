#!/usr/bin/env bash
# CarNet_v0.8.2 — v0.8.1 minus the corrupted_noise warm-start.
# Pure-noise init, EdgeConv k=16, K=8 Euler steps. Isolation experiment
# to determine if warm-start was the v0.8.1 regression's single cause.
# See configs/ss3dm_prior/carnet_v0_8_2/model_carnet_v0_8_2.yaml.

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v5}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_8_2/model_carnet_v0_8_2.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_8_2/train_carnet_v0_8_2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0_8_2}"
RUN_NAME="${RUN_NAME:-carnet_v0_8_2}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/full}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_2}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-5}"
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
