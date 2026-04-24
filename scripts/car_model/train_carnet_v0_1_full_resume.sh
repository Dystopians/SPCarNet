#!/usr/bin/env bash
# CarNet_v0.1 FULL warm-restart resume launcher.
#
# Loads outputs/carnet/v0_1/full/checkpoints/last.pt (epoch 44) and trains
# 30 more epochs at lr=1e-4 (cosine) into a separate output directory so the
# original run's artifacts stay intact.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_1/model_carnet_v0_1_full.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/carnet_v0_1/train_carnet_v0_1_full_resume.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$ROOT/outputs/carnet/v0_1/full/checkpoints/last.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0_1}"
RUN_NAME="${RUN_NAME:-carnet_v0_1_full_resume}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/full_resume}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_1}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_PATH="${LOG_PATH:-$OUTPUT_ROOT/logs/full_resume.log}"

echo "[resume] gpu=$GPU checkpoint=$RESUME_CHECKPOINT output_dir=$OUTPUT_DIR" >&2

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
    --resume "$RESUME_CHECKPOINT" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --device "$DEVICE"
