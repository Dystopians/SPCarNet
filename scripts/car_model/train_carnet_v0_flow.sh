#!/usr/bin/env bash
# CarNet_v0 Phase 1 launcher — flow-matching only (A1).
#
# Runs a single training variant `carnet_v0_flow` on one GPU. A2 (symmetry)
# and A3 (retrieval) are not yet wired; once Phase 2/4 land this script will
# extend to a trio (flow / flow_sym / flow_full) along the same lines as
# `train_meshfleet_car_v4_trio.sh`.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"

VARIANT="${VARIANT:-flow}"
MODEL_CFG="$ROOT/configs/ss3dm_prior/carnet_v0/model_carnet_v0_${VARIANT}.yaml"
TRAIN_CFG="$ROOT/configs/ss3dm_prior/carnet_v0/train_carnet_v0_${VARIANT}.yaml"
OUTPUT_DIR="$OUTPUT_ROOT/${VARIANT}"
RUN_NAME="carnet_v0_${VARIANT}"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_PATH="$OUTPUT_ROOT/logs/${VARIANT}.log"

if [ ! -f "$MODEL_CFG" ] || [ ! -f "$TRAIN_CFG" ]; then
  echo "missing config(s):" >&2
  [ ! -f "$MODEL_CFG" ] && echo "  $MODEL_CFG" >&2
  [ ! -f "$TRAIN_CFG" ] && echo "  $TRAIN_CFG" >&2
  exit 2
fi

echo "[launch] variant=${VARIANT} gpu=${GPU} output_dir=${OUTPUT_DIR}" >&2
CUDA_VISIBLE_DEVICES="$GPU" \
  python -m ss3dm_prior.train \
    --data_config "$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml" \
    --model_config "$MODEL_CFG" \
    --train_config "$TRAIN_CFG" \
    --manifest_path "$MANIFEST_PATH" \
    --patch_cache_dir "$CACHE_DIR" \
    --split_config "$SPLIT_CONFIG" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --device "$DEVICE" \
    > "$LOG_PATH" 2>&1 &
PID=$!

echo "[pid] ${PID}"
echo "[log] ${LOG_PATH}"
wait "$PID"
