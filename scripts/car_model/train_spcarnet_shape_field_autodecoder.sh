#!/usr/bin/env bash
# SP-CarNet Stage 2 — shape-field auto-decoder launcher.
#
# Pairs with:
#   configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml
#   configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"

MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml}"
OBJECT_INDEX="${OBJECT_INDEX:-$ROOT/outputs/carnet/spcarnet/object_index_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/carnet/spcarnet/autodecoder_v2}"
RUN_NAME="${RUN_NAME:-spcarnet_autodecoder_v2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-5}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"

mkdir -p "$OUTPUT_DIR/logs"
echo "[launch] gpu=$GPU output=$OUTPUT_DIR object_index=$OBJECT_INDEX" >&2

CUDA_VISIBLE_DEVICES="$GPU" \
WANDB_MODE="$WANDB_MODE" \
WANDB_PROJECT="$WANDB_PROJECT" \
  "$PYTHON_BIN" -m ss3dm_prior.training.spcarnet_autodecoder_cli \
    --model_config "$MODEL_CONFIG" \
    --train_config "$TRAIN_CONFIG" \
    --object_index "$OBJECT_INDEX" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --device "$DEVICE"
