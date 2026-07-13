#!/usr/bin/env bash
# SP-CarNet Stage 2 v4 — v3-capacity retrain with normal-band boundary supervision.

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"

export MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/carnet/spcarnet/autodecoder_v4_band}"
export RUN_NAME="${RUN_NAME:-spcarnet_autodecoder_v4_band}"
export GPU="${GPU:-2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-spcarnet}"

exec bash "$ROOT/scripts/car_model/train_spcarnet_shape_field_autodecoder.sh"
