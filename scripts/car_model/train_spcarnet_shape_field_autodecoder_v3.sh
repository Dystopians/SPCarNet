#!/usr/bin/env bash
# SP-CarNet Stage 2 v3 — bigger-decoder retrain launcher.
#
# Pairs with:
#   configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v3.yaml
#   configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v3.yaml
#
# v3 deltas vs v2:
#   latent_dim 256 -> 512, hidden_dim 384 -> 768, depth 6 -> 8
#
# Forwards to the canonical launcher with v3 env overrides so launch logic stays
# in one place. v1/v2 outputs untouched.

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"

export MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v3.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v3.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/carnet/spcarnet/autodecoder_v3}"
export RUN_NAME="${RUN_NAME:-spcarnet_autodecoder_v3}"
export GPU="${GPU:-5}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-spcarnet}"

exec bash "$ROOT/scripts/car_model/train_spcarnet_shape_field_autodecoder.sh"
