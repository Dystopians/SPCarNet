#!/usr/bin/env bash
# CarNet_v0.1 FULL evaluation — runs ss3dm_prior.eval against the test split of
# meshfleet_car_cache_v4. In addition to point-cloud triptychs, the eval entry
# renders photorealistic textured-mesh triptychs via render_textured_mesh_panels
# (requires the per-patch source_mesh_path to be reachable).

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
CHECKPOINT="${CHECKPOINT:-$ROOT/outputs/carnet/v0_1/full/checkpoints/best_composite.pt}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/outputs/carnet/v0_1/eval}"
EVAL_NAME="${EVAL_NAME:-carnet_v0_1_full_eval}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_1_eval}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
# Base conda env's torch is built for CUDA 13 but system driver is 12.4.
# mesh_splatting micromamba env carries a CUDA 12.6 build that matches.
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
