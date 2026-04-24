#!/usr/bin/env bash
# CarNet_v0.1 launcher — two-variant head-to-head.
#
#   det  : v11 backbone only (no flow, no symmetry, no RAG).
#   full : flow + symmetry + RAG stacked on the same backbone.
#
# Relative to v0: AMP disabled, model size reduced ~45%, nearest-neighbour
# L1 supervision added, 5-epoch recon-only warmup with tripled chamfer/nn
# weights. See docs/car_model/CarNet_v0_update_log.md for the diagnosis
# of the v0 identity-collapse + GradScaler failures that drove v0.1.
#
# Prerequisites:
#   - Cache format v3 at $CACHE_DIR (built already for v0; unchanged here).
#   - Previous v0 runs stopped (this script does not kill them).
#   - `wandb login` completed.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0_1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0_1}"
DEVICE="${DEVICE:-cuda}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"

# Two-variant GPU assignment — override with env vars if needed.
# Default lands on two lightly-loaded cards and avoids GPU 4 (another user).
GPU_DET="${GPU_DET:-0}"
GPU_FULL="${GPU_FULL:-1}"

mkdir -p "$OUTPUT_ROOT/logs"

launch_variant() {
  local pid_var_name="$1"
  local gpu="$2"
  local variant="$3"
  local run_name="carnet_v0_1_${variant}"
  local model_cfg="$ROOT/configs/ss3dm_prior/carnet_v0_1/model_carnet_v0_1_${variant}.yaml"
  local train_cfg="$ROOT/configs/ss3dm_prior/carnet_v0_1/train_carnet_v0_1_${variant}.yaml"
  local output_dir="$OUTPUT_ROOT/${variant}"
  local log_path="$OUTPUT_ROOT/logs/${variant}.log"

  if [ ! -f "$model_cfg" ] || [ ! -f "$train_cfg" ]; then
    echo "[skip ${variant}] missing config(s):" >&2
    [ ! -f "$model_cfg" ] && echo "  $model_cfg" >&2
    [ ! -f "$train_cfg" ] && echo "  $train_cfg" >&2
    return 0
  fi

  echo "[launch] variant=${variant} gpu=${gpu} output_dir=${output_dir}" >&2
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m ss3dm_prior.train \
      --data_config "$DATA_CONFIG" \
      --model_config "$model_cfg" \
      --train_config "$train_cfg" \
      --manifest_path "$MANIFEST_PATH" \
      --patch_cache_dir "$CACHE_DIR" \
      --split_config "$SPLIT_CONFIG" \
      --run_name "$run_name" \
      --output_dir "$output_dir" \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_mode "$WANDB_MODE" \
      --device "$DEVICE" \
      > "$log_path" 2>&1 &
  printf -v "$pid_var_name" '%s' "$!"
}

launch_variant PID_DET  "$GPU_DET"  "det"
launch_variant PID_FULL "$GPU_FULL" "full"

echo "[pids]  det=${PID_DET:-skip}  full=${PID_FULL:-skip}"
echo "[logs]  tail -f $OUTPUT_ROOT/logs/<variant>.log"
echo "[wandb] project=$WANDB_PROJECT  mode=$WANDB_MODE"

for pid_var in PID_DET PID_FULL; do
  pid=${!pid_var:-}
  if [ -n "$pid" ]; then
    wait "$pid" || echo "[exit] ${pid_var}=${pid} returned non-zero" >&2
  fi
done

echo "[done] CarNet_v0.1 ablation finished; outputs under $OUTPUT_ROOT"
