#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/ss3dm_prior_car/v4_ablations}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-ss3dm_prior_car}"
DEVICE="${DEVICE:-cuda}"

GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_OCC="${GPU_OCC:-1}"
GPU_OCC_VQ="${GPU_OCC_VQ:-2}"

mkdir -p "$OUTPUT_ROOT/logs"

launch_variant() {
  local pid_var_name="$1"
  local gpu="$2"
  local variant="$3"
  local model_cfg="$4"
  local train_cfg="$5"
  local run_name="car_model_${variant}_v4"
  local output_dir="$OUTPUT_ROOT/${variant}_v4"
  local log_path="$OUTPUT_ROOT/logs/${variant}_v4.log"

  echo "[launch] variant=${variant} gpu=${gpu} output_dir=${output_dir}" >&2
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m ss3dm_prior.train \
      --data_config "$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml" \
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

launch_variant \
  PID_BASELINE \
  "$GPU_BASELINE" \
  "v3_5_baseline" \
  "$ROOT/configs/ss3dm_prior/model_v3_5_car_baseline.yaml" \
  "$ROOT/configs/ss3dm_prior/train_v3_5_car_baseline.yaml"

launch_variant \
  PID_OCC \
  "$GPU_OCC" \
  "v3_5_occ" \
  "$ROOT/configs/ss3dm_prior/model_v3_5_car_occ.yaml" \
  "$ROOT/configs/ss3dm_prior/train_v3_5_car_occ.yaml"

launch_variant \
  PID_OCC_VQ \
  "$GPU_OCC_VQ" \
  "v3_5_occ_vq" \
  "$ROOT/configs/ss3dm_prior/model_v3_5_car_occ_vq.yaml" \
  "$ROOT/configs/ss3dm_prior/train_v3_5_car_occ_vq.yaml"

echo "[pids] baseline=${PID_BASELINE} occ=${PID_OCC} occ_vq=${PID_OCC_VQ}"
echo "[logs] $OUTPUT_ROOT/logs/v3_5_baseline_v4.log"
echo "[logs] $OUTPUT_ROOT/logs/v3_5_occ_v4.log"
echo "[logs] $OUTPUT_ROOT/logs/v3_5_occ_vq_v4.log"

wait "$PID_BASELINE"
wait "$PID_OCC"
wait "$PID_OCC_VQ"
