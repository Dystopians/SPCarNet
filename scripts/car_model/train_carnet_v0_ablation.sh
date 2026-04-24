#!/usr/bin/env bash
# CarNet_v0 ablation-matrix launcher (A1 + A2 + A3 + LiDAR).
#
# Launches six training runs in parallel on six GPUs and waits for all to
# finish. wandb is online; each run logs to project `carnet_v0` with a
# distinct run_name. Outputs land under $OUTPUT_ROOT/<variant>/.
#
# Prerequisites (user-side, run manually before this script):
#   1. Rebuild the cache to format v3 (includes symmetry + query + LiDAR-
#      ready geometry):
#         rm -rf $ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4
#         python -m ss3dm_prior.tools.build_car_mesh_patch_cache \
#             --dataset_root /data2/car_meshes/MeshFleet_TRELLIS_RECONSTRUCTED_v4 \
#             --mesh_root    /data2/car_meshes/MeshFleet_TRELLIS_RECONSTRUCTED_v4 \
#             --out_dir      $ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4 \
#             --val_fraction 0.1 \
#             --clean_sample_count 2048 --observed_sample_count 768 \
#             --normalized_radius 1.0 --observed_view_count 3 \
#             --min_visibility_cosine 0.05 --clean_visibility_cosine 0.5 \
#             --surface_query_count 512 --free_query_count 512 \
#             --unknown_query_count 256 --query_ball_radius 1.1 \
#             --surface_query_eps 0.025 --free_query_eps 0.04 \
#             --hard_negative_count 128 --num_workers 8 --seed 0 --skip_existing
#   2. Verify the cache is format v3 (see docs/car_model/CarNet_v0.md).
#   3. Confirm wandb is logged in: `wandb login`.

set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

CACHE_DIR="${CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$CACHE_DIR/split_meshfleet_car.yaml}"
MANIFEST_PATH="${MANIFEST_PATH:-$CACHE_DIR/source_mesh_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/carnet/v0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-carnet_v0}"
DEVICE="${DEVICE:-cuda}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/configs/ss3dm_prior/data_meshfleet_car.yaml}"

mkdir -p "$OUTPUT_ROOT/logs"

# Variant -> GPU assignment. Override via env vars if needed
# (e.g. `GPU_FULL=5 bash scripts/car_model/train_carnet_v0_ablation.sh`).
GPU_DET="${GPU_DET:-0}"
GPU_FLOW="${GPU_FLOW:-1}"
GPU_FLOW_SYM="${GPU_FLOW_SYM:-2}"
GPU_FLOW_RAG="${GPU_FLOW_RAG:-3}"
GPU_FULL="${GPU_FULL:-5}"
GPU_FULL_LIDAR="${GPU_FULL_LIDAR:-6}"
# GPU 4 is currently partially occupied; skip it by default.

launch_variant() {
  local pid_var_name="$1"
  local gpu="$2"
  local variant="$3"
  local run_name="carnet_v0_${variant}"
  local model_cfg="$ROOT/configs/ss3dm_prior/carnet_v0/model_carnet_v0_${variant}.yaml"
  local train_cfg="$ROOT/configs/ss3dm_prior/carnet_v0/train_carnet_v0_${variant}.yaml"
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

# ---- ablation matrix (CarNet_v0.md §6.1) --------------------------------
# carnet_v0_det         : deterministic baseline (no flow, no sym, no RAG)
# carnet_v0_flow        : flow matching only (A1)
# carnet_v0_flow_sym    : flow + symmetry      (A1 + A2)
# carnet_v0_flow_rag    : flow + retrieval     (A1 + A3)
# carnet_v0_full        : flow + sym + RAG     (A1 + A2 + A3 — headline)
# carnet_v0_full_lidar  : full stack + LiDAR-realistic corruption (D5)
launch_variant PID_DET         "$GPU_DET"         "det"
launch_variant PID_FLOW        "$GPU_FLOW"        "flow"
launch_variant PID_FLOW_SYM    "$GPU_FLOW_SYM"    "flow_sym"
launch_variant PID_FLOW_RAG    "$GPU_FLOW_RAG"    "flow_rag"
launch_variant PID_FULL        "$GPU_FULL"        "full"
launch_variant PID_FULL_LIDAR  "$GPU_FULL_LIDAR"  "full_lidar"

echo "[pids] det=${PID_DET:-skip} flow=${PID_FLOW:-skip} flow_sym=${PID_FLOW_SYM:-skip} flow_rag=${PID_FLOW_RAG:-skip} full=${PID_FULL:-skip} full_lidar=${PID_FULL_LIDAR:-skip}"
echo "[logs] tail -f $OUTPUT_ROOT/logs/<variant>.log"
echo "[wandb] project=$WANDB_PROJECT  mode=$WANDB_MODE"

for pid_var in PID_DET PID_FLOW PID_FLOW_SYM PID_FLOW_RAG PID_FULL PID_FULL_LIDAR; do
  pid=${!pid_var:-}
  if [ -n "$pid" ]; then
    wait "$pid" || echo "[exit] ${pid_var}=${pid} returned non-zero" >&2
  fi
done

echo "[done] all CarNet_v0 runs finished; outputs under $OUTPUT_ROOT"
