#!/usr/bin/env bash
set -euo pipefail

# Parallel full suite launcher:
# - auto-picks 4 least-busy GPUs
# - runs 4 training variants concurrently (one per GPU)
# - then runs fair quantitative benchmark + qualitative panels
#
# Required env:
#   SCENE_PATH
#   SPLIT_FILE
#
# Optional env:
#   MODEL_ROOT, RUN_TAG, ITERATIONS
#   WANDB_ENABLE, WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP
#   WANDB_SCALAR_LOG_INTERVAL, WANDB_IMAGE_LOG_INTERVAL, WANDB_DISABLE_FIXED_VIEWS
#   GROUND_MASK_DIR

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${SCENE_PATH:-}" ]]; then
  echo "Please set SCENE_PATH=/abs/path/to/parking_scene"
  exit 1
fi
if [[ -z "${SPLIT_FILE:-}" ]]; then
  echo "Please set SPLIT_FILE=/abs/path/to/split_file.json"
  exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-./models}"
RUN_TAG="${RUN_TAG:-parking_full_practice}"
ITERATIONS="${ITERATIONS:-30000}"
WANDB_PROJECT="${WANDB_PROJECT:-mesh-splatting}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-parking_practice}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_SCALAR_LOG_INTERVAL="${WANDB_SCALAR_LOG_INTERVAL:-10}"
WANDB_IMAGE_LOG_INTERVAL="${WANDB_IMAGE_LOG_INTERVAL:-5000}"
WANDB_DISABLE_FIXED_VIEWS="${WANDB_DISABLE_FIXED_VIEWS:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs."
  exit 1
fi

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
if [[ "${#GPU_ROWS[@]}" -lt 4 ]]; then
  echo "Need at least 4 GPUs, found ${#GPU_ROWS[@]}."
  exit 1
fi

mapfile -t BEST_GPUS < <(
  printf "%s\n" "${GPU_ROWS[@]}" \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); score=($2*100000)+$3; printf "%012d %s\n", score, $1}' \
    | sort -n \
    | head -n 4 \
    | awk '{print $2}'
)

if [[ "${#BEST_GPUS[@]}" -lt 4 ]]; then
  echo "Failed to pick 4 GPUs."
  exit 1
fi

echo "[AutoGPU] Selected GPUs: ${BEST_GPUS[*]}"

COMMON_WANDB=()
if [[ "${WANDB_ENABLE}" == "1" ]]; then
  COMMON_WANDB=(
    --enable_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_scalar_log_interval "${WANDB_SCALAR_LOG_INTERVAL}"
    --wandb_image_log_interval "${WANDB_IMAGE_LOG_INTERVAL}"
  )
  if [[ "${WANDB_DISABLE_FIXED_VIEWS}" == "1" ]]; then
    COMMON_WANDB+=(--wandb_disable_fixed_views)
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_WANDB+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

export MODEL_ROOT RUN_TAG ITERATIONS SCENE_PATH SPLIT_FILE
if [[ -n "${GROUND_MASK_DIR:-}" ]]; then
  export GROUND_MASK_DIR
fi

launch_case() {
  local gpu_id="$1"
  local case_name="$2"
  local wandb_name="$3"
  echo "[AutoGPU] Launch case=${case_name} on GPU=${gpu_id}" >&2
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${wandb_name}" \
    bash scripts/parking_ground/run_case.sh "${case_name}" \
    > "${MODEL_ROOT}/${RUN_TAG}_${case_name}.log" 2>&1 &
  LAST_PID="$!"
}

launch_case "${BEST_GPUS[0]}" "no_prism" "${RUN_TAG}_baseline"
P1="${LAST_PID}"
launch_case "${BEST_GPUS[1]}" "grounding_only" "${RUN_TAG}_grounding"
P2="${LAST_PID}"
launch_case "${BEST_GPUS[2]}" "full_prism" "${RUN_TAG}_prism"
P3="${LAST_PID}"
launch_case "${BEST_GPUS[3]}" "full_prism_ground_protect" "${RUN_TAG}_prism_ground_protect"
P4="${LAST_PID}"

echo "[AutoGPU] PIDs: ${P1} ${P2} ${P3} ${P4}"

FAIL=0
for pid in "${P1}" "${P2}" "${P3}" "${P4}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" == "1" ]]; then
  echo "[AutoGPU] One or more training jobs failed. Check logs under ${MODEL_ROOT}/*.log"
  exit 1
fi

echo "[AutoGPU] All training jobs finished. Running benchmark..."
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "${SCENE_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --run baseline="${MODEL_ROOT}/${RUN_TAG}_geom_first_no_prism" \
  --run grounding="${MODEL_ROOT}/${RUN_TAG}_geom_first_grounding" \
  --run prism="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism" \
  --run prism_ground="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism_ground_protect"

LATEST_BENCH_DIR="$(ls -dt benchmarks/prism_parking_ground/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BENCH_DIR}" ]]; then
  LATEST_BENCH_DIR="benchmarks/prism_parking_ground/manual_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LATEST_BENCH_DIR}"
fi

python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run baseline="${MODEL_ROOT}/${RUN_TAG}_geom_first_no_prism" \
  --run grounding="${MODEL_ROOT}/${RUN_TAG}_geom_first_grounding" \
  --run prism="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism" \
  --run prism_ground="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism_ground_protect"

echo "[AutoGPU] Done."
echo "[AutoGPU] Results at: ${LATEST_BENCH_DIR}"
