#!/usr/bin/env bash
set -euo pipefail

# Round-2 PRISM geogate high-density suite (parallel, 3 runs):
# 1) PRISM-GeoGateFix
# 2) PRISM-LatePrune
# 3) PRISM-GeoGateFixKeep
#
# Defaults:
# - SCENE_PATH=../parking_phone_tiny_anonymized
# - SPLIT_FILE auto-detected from common names under SCENE_PATH
# - WandB enabled by default

SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_ROOT}}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  echo "Please set PROJECT_ROOT to your prune repo root (expected: /data2/peilincai/mesh-splatting-prune)."
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  echo "PROJECT_ROOT is not a git repo: ${PROJECT_ROOT}"
  exit 1
fi
if [[ "${PROJECT_ROOT}" != "${SCRIPT_ROOT}" ]]; then
  echo "[Round2] Note: script file is under ${SCRIPT_ROOT}, but run root is ${PROJECT_ROOT}."
fi
cd "${PROJECT_ROOT}"

SCENE_PATH="${SCENE_PATH:-../parking_phone_tiny_anonymized}"
MODEL_ROOT="${MODEL_ROOT:-./models}"
RUN_TAG="${RUN_TAG:-parking_phone_tiny_geogate_round2}"
ITERATIONS="${ITERATIONS:-30000}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-mesh-splatting-prune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-parking_phone_tiny_geogate_round2}"
WANDB_SCALAR_LOG_INTERVAL="${WANDB_SCALAR_LOG_INTERVAL:-10}"
WANDB_IMAGE_LOG_INTERVAL="${WANDB_IMAGE_LOG_INTERVAL:-5000}"
WANDB_DISABLE_FIXED_VIEWS="${WANDB_DISABLE_FIXED_VIEWS:-1}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-42000}"

if [[ -z "${SPLIT_FILE:-}" ]]; then
  CANDIDATE_SPLITS=(
    "${SCENE_PATH}/split_file.json"
    "${SCENE_PATH}/split.json"
    "${SCENE_PATH}/train_test_split.json"
    "${SCENE_PATH}/splits/split_file.json"
    "${SCENE_PATH}/splits/train_test_split.json"
  )
  for p in "${CANDIDATE_SPLITS[@]}"; do
    if [[ -f "${p}" ]]; then
      SPLIT_FILE="${p}"
      break
    fi
  done
fi

if [[ ! -d "${SCENE_PATH}" ]]; then
  echo "SCENE_PATH not found: ${SCENE_PATH}"
  exit 1
fi
if [[ -z "${SPLIT_FILE:-}" || ! -f "${SPLIT_FILE}" ]]; then
  echo "SPLIT_FILE not found. Please set SPLIT_FILE explicitly."
  echo "Tried under SCENE_PATH=${SCENE_PATH}"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs."
  exit 1
fi

mkdir -p "${MODEL_ROOT}"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
if [[ "${#GPU_ROWS[@]}" -lt 3 ]]; then
  echo "Need at least 3 GPUs, found ${#GPU_ROWS[@]}."
  exit 1
fi

mapfile -t BEST_GPUS < <(
  printf "%s\n" "${GPU_ROWS[@]}" \
    | awk -F',' -v min_free="${MIN_FREE_GPU_MEM_MB}" '
      {
        gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); gsub(/ /,"",$4);
        free_mb = $4 - $3;
        if (free_mb >= min_free) {
          score = ($2*100000) + $3;
          printf "%012d %s\n", score, $1;
        }
      }' \
    | sort -n \
    | head -n 3 \
    | awk '{print $2}'
)
if [[ "${#BEST_GPUS[@]}" -lt 3 ]]; then
  echo "Failed to pick 3 GPUs with free_mem >= ${MIN_FREE_GPU_MEM_MB} MB."
  echo "Tip: lower MIN_FREE_GPU_MEM_MB (e.g., 36000) only if you understand OOM risk."
  exit 1
fi

echo "[Round2] Selected GPUs: ${BEST_GPUS[*]}"
echo "[Round2] MIN_FREE_GPU_MEM_MB=${MIN_FREE_GPU_MEM_MB}"
echo "[Round2] SCENE_PATH=${SCENE_PATH}"
echo "[Round2] SPLIT_FILE=${SPLIT_FILE}"
echo "[Round2] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[Round2] MODEL_ROOT=${MODEL_ROOT}"
echo "[Round2] RUN_TAG=${RUN_TAG}"

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

export SCENE_PATH SPLIT_FILE MODEL_ROOT RUN_TAG ITERATIONS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch_case() {
  local gpu_id="$1"
  local case_name="$2"
  local wandb_name="$3"
  local log_path="${MODEL_ROOT}/${RUN_TAG}_${case_name}.log"
  echo "[Round2] Launch case=${case_name} gpu=${gpu_id} log=${log_path}" >&2
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${wandb_name}" \
    bash scripts/parking_ground/run_case.sh "${case_name}" > "${log_path}" 2>&1 &
  LAST_PID="$!"
}

launch_case "${BEST_GPUS[0]}" "prism_geogate_fix" "${RUN_TAG}_prism_geogate_fix"
P1="${LAST_PID}"
launch_case "${BEST_GPUS[1]}" "prism_late_prune" "${RUN_TAG}_prism_late_prune"
P2="${LAST_PID}"
launch_case "${BEST_GPUS[2]}" "prism_geogate_fix_keep" "${RUN_TAG}_prism_geogate_fix_keep"
P3="${LAST_PID}"

echo "[Round2] PIDs: ${P1} ${P2} ${P3}"

FAIL=0
for pid in "${P1}" "${P2}" "${P3}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" == "1" ]]; then
  echo "[Round2] Some training jobs failed. Check logs under ${MODEL_ROOT}/*.log"
  exit 1
fi

echo "[Round2] Training finished. Running benchmark..."
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "${SCENE_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-LatePrune="${MODEL_ROOT}/${RUN_TAG}_prism_late_prune" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep"

LATEST_BENCH_DIR="$(ls -dt benchmarks/prism_parking_ground/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BENCH_DIR}" ]]; then
  echo "[Round2] Benchmark output not found."
  exit 1
fi

echo "[Round2] Generating qualitative panels..."
python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-LatePrune="${MODEL_ROOT}/${RUN_TAG}_prism_late_prune" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep"

echo "[Round2] Done."
echo "[Round2] Results: ${LATEST_BENCH_DIR}"
