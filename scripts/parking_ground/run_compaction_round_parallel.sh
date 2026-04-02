#!/usr/bin/env bash
set -euo pipefail

# Compaction-round acceptance suite (parallel).
# Cases:
# 1) Baseline
# 2) PRISM-GeoGateFix
# 3) PRISM-GeoGateFixKeep
# 4) PRISM-GeoGateCompact
#
# Defaults:
# - SCENE_PATH=../parking_phone_tiny_anonymized/colmap_undistorted_fix
# - SPLIT_FILE=../parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json
# - fallback auto-detection kept for future variants
# - WandB enabled by default
# - Requires at least 3 GPUs; uses 4 if available, otherwise runs baseline sequentially

SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_ROOT}}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
  echo "PROJECT_ROOT is not a git repo: ${PROJECT_ROOT}"
  exit 1
fi
cd "${PROJECT_ROOT}"

SCENE_PATH="${SCENE_PATH:-../parking_phone_tiny_anonymized/colmap_undistorted_fix}"
MODEL_ROOT="${MODEL_ROOT:-./models}"
RUN_TAG="${RUN_TAG:-parking_phone_tiny_compaction_round}"
ITERATIONS="${ITERATIONS:-30000}"
DEFAULT_SPLIT_FILE="../parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-mesh-splatting-prune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-parking_phone_tiny_compaction_round}"
WANDB_SCALAR_LOG_INTERVAL="${WANDB_SCALAR_LOG_INTERVAL:-10}"
WANDB_IMAGE_LOG_INTERVAL="${WANDB_IMAGE_LOG_INTERVAL:-5000}"
WANDB_DISABLE_FIXED_VIEWS="${WANDB_DISABLE_FIXED_VIEWS:-1}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-42000}"

if [[ -z "${SPLIT_FILE:-}" ]]; then
  CANDIDATE_SPLITS=(
    "${DEFAULT_SPLIT_FILE}"
    "${SCENE_PATH}/split_file.json"
    "${SCENE_PATH}/split.json"
    "${SCENE_PATH}/train_test_split.json"
    "${SCENE_PATH}/splits/split_file.json"
    "${SCENE_PATH}/splits/train_test_split.json"
    "${SCENE_PATH}/sparse/0/split_outoftrain_v1.json"
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
    | head -n 4 \
    | awk '{print $2}'
)

if [[ "${#BEST_GPUS[@]}" -lt 3 ]]; then
  echo "Failed to pick at least 3 GPUs with free_mem >= ${MIN_FREE_GPU_MEM_MB} MB."
  exit 1
fi

echo "[CompactRound] Selected GPUs: ${BEST_GPUS[*]}"
echo "[CompactRound] MIN_FREE_GPU_MEM_MB=${MIN_FREE_GPU_MEM_MB}"
echo "[CompactRound] SCENE_PATH=${SCENE_PATH}"
echo "[CompactRound] SPLIT_FILE=${SPLIT_FILE}"
echo "[CompactRound] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[CompactRound] MODEL_ROOT=${MODEL_ROOT}"
echo "[CompactRound] RUN_TAG=${RUN_TAG}"

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
  echo "[CompactRound] Launch case=${case_name} gpu=${gpu_id} log=${log_path}" >&2
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${wandb_name}" \
    bash scripts/parking_ground/run_case.sh "${case_name}" > "${log_path}" 2>&1 &
  LAST_PID="$!"
}

launch_case "${BEST_GPUS[0]}" "prism_geogate_fix" "${RUN_TAG}_prism_geogate_fix"
P1="${LAST_PID}"
launch_case "${BEST_GPUS[1]}" "prism_geogate_fix_keep" "${RUN_TAG}_prism_geogate_fix_keep"
P2="${LAST_PID}"
launch_case "${BEST_GPUS[2]}" "prism_geogate_compact" "${RUN_TAG}_prism_geogate_compact"
P3="${LAST_PID}"
P4=""
if [[ "${#BEST_GPUS[@]}" -ge 4 ]]; then
  launch_case "${BEST_GPUS[3]}" "baseline" "${RUN_TAG}_baseline"
  P4="${LAST_PID}"
fi

echo "[CompactRound] PIDs: ${P1} ${P2} ${P3} ${P4}"

FAIL=0
for pid in "${P1}" "${P2}" "${P3}" "${P4}"; do
  if [[ -z "${pid}" ]]; then
    continue
  fi
  if ! wait "${pid}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" == "1" ]]; then
  echo "[CompactRound] Some training jobs failed. Check logs under ${MODEL_ROOT}/*.log"
  exit 1
fi

if [[ "${#BEST_GPUS[@]}" -lt 4 ]]; then
  echo "[CompactRound] Running baseline sequentially on GPU=${BEST_GPUS[0]}"
  CUDA_VISIBLE_DEVICES="${BEST_GPUS[0]}" \
  EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_baseline" \
    bash scripts/parking_ground/run_case.sh baseline > "${MODEL_ROOT}/${RUN_TAG}_baseline.log" 2>&1
fi

echo "[CompactRound] Training finished. Running benchmark..."
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "${SCENE_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --run Baseline="${MODEL_ROOT}/${RUN_TAG}_baseline" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep" \
  --run PRISM-GeoGateCompact="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_compact"

LATEST_BENCH_DIR="$(ls -dt benchmarks/prism_parking_ground/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BENCH_DIR}" ]]; then
  echo "[CompactRound] Benchmark output not found."
  exit 1
fi

echo "[CompactRound] Generating qualitative panels..."
python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep" \
  --run PRISM-GeoGateCompact="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_compact"

echo "[CompactRound] Done."
echo "[CompactRound] Results: ${LATEST_BENCH_DIR}"
