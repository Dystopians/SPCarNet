#!/usr/bin/env bash
set -euo pipefail

# Compaction-round acceptance suite (sequential).
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

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

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

mkdir -p "${MODEL_ROOT}"

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

echo "[CompactSuite] SCENE_PATH=${SCENE_PATH}"
echo "[CompactSuite] SPLIT_FILE=${SPLIT_FILE}"
echo "[CompactSuite] MODEL_ROOT=${MODEL_ROOT}"
echo "[CompactSuite] RUN_TAG=${RUN_TAG}"

echo "[CompactSuite] 1/4 Baseline"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_baseline" \
  bash scripts/parking_ground/run_case.sh baseline

echo "[CompactSuite] 2/4 PRISM-GeoGateFix"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_geogate_fix" \
  bash scripts/parking_ground/run_case.sh prism_geogate_fix

echo "[CompactSuite] 3/4 PRISM-GeoGateFixKeep"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_geogate_fix_keep" \
  bash scripts/parking_ground/run_case.sh prism_geogate_fix_keep

echo "[CompactSuite] 4/4 PRISM-GeoGateCompact"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_geogate_compact" \
  bash scripts/parking_ground/run_case.sh prism_geogate_compact

echo "[CompactSuite] Running benchmark..."
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
  echo "[CompactSuite] Benchmark output not found."
  exit 1
fi

echo "[CompactSuite] Generating qualitative panels..."
python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep" \
  --run PRISM-GeoGateCompact="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_compact"

echo "[CompactSuite] Done."
echo "[CompactSuite] Results: ${LATEST_BENCH_DIR}"
