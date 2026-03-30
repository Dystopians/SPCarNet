#!/usr/bin/env bash
set -euo pipefail

# End-to-end parking dataset PRISM geogate suite:
# - PRISM-GeoGateFix
# - PRISM-LatePrune
# - PRISM-GeoGateFixKeep
# plus fair quantitative benchmark + qualitative panels.

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
WANDB_GROUP="${WANDB_GROUP:-parking_prism_geogate}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_SCALAR_LOG_INTERVAL="${WANDB_SCALAR_LOG_INTERVAL:-10}"
WANDB_IMAGE_LOG_INTERVAL="${WANDB_IMAGE_LOG_INTERVAL:-5000}"
WANDB_DISABLE_FIXED_VIEWS="${WANDB_DISABLE_FIXED_VIEWS:-0}"

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

export MODEL_ROOT RUN_TAG ITERATIONS

echo "[Suite] 1/3 PRISM-GeoGateFix"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_geogate_fix" \
  bash scripts/parking_ground/run_case.sh prism_geogate_fix

echo "[Suite] 2/3 PRISM-LatePrune"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_late_prune" \
  bash scripts/parking_ground/run_case.sh prism_late_prune

echo "[Suite] 3/3 PRISM-GeoGateFixKeep"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_geogate_fix_keep" \
  bash scripts/parking_ground/run_case.sh prism_geogate_fix_keep

echo "[Suite] Quantitative fair benchmark"
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "${SCENE_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-LatePrune="${MODEL_ROOT}/${RUN_TAG}_prism_late_prune" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep"

# Find latest benchmark directory for qualitative output colocation.
LATEST_BENCH_DIR="$(ls -dt benchmarks/prism_parking_ground/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BENCH_DIR}" ]]; then
  LATEST_BENCH_DIR="benchmarks/prism_parking_ground/manual_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LATEST_BENCH_DIR}"
fi

echo "[Suite] Qualitative side-by-side panels"
python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run PRISM-GeoGateFix="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix" \
  --run PRISM-LatePrune="${MODEL_ROOT}/${RUN_TAG}_prism_late_prune" \
  --run PRISM-GeoGateFixKeep="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep"

echo "[Suite] Done."
echo "[Suite] Quantitative JSON/MD + qualitative panels are under: ${LATEST_BENCH_DIR}"
