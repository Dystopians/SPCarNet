#!/usr/bin/env bash
set -euo pipefail

# End-to-end parking dataset practice suite:
# - baseline
# - previous grounding method
# - full PRISM
# - full PRISM + optional ground protect
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
WANDB_GROUP="${WANDB_GROUP:-parking_practice}"
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

echo "[Suite] 1/4 Baseline (geometry-first, no PRISM)"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_baseline" \
  bash scripts/parking_ground/run_geom_first_no_prism.sh

echo "[Suite] 2/4 Previous grounding method"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_grounding" \
  bash scripts/parking_ground/run_geom_first_grounding.sh

echo "[Suite] 3/4 Full PRISM"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism" \
  bash scripts/parking_ground/run_geom_first_full_prism.sh

echo "[Suite] 4/4 Full PRISM + optional ground protect"
EXTRA_ARGS="${COMMON_WANDB[*]} --wandb_name ${RUN_TAG}_prism_ground_protect" \
  bash scripts/parking_ground/run_geom_first_full_prism_ground_protect.sh

echo "[Suite] Quantitative fair benchmark"
python scripts/parking_ground/benchmark_prism_runs.py \
  --repo_root . \
  --scene_path "${SCENE_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --run baseline="${MODEL_ROOT}/${RUN_TAG}_geom_first_no_prism" \
  --run grounding="${MODEL_ROOT}/${RUN_TAG}_geom_first_grounding" \
  --run prism="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism" \
  --run prism_ground="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism_ground_protect"

# Find latest benchmark directory for qualitative output colocation.
LATEST_BENCH_DIR="$(ls -dt benchmarks/prism_parking_ground/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_BENCH_DIR}" ]]; then
  LATEST_BENCH_DIR="benchmarks/prism_parking_ground/manual_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LATEST_BENCH_DIR}"
fi

echo "[Suite] Qualitative side-by-side panels"
python scripts/parking_ground/make_qualitative_panels.py \
  --output_dir "${LATEST_BENCH_DIR}" \
  --run baseline="${MODEL_ROOT}/${RUN_TAG}_geom_first_no_prism" \
  --run grounding="${MODEL_ROOT}/${RUN_TAG}_geom_first_grounding" \
  --run prism="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism" \
  --run prism_ground="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism_ground_protect"

echo "[Suite] Done."
echo "[Suite] Quantitative JSON/MD + qualitative panels are under: ${LATEST_BENCH_DIR}"
