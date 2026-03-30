#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for parking-ground PRISM ablations.
# Usage:
#   bash scripts/parking_ground/run_case.sh <case_name>
#
# Required env:
#   SCENE_PATH=/abs/path/to/colmap/scene
#   SPLIT_FILE=/abs/path/to/split.json
#
# Optional env:
#   MODEL_ROOT=./models
#   RUN_TAG=parking_ground
#   ITERATIONS=30000
#   IMAGE_DIR=images
#   EXTRA_ARGS="..."
#   GROUND_MASK_DIR=/abs/path/to/ground_masks (optional)
#   TEST_ITER_EVERY=1000
#   TEST_ITER_START=1000
#   TEST_ITER_END=$ITERATIONS
#   SAVE_ITERATIONS="7000 15000 16000 20000 21000 30000"

CASE_NAME="${1:-}"
if [[ -z "${CASE_NAME}" ]]; then
  echo "Missing case name. Expected one of:"
  echo "  no_prism | grounding_only | dead_only | full_prism | full_prism_ground_protect"
  exit 1
fi

if [[ -z "${SCENE_PATH:-}" ]]; then
  echo "Please set SCENE_PATH"
  exit 1
fi
if [[ -z "${SPLIT_FILE:-}" ]]; then
  echo "Please set SPLIT_FILE"
  exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-./models}"
RUN_TAG="${RUN_TAG:-parking_ground}"
ITERATIONS="${ITERATIONS:-30000}"
IMAGE_DIR="${IMAGE_DIR:-images}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TEST_ITER_EVERY="${TEST_ITER_EVERY:-1000}"
TEST_ITER_START="${TEST_ITER_START:-1000}"
TEST_ITER_END="${TEST_ITER_END:-${ITERATIONS}}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000 15000 16000 20000 21000 30000}"

mkdir -p "${MODEL_ROOT}"

TEST_ITER_LIST=()
if [[ "${TEST_ITER_EVERY}" -gt 0 ]]; then
  for ((it=TEST_ITER_START; it<=TEST_ITER_END; it+=TEST_ITER_EVERY)); do
    TEST_ITER_LIST+=("${it}")
  done
fi

SAVE_ITER_LIST=()
for token in ${SAVE_ITERATIONS}; do
  SAVE_ITER_LIST+=("${token}")
done

COMMON_ARGS=(
  -s "${SCENE_PATH}"
  -m ""
  -i "${IMAGE_DIR}"
  --eval
  --iterations "${ITERATIONS}"
  --split_strategy file
  --split_file "${SPLIT_FILE}"
)
if [[ "${#TEST_ITER_LIST[@]}" -gt 0 ]]; then
  COMMON_ARGS+=(--test_iterations "${TEST_ITER_LIST[@]}")
fi
if [[ "${#SAVE_ITER_LIST[@]}" -gt 0 ]]; then
  COMMON_ARGS+=(--save_iterations "${SAVE_ITER_LIST[@]}")
fi

PRISM_COMMON_ARGS=(
  --enable_prism_pruning
  --prism_collect_stats
  --prism_collect_interval 20
  --prism_stats_warmup_iters 2000
  --prism_geometry_acq_until_iter 12000
  --prism_stats_collection_iters 800
  --prism_dead_rounds 1
  --prism_candidate_rounds 3
  --prism_recovery_iters 400
  --prism_candidate_prune_ratio_per_round 0.015
  --prism_dead_prune_ratio 0.005
  --prism_use_counterfactual_gate
  --prism_calib_num_buffer_views 8
  --prism_calib_num_hard_train_views 8
  --prism_round_checkpoint
  --prism_save_debug_json
  --prism_validation_interval 1000
  --prism_validation_max_views 32
)

# Conservative grounding recipe (non-PRISM), derived from previous ground optimization settings.
GROUNDING_ARGS=(
  --enable_ground_plane_estimation
  --enable_ground_regularization
  --enable_ground_plane_loss
  --enable_ground_normal_loss
  --enable_ground_smoothness_loss
  --enable_ground_mesh_assignment
  --ground_reg_start_iter 2000
  --ground_reg_warmup_iters 3000
  --lambda_ground_plane 0.015
  --lambda_ground_normal 0.008
  --lambda_ground_smoothness 0.004
  --ground_reg_target_ratio 0.08
  --ground_reg_adaptive_min_scale 1.0
  --ground_reg_adaptive_max_scale 30.0
  --ground_reg_max_total 0.08
  --ground_normal_max_abs_height 0.2
  --ground_plane_max_abs_height 0.4
  --ground_smooth_max_abs_height 0.3
)
if [[ -n "${GROUND_MASK_DIR:-}" ]]; then
  GROUNDING_ARGS+=(
    --enable_ground_masks
    --ground_mask_dir "${GROUND_MASK_DIR}"
    --ground_mask_matching auto
  )
fi

case "${CASE_NAME}" in
  no_prism)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_geom_first_no_prism"
    RUN_ARGS=()
    ;;
  grounding_only)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_geom_first_grounding"
    RUN_ARGS=(
      "${GROUNDING_ARGS[@]}"
    )
    ;;
  dead_only)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_geom_first_dead_only"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_candidate_rounds 0
      --prism_use_counterfactual_gate
    )
    ;;
  full_prism)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
    )
    ;;
  full_prism_ground_protect)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_geom_first_full_prism_ground_protect"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_use_ground_protect
      --prism_use_roi_protect
    )
    ;;
  *)
    echo "Unknown case: ${CASE_NAME}"
    exit 1
    ;;
esac

COMMON_ARGS[3]="${MODEL_PATH}"

CMD=(python train.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}")
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=(${EXTRA_ARGS})
  CMD+=("${EXTRA_SPLIT[@]}")
fi

echo "[PRISM-Run] case=${CASE_NAME}"
echo "[PRISM-Run] model=${MODEL_PATH}"
echo "[PRISM-Run] cmd: ${CMD[*]}"
"${CMD[@]}"

echo "[PRISM-Run] done: ${MODEL_PATH}"
