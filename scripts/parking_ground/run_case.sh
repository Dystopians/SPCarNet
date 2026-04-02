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
#   SAVE_ITERATIONS="15000 16000 18000 20000 21000 24000 30000"

CASE_NAME="${1:-}"
if [[ -z "${CASE_NAME}" ]]; then
  echo "Missing case name. Expected one of:"
  echo "  baseline | prism_geogate_fix | prism_late_prune | prism_geogate_fix_keep | prism_geogate_compact"
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
SAVE_ITERATIONS="${SAVE_ITERATIONS:-15000 16000 18000 20000 21000 24000 30000}"

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
  --prism_force_full_heavy_eval_below 400000
  --prism_heavy_eval_budget 120000
  --prism_heavy_eval_neighbor_rings 2
  --prism_geometry_acq_until_iter 16000
  --prism_stats_collection_iters 1500
  --prism_dead_rounds 1
  --prism_candidate_rounds 2
  --prism_recovery_iters 1000
  --prism_post_commit_recollect_iters 300
  --prism_candidate_prune_ratio_per_round 0.0075
  --prism_dead_prune_ratio 0.005
  --prism_use_counterfactual_gate
  --prism_calib_num_buffer_views 8
  --prism_calib_num_hard_train_views 8
  --prism_round_checkpoint
  --prism_save_debug_json
  --prism_validation_interval 1000
  --prism_validation_max_views 32
)

case "${CASE_NAME}" in
  baseline)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_baseline"
    RUN_ARGS=()
    ;;
  prism_geogate_fix)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_geometry_acq_until_iter 16000
      --prism_stats_collection_iters 1500
      --prism_candidate_rounds 2
      --prism_candidate_prune_ratio_per_round 0.0075
      --prism_recovery_iters 1000
      --prism_post_commit_recollect_iters 300
      --enable_sparse_colmap_depth_loss
      --lambda_sparse_colmap_depth 0.01
      --sparse_colmap_depth_start_iter 1000
      --sparse_colmap_depth_warmup_iters 3000
      --sparse_colmap_depth_min_matches 32
      --prism_protected_dilation_rings 0
      --prism_keep_geometry_threshold 1.1
      --prism_keep_orientation_threshold 1.1
      --prism_keep_render_threshold 1.1
      --prism_candidate_block_geometry_keep_threshold 1.1
    )
    ;;
  prism_late_prune)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_prism_late_prune"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_geometry_acq_until_iter 18000
      --prism_stats_collection_iters 2000
      --prism_candidate_rounds 2
      --prism_candidate_prune_ratio_per_round 0.0075
      --prism_recovery_iters 1000
      --prism_post_commit_recollect_iters 300
      --prism_protected_dilation_rings 0
      --prism_keep_geometry_threshold 1.1
      --prism_keep_orientation_threshold 1.1
      --prism_keep_render_threshold 1.1
      --prism_candidate_block_geometry_keep_threshold 1.1
    )
    ;;
  prism_geogate_fix_keep)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_fix_keep"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_geometry_acq_until_iter 16000
      --prism_stats_collection_iters 1500
      --prism_candidate_rounds 2
      --prism_candidate_prune_ratio_per_round 0.0075
      --prism_recovery_iters 1000
      --prism_post_commit_recollect_iters 300
      --enable_sparse_colmap_depth_loss
      --lambda_sparse_colmap_depth 0.01
      --sparse_colmap_depth_start_iter 1000
      --sparse_colmap_depth_warmup_iters 3000
      --sparse_colmap_depth_min_matches 32
      --prism_heavy_eval_budget 180000
      --prism_protected_dilation_rings 1
      --prism_keep_geometry_threshold 0.6
      --prism_keep_orientation_threshold 0.6
      --prism_keep_render_threshold 0.6
      --prism_candidate_block_geometry_keep_threshold 0.6
    )
    ;;
  prism_geogate_compact)
    MODEL_PATH="${MODEL_ROOT}/${RUN_TAG}_prism_geogate_compact"
    RUN_ARGS=(
      "${PRISM_COMMON_ARGS[@]}"
      --prism_geometry_acq_until_iter 16000
      --prism_stats_collection_iters 1500
      --prism_candidate_rounds 2
      --prism_candidate_prune_ratio_per_round 0.0075
      --prism_recovery_iters 1000
      --prism_post_commit_recollect_iters 300
      --enable_sparse_colmap_depth_loss
      --lambda_sparse_colmap_depth 0.01
      --sparse_colmap_depth_start_iter 1000
      --sparse_colmap_depth_warmup_iters 3000
      --sparse_colmap_depth_min_matches 32
      --prism_enable_compaction_stage
      --prism_compaction_rounds 2
      --prism_compaction_microbatch_active_ratio 0.0035
      --prism_compaction_max_microbatches_per_round 6
      --prism_compaction_candidate_pool_multiplier 6.0
      --prism_compaction_min_prune_count 256
      --prism_compaction_roi_budget_fraction 0.10
      --prism_compaction_near_field_budget_fraction 0.25
      --prism_compaction_roi_signal_threshold 0.05
      --prism_compaction_near_field_area_percentile 80
      --prism_protected_dilation_rings 0
      --prism_keep_geometry_threshold 1.1
      --prism_keep_orientation_threshold 1.1
      --prism_keep_render_threshold 1.1
      --prism_candidate_block_geometry_keep_threshold 1.1
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
