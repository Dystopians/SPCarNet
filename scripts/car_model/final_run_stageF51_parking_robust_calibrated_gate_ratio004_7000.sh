#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
ITERATIONS="${ITERATIONS:-7000}"
MODEL_ROOT="${MODEL_ROOT:-outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/parking_${ITERATIONS}iter_ratio004_robust_calibrated_gate}"
MODEL_PATH="${MODEL_ROOT}/model"
SOURCE_PATH="${SOURCE_PATH:-outputs/carnet/meshprior/parking_phone_tiny/dataset_view}"
IMAGES="${IMAGES:-images}"
RESOLUTION="${RESOLUTION:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-final_stageF51_parking_robust_calibrated_gate_ratio004_7000}"
WANDB_NAME="${WANDB_NAME:-F51_parking_ratio004_${ITERATIONS}_robust_calibrated_gate}"
SCENE_NAME="${SCENE_NAME:-F51_parking_ratio004_${ITERATIONS}_robust_calibrated_gate}"
PRISM_GEOMETRY_ACQ_UNTIL_ITER="${PRISM_GEOMETRY_ACQ_UNTIL_ITER:-80}"
PRISM_STATS_COLLECTION_ITERS="${PRISM_STATS_COLLECTION_ITERS:-60}"
PRISM_CANDIDATE_PRUNE_RATIO="${PRISM_CANDIDATE_PRUNE_RATIO:-0.04}"

mkdir -p "${MODEL_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_PROJECT
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_meshprior}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train.py
  -s "${SOURCE_PATH}"
  -m "${MODEL_PATH}"
  --images "${IMAGES}"
  --eval
  --iterations "${ITERATIONS}"
  --test_iterations 400 "${ITERATIONS}"
  --save_iterations 1000 2000 3000 4000 5000 6000 "${ITERATIONS}"
  --checkpoint_iterations 1000 2000 3000 4000 5000 6000 "${ITERATIONS}"
  --resolution "${RESOLUTION}"
  --scene_name "${SCENE_NAME}"
  --enable_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group "${WANDB_GROUP}"
  --wandb_name "${WANDB_NAME}"
  --wandb_scalar_log_interval 25
  --wandb_disable_fixed_views
  --enable_prism_pruning
  --prism_collect_stats
  --prism_collect_interval 10
  --prism_stats_warmup_iters 10
  --prism_score_recompute_interval 25
  --prism_max_triangles_for_heavy_metrics 100000
  --prism_heavy_eval_budget 20000
  --prism_force_full_heavy_eval_below 100000
  --prism_geometry_acq_until_iter "${PRISM_GEOMETRY_ACQ_UNTIL_ITER}"
  --prism_stats_collection_iters "${PRISM_STATS_COLLECTION_ITERS}"
  --prism_dead_rounds 0
  --prism_candidate_rounds 1
  --prism_candidate_prune_ratio_per_round "${PRISM_CANDIDATE_PRUNE_RATIO}"
  --prism_recent_age_iters 0
  --prism_thresh_protected_edge 1.1
  --prism_thresh_protected_geo 1.1
  --prism_thresh_protected_sens 1.1
  --prism_thresh_protected_unc 1.1
  --prism_boundary_risk_value 0.0
  --prism_nonmanifold_risk_value 0.0
  --prism_protected_dilation_rings 0
  --prism_keep_geometry_threshold 1.1
  --prism_keep_orientation_threshold 1.1
  --prism_keep_render_threshold 1.1
  --prism_candidate_block_geometry_keep_threshold 1.1
  --prism_use_counterfactual_gate
  --prism_gate_min_delta_psnr_db -2.0
  --prism_gate_max_delta_mae 0.010
  --prism_gate_max_delta_absrel 0.020
  --prism_gate_max_baseline_absrel_for_absrel_check 2.0
  --prism_gate_max_delta_mean_angle_deg 0.20
  --prism_gate_max_changed_pixel_ratio 0.080
  --prism_calib_num_buffer_views 2
  --prism_calib_num_hard_train_views 2
  --prism_gate_min_valid_depth_matches 16
  --prism_gate_min_valid_normal_matches 0
  --prism_validation_interval 200
  --prism_validation_max_views 4
  --prism_validation_num_buffer_views 2
  --prism_validation_num_train_views 2
  --prism_validation_train_pool_size 16
  --prism_validation_min_valid_depth_matches 16
  --prism_validation_min_valid_normal_matches 0
  --prism_recovery_iters 60
  --prism_post_commit_recollect_iters 30
  --prism_final_finetune_iters 100
  --prism_disable_final_cleanup_prune
  --prism_save_debug_json
  --enable_sparse_colmap_depth_loss
  --lambda_sparse_colmap_depth 0.005
  --sparse_colmap_depth_start_iter 50
  --sparse_colmap_depth_warmup_iters 100
  --sparse_colmap_depth_min_matches 16
)

printf '%q ' "${TRAIN_CMD[@]}" > "${MODEL_ROOT}/logs/train_command.txt"
printf '\n' >> "${MODEL_ROOT}/logs/train_command.txt"
"${TRAIN_CMD[@]}" 2>&1 | tee "${MODEL_ROOT}/logs/train.log"

RENDER_CMD=(
  "${PYTHON_BIN}" render.py
  -s "${SOURCE_PATH}"
  -m "${MODEL_PATH}"
  --images "${IMAGES}"
  --eval
  --iteration "${ITERATIONS}"
  --skip_train
  --quiet
)
printf '%q ' "${RENDER_CMD[@]}" > "${MODEL_ROOT}/logs/render_command.txt"
printf '\n' >> "${MODEL_ROOT}/logs/render_command.txt"
"${RENDER_CMD[@]}" 2>&1 | tee "${MODEL_ROOT}/logs/render.log"

METRICS_CMD=("${PYTHON_BIN}" metrics.py -m "${MODEL_PATH}")
printf '%q ' "${METRICS_CMD[@]}" > "${MODEL_ROOT}/logs/metrics_command.txt"
printf '\n' >> "${MODEL_ROOT}/logs/metrics_command.txt"
"${METRICS_CMD[@]}" 2>&1 | tee "${MODEL_ROOT}/logs/metrics.log"

GEOM_CMD=(
  "${PYTHON_BIN}" evaluate_geometry_colmap.py
  -s "${SOURCE_PATH}"
  -m "${MODEL_PATH}"
  --images "${IMAGES}"
  --eval
  --iteration "${ITERATIONS}"
  --max_points_per_view 500
  --output "${MODEL_PATH}/geometry_eval_colmap/iter_${ITERATIONS}.json"
)
printf '%q ' "${GEOM_CMD[@]}" > "${MODEL_ROOT}/logs/geometry_command.txt"
printf '\n' >> "${MODEL_ROOT}/logs/geometry_command.txt"
"${GEOM_CMD[@]}" 2>&1 | tee "${MODEL_ROOT}/logs/geometry.log"
