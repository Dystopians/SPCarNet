#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
ITERATIONS="${ITERATIONS:-800}"
MODEL_ROOT="${MODEL_ROOT:-outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_smoke_${ITERATIONS}iter}"
MODEL_PATH="${MODEL_ROOT}/model"
DATASET_VIEW="${DATASET_VIEW:-outputs/carnet/meshprior/parking_phone_tiny/dataset_view}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-parking_stage23_5_integrated_topology}"
WANDB_NAME="${WANDB_NAME:-parking_stage23_5_prism_smoke_${ITERATIONS}iter}"

mkdir -p "${MODEL_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_PROJECT
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_meshprior}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CMD=(
  "${PYTHON_BIN}" train.py
  -s "${DATASET_VIEW}"
  -m "${MODEL_PATH}"
  --images images
  --eval
  --iterations "${ITERATIONS}"
  --test_iterations 400 "${ITERATIONS}"
  --save_iterations "${ITERATIONS}"
  --checkpoint_iterations "${ITERATIONS}"
  --resolution 4
  --scene_name "parking_phone_tiny_stage23_5_prism_smoke_${ITERATIONS}iter"
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
  --prism_geometry_acq_until_iter 80
  --prism_stats_collection_iters 60
  --prism_dead_rounds 0
  --prism_candidate_rounds 1
  --prism_candidate_prune_ratio_per_round 0.02
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
  -s "${DATASET_VIEW}"
  -m "${MODEL_PATH}"
  --images images
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
  -s "${DATASET_VIEW}"
  -m "${MODEL_PATH}"
  --images images
  --eval
  --iteration "${ITERATIONS}"
  --max_points_per_view 500
  --output "${MODEL_PATH}/geometry_eval_colmap/iter_${ITERATIONS}.json"
)
printf '%q ' "${GEOM_CMD[@]}" > "${MODEL_ROOT}/logs/geometry_command.txt"
printf '\n' >> "${MODEL_ROOT}/logs/geometry_command.txt"
"${GEOM_CMD[@]}" 2>&1 | tee "${MODEL_ROOT}/logs/geometry.log"

"${PYTHON_BIN}" scripts/car_model/meshprior_collect_stage23_5_integrated_topology.py \
  --model "${MODEL_PATH}" \
  --iteration "${ITERATIONS}" \
  --output_dir "${MODEL_ROOT}/summary" \
  2>&1 | tee "${MODEL_ROOT}/logs/collect.log"
