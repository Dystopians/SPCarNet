#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-5}"
ITERATIONS="${ITERATIONS:-7000}"
MODEL_ROOT="${MODEL_ROOT:-outputs/carnet/meshprior/stageF44_bonsai_calibrated_gate_7000/bonsai_${ITERATIONS}iter_calibrated_gate}"
MODEL_PATH="${MODEL_ROOT}/model"
SOURCE_PATH="${SOURCE_PATH:-/data/peilincai/mesh_datasets/mipnerf360/bonsai}"
IMAGES="${IMAGES:-images_4}"
RESOLUTION="${RESOLUTION:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-final_stageF44_bonsai_calibrated_gate_7000}"
WANDB_NAME="${WANDB_NAME:-F44_bonsai_${ITERATIONS}_calibrated_gate}"

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
  --test_iterations 1000 "${ITERATIONS}"
  --save_iterations "${ITERATIONS}"
  --checkpoint_iterations "${ITERATIONS}"
  --resolution "${RESOLUTION}"
  --scene_name "F44_bonsai_ratio002_${ITERATIONS}_calibrated_gate"
  --enable_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_group "${WANDB_GROUP}"
  --wandb_name "${WANDB_NAME}"
  --wandb_scalar_log_interval 50
  --wandb_disable_fixed_views
  --enable_sparse_colmap_depth_loss
  --lambda_sparse_colmap_depth 0.005
  --sparse_colmap_depth_start_iter 100
  --sparse_colmap_depth_warmup_iters 300
  --sparse_colmap_depth_min_matches 16
  --enable_prism_pruning
  --prism_collect_stats
  --prism_collect_interval 10
  --prism_stats_warmup_iters 25
  --prism_score_recompute_interval 100
  --prism_max_triangles_for_heavy_metrics 0
  --prism_heavy_eval_budget 20000
  --prism_force_full_heavy_eval_below -1
  --prism_geometry_acq_until_iter 1400
  --prism_stats_collection_iters 100
  --prism_dead_rounds 0
  --prism_candidate_rounds 6
  --prism_candidate_prune_ratio_per_round 0.02
  --prism_no_candidate_retry_iters 10
  --prism_freeze_densification_after_first_commit
  --prism_keep_geometry_threshold 0.85
  --prism_keep_orientation_threshold 1.1
  --prism_keep_render_threshold 0.85
  --prism_boundary_risk_value 0
  --prism_nonmanifold_risk_value 0
  --prism_protected_dilation_rings 0
  --prism_validation_interval 250
  --prism_validation_max_views 8
  --prism_validation_num_buffer_views 4
  --prism_validation_num_train_views 4
  --prism_validation_train_pool_size 32
  --prism_validation_min_valid_depth_matches 32
  --prism_validation_min_valid_normal_matches 0
  --prism_rollback_absrel_rel_thresh 0.30
  --prism_rollback_mean_angle_thresh 2.0
  --prism_rollback_psnr_drop_thresh 1.0
  --prism_rollback_mae_increase_thresh 0.030
  --prism_use_counterfactual_gate
  --prism_gate_min_delta_psnr_db -2.0
  --prism_gate_max_delta_mae 0.010
  --prism_gate_max_delta_absrel 0.020
  --prism_gate_max_delta_mean_angle_deg 0.20
  --prism_gate_max_changed_pixel_ratio 0.080
  --prism_recovery_iters 80
  --prism_post_commit_recollect_iters 10
  --prism_final_finetune_iters 200
  --prism_disable_final_cleanup_prune
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
