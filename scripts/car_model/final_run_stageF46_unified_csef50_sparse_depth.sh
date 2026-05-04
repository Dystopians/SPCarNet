#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-5}"
SCENE="${SCENE:-counter}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_MODE="${WANDB_MODE:-online}"
ROOT="outputs/carnet/meshsplatopt/final_stageF46_unified_csef50_sparse_depth"

case "${SCENE}" in
  bonsai)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/bonsai"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/bonsai/csef_low_evidence_boundary_protected/prune50/compact_model"
    ;;
  room)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/room"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="outputs/carnet/meshsplatopt/final_stageF9_third_scene_room/csef_low_evidence_boundary_protected/prune50/compact_model"
    ;;
  room20)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/room"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="${ROOT}/room/prune20/compact_model"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000"
    TARGET_PRUNE_FRACTION="0.20"
    ;;
  counter)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/counter"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune50/compact_model"
    ;;
  counter40)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/counter"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune40/compact_model"
    ;;
  counter30)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/counter"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="${ROOT}/counter/prune30/compact_model"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000"
    TARGET_PRUNE_FRACTION="0.30"
    ;;
  counter20)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/counter"
    IMAGES="images_4"
    RESOLUTION="4"
    COMPACT_MODEL="${ROOT}/counter/prune20/compact_model"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000"
    TARGET_PRUNE_FRACTION="0.20"
    ;;
  *)
    echo "Unknown SCENE=${SCENE}. Expected bonsai, room, room20, counter, counter40, counter30, or counter20." >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_PROJECT
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_meshprior}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${SCENE}" == "counter40" ]]; then
  OUT_DIR="${ROOT}/counter/prune40"
  WANDB_SCENE="counter_csef40"
elif [[ "${SCENE}" == "room20" ]]; then
  OUT_DIR="${ROOT}/room/prune20"
  WANDB_SCENE="room_csef20"
elif [[ "${SCENE}" == "counter30" ]]; then
  OUT_DIR="${ROOT}/counter/prune30"
  WANDB_SCENE="counter_csef30"
elif [[ "${SCENE}" == "counter20" ]]; then
  OUT_DIR="${ROOT}/counter/prune20"
  WANDB_SCENE="counter_csef20"
else
  OUT_DIR="${ROOT}/${SCENE}/prune50"
  WANDB_SCENE="${SCENE}_csef50"
fi
RECOVERY_MODEL="${OUT_DIR}/recovery_model"
CONTRACT_DIR="${OUT_DIR}/recovery_contract"
mkdir -p "${OUT_DIR}/logs"

if [[ "${RESET_RECOVERY:-0}" == "1" ]]; then
  rm -rf "${RECOVERY_MODEL}"
fi

if [[ -n "${TARGET_PRUNE_FRACTION:-}" && ! -f "${COMPACT_MODEL}/point_cloud/iteration_22000/point_cloud_state_dict.pt" ]]; then
  "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py \
    --source_model "${CLEAN_MODEL}" \
    --iteration 22000 \
    --output_model "${COMPACT_MODEL}" \
    --selector_mode csef_low_evidence_boundary_protected \
    --target_prune_fraction "${TARGET_PRUNE_FRACTION}"
fi

if [[ ! -f "${RECOVERY_MODEL}/point_cloud/iteration_22000/point_cloud_state_dict.pt" ]]; then
  rm -rf "${RECOVERY_MODEL}"
  rsync -a "${COMPACT_MODEL}/" "${RECOVERY_MODEL}/"
fi

"${PYTHON_BIN}" scripts/car_model/meshsplatopt_run_strict_compact_recovery.py \
  --source_path "${SOURCE_PATH}" \
  --output_path "${RECOVERY_MODEL}" \
  --load_iteration 22000 \
  --final_iteration 26000 \
  --images "${IMAGES}" \
  --resolution "${RESOLUTION}" \
  --preset compact_sparse_low_lambda \
  --sparse_lambda 0.001 \
  --sparse_start_iter 22000 \
  --sparse_warmup_iters 300 \
  --sparse_min_matches 16 \
  --sparse_sample_mode mixed_low_error \
  --sparse_fraction 0.5 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "final_stageF46_unified_csef50_sparse_depth" \
  --wandb_name "F46_${WANDB_SCENE}_sparse_depth_22000to26000" \
  --contract_out_dir "${CONTRACT_DIR}" \
  --python "${PYTHON_BIN}" \
  --execute \
  2>&1 | tee "${OUT_DIR}/logs/run.log"
