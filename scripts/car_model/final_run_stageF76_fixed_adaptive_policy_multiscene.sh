#!/usr/bin/env bash
set -euo pipefail

SCENE="${SCENE:-bonsai}"
GPU_ID="${GPU_ID:-6}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_MODE="${WANDB_MODE:-online}"
STAGE_ID="${STAGE_ID:-F76}"
STAGE_GROUP="${STAGE_GROUP:-final_stage${STAGE_ID}_fixed_adaptive_policy_multiscene}"
ROOT="${ROOT:-outputs/carnet/meshsplatopt/${STAGE_GROUP}}"
POLICY_TAG="${POLICY_TAG:-adaptive_f75_policy}"

case "${SCENE}" in
  bonsai)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/bonsai"
    IMAGES="images_4"
    RESOLUTION="4"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000"
    ;;
  courtyard)
    SOURCE_PATH="/data/peilincai/mesh_datasets/eth3d_colmap/courtyard"
    IMAGES="images"
    RESOLUTION="8"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000"
    ;;
  room)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/room"
    IMAGES="images_4"
    RESOLUTION="4"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000"
    ;;
  counter)
    SOURCE_PATH="/data/peilincai/mesh_datasets/mipnerf360/counter"
    IMAGES="images_4"
    RESOLUTION="4"
    CLEAN_MODEL="outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000"
    ;;
  *)
    echo "Unknown SCENE=${SCENE}; expected bonsai, courtyard, room, or counter." >&2
    exit 2
    ;;
esac

OUT_DIR="${ROOT}/${SCENE}/${POLICY_TAG}"
COMPACT_MODEL="${OUT_DIR}/compact_model"
RECOVERY_MODEL="${OUT_DIR}/recovery_model"
CONTRACT_DIR="${OUT_DIR}/recovery_contract"

mkdir -p "${OUT_DIR}/logs"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_PROJECT
export WANDB_MODE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_meshprior}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${RESET_ALL:-0}" == "1" ]]; then
  rm -rf "${COMPACT_MODEL}" "${RECOVERY_MODEL}" "${CONTRACT_DIR}"
elif [[ "${RESET_RECOVERY:-0}" == "1" ]]; then
  rm -rf "${RECOVERY_MODEL}" "${CONTRACT_DIR}"
fi

if [[ ! -f "${COMPACT_MODEL}/point_cloud/iteration_22000/point_cloud_state_dict.pt" ]]; then
  "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py \
    --source_model "${CLEAN_MODEL}" \
    --iteration 22000 \
    --output_model "${COMPACT_MODEL}" \
    --selector_mode csef_adaptive_policy \
    --selector_out_dir "${OUT_DIR}/selector" \
    2>&1 | tee "${OUT_DIR}/logs/compaction.log"
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
  --lpips_lambda 0.00025 \
  --lpips_start_iter 22000 \
  --lpips_warmup_iters 300 \
  --lpips_max_side 512 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_group "${STAGE_GROUP}" \
  --wandb_name "${STAGE_ID}_${SCENE}_fixed_adaptive_policy_sparse_lpips0p00025_22000to26000" \
  --contract_out_dir "${CONTRACT_DIR}" \
  --python "${PYTHON_BIN}" \
  --execute \
  2>&1 | tee "${OUT_DIR}/logs/recovery_eval.log"
