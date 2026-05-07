#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data/peilincai/mesh_datasets/mipnerf360}"
OUT_ROOT="${OUT_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/miniconda3/envs/Difix/bin/python}"
GPU_ID="${GPU_ID:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_GROUP="${WANDB_GROUP:-paper_m360_official_clean30k}"
SCENES="${SCENES:-bicycle flowers garden stump treehill room counter kitchen bonsai}"
METHOD_SPLIT_ITERATION="${METHOD_SPLIT_ITERATION:-26000}"
FINAL_ITERATION="${FINAL_ITERATION:-30000}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT

for scene in ${SCENES}; do
  source_path="${DATA_ROOT}/${scene}"
  if [[ ! -d "${source_path}" ]]; then
    echo "[M360 clean30k] skip missing scene: ${scene} (${source_path})"
    continue
  fi

  image_arg="images_4"
  indoor_arg=()
  case "${scene}" in
    room|counter|kitchen|bonsai)
      image_arg="images_2"
      indoor_arg=(--indoor)
      ;;
  esac

  out_path="${OUT_ROOT}/${scene}"
  split_file="${out_path}/point_cloud/iteration_${METHOD_SPLIT_ITERATION}/point_cloud_state_dict.pt"
  done_file="${out_path}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud_state_dict.pt"
  if [[ -f "${done_file}" && -f "${split_file}" ]]; then
    echo "[M360 clean30k] skip completed scene: ${scene}"
    continue
  fi
  if [[ -f "${done_file}" && ! -f "${split_file}" ]]; then
    if [[ "${RERUN_FINAL_ONLY_FOR_SPLIT:-1}" == "1" ]]; then
      echo "[M360 clean30k] rerun final-only scene to add fixed-budget split checkpoint: ${scene}"
    else
      echo "[M360 clean30k] skip final-only scene without split checkpoint: ${scene}; set RERUN_FINAL_ONLY_FOR_SPLIT=1 to rerun"
      continue
    fi
  fi

  echo "[M360 clean30k] start scene=${scene} image=${image_arg} out=${out_path}"
  "${PYTHON_BIN}" train.py \
    -s "${source_path}" \
    -i "${image_arg}" \
    -m "${out_path}" \
    --quiet \
    --eval \
    --test_iterations -1 \
    --save_iterations "${METHOD_SPLIT_ITERATION}" "${FINAL_ITERATION}" \
    --iterations "${FINAL_ITERATION}" \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_name "clean30k_${scene}_official_${image_arg}" \
    --wandb_scalar_log_interval 100 \
    --wandb_image_log_interval 3000 \
    "${indoor_arg[@]}"
done

echo "[M360 clean30k] available-scene queue finished"
