#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data/peilincai/mesh_datasets/mipnerf360}"
OUT_ROOT="${OUT_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/miniconda3/envs/Difix/bin/python}"
GPU_ID="${GPU_ID:-4}"
SCENES="${SCENES:-bicycle flowers garden stump treehill room counter kitchen bonsai}"
FINAL_ITERATION="${FINAL_ITERATION:-30000}"
RUN_GEOMETRY="${RUN_GEOMETRY:-1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

model_paths=()
for scene in ${SCENES}; do
  source_path="${DATA_ROOT}/${scene}"
  model_path="${OUT_ROOT}/${scene}"
  ckpt="${model_path}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud_state_dict.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[M360 clean30k eval] skip incomplete scene: ${scene}"
    continue
  fi

  image_arg="images_4"
  case "${scene}" in
    room|counter|kitchen|bonsai)
      image_arg="images_2"
      ;;
  esac

  echo "[M360 clean30k eval] render scene=${scene}"
  "${PYTHON_BIN}" render.py \
    -s "${source_path}" \
    -i "${image_arg}" \
    -m "${model_path}" \
    --eval \
    --iteration "${FINAL_ITERATION}" \
    --skip_train \
    --quiet
  if [[ "${RUN_GEOMETRY}" == "1" ]]; then
    echo "[M360 clean30k eval] geometry scene=${scene}"
    "${PYTHON_BIN}" evaluate_geometry_colmap.py \
      -s "${source_path}" \
      -m "${model_path}" \
      --images "${image_arg}" \
      --resolution -1 \
      --eval \
      --iteration "${FINAL_ITERATION}" \
      --max_points_per_view 500 \
      --output "${model_path}/geometry_eval_colmap/iter_${FINAL_ITERATION}_max500.json"
  fi
  model_paths+=("${model_path}")
done

if [[ "${#model_paths[@]}" -eq 0 ]]; then
  echo "[M360 clean30k eval] no completed scenes to evaluate"
  exit 0
fi

echo "[M360 clean30k eval] metrics for ${#model_paths[@]} completed scenes"
"${PYTHON_BIN}" metrics.py -m "${model_paths[@]}"

echo "[M360 clean30k eval] collect and log metrics"
"${PYTHON_BIN}" scripts/car_model/collect_paper_m360_repro_metrics.py \
  --root "${OUT_ROOT}" \
  --scenes "$(echo "${SCENES}" | tr ' ' ',')" \
  --iteration "${FINAL_ITERATION}" \
  --out-csv "${OUT_ROOT}/repro_metrics_vs_paper_iter${FINAL_ITERATION}.csv" \
  --out-json "${OUT_ROOT}/repro_metrics_vs_paper_iter${FINAL_ITERATION}.json" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-spcarnet_meshprior}" \
  --wandb_group "${WANDB_GROUP:-paper_m360_official_clean30k}" \
  --wandb_name "paper_m360_clean30k_available_metrics"

echo "[M360 clean30k eval] finished"
