#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data/peilincai/mesh_datasets/mipnerf360}"
CLEAN_ROOT="${CLEAN_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k}"
OUT_ROOT="${OUT_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_csef_atr_26k}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/miniconda3/envs/Difix/bin/python}"
GPU_ID="${GPU_ID:-6}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_GROUP="${WANDB_GROUP:-paper_m360_compact_ela_csef_atr_26k}"
SCENES="${SCENES:-bicycle flowers garden stump treehill room counter kitchen bonsai}"
COMPACT_ITERATION="${COMPACT_ITERATION:-26000}"
POLICY_TAG="${POLICY_TAG:-csef_atr_compact_ela}"
METHOD_NAME="${METHOD_NAME:-ours_${COMPACT_ITERATION}_csef_atr_compact_ela}"
SELECTOR_SEED="${SELECTOR_SEED:-0}"

ELA_ALPHA_GRID="${ELA_ALPHA_GRID:-0,0.125,0.25,0.5,0.75,1.0}"
ELA_K_VALUES="${ELA_K_VALUES:-4,8}"
ELA_DEPTH_REL_VALUES="${ELA_DEPTH_REL_VALUES:-0.06,0.12}"
ELA_RESIDUAL_CLIP_VALUES="${ELA_RESIDUAL_CLIP_VALUES:-0.20,0.25}"
ELA_DIRECTION_WEIGHT_VALUES="${ELA_DIRECTION_WEIGHT_VALUES:-0.20,0.35}"
ELA_CALIB_MAX_VIEWS="${ELA_CALIB_MAX_VIEWS:-16}"
ELA_CALIB_STRIDE="${ELA_CALIB_STRIDE:-12}"
ELA_EDGE_QUANTILE="${ELA_EDGE_QUANTILE:-0.70}"
ELA_EDGE_DILATE="${ELA_EDGE_DILATE:-1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_meshprior}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for scene in ${SCENES}; do
  source_path="${DATA_ROOT}/${scene}"
  clean_model="${CLEAN_ROOT}/${scene}"
  clean_split_ckpt="${clean_model}/point_cloud/iteration_${COMPACT_ITERATION}/point_cloud_state_dict.pt"
  if [[ ! -d "${source_path}" ]]; then
    echo "[M360 compact-ELA] skip missing source scene: ${scene} (${source_path})"
    continue
  fi
  if [[ ! -f "${clean_split_ckpt}" ]]; then
    echo "[M360 compact-ELA] skip incomplete clean checkpoint for ${scene}: need ${clean_split_ckpt}"
    continue
  fi

  image_arg="images_4"
  case "${scene}" in
    room|counter|kitchen|bonsai)
      image_arg="images_2"
      ;;
  esac

  out_dir="${OUT_ROOT}/${scene}/${POLICY_TAG}"
  compact_model="${out_dir}/compact_model"
  mkdir -p "${out_dir}/logs"

  if [[ ! -f "${compact_model}/point_cloud/iteration_${COMPACT_ITERATION}/point_cloud_state_dict.pt" ]]; then
    echo "[M360 compact-ELA] compact scene=${scene} iter=${COMPACT_ITERATION}"
    compaction_args=(
      --source_model "${clean_model}"
      --iteration "${COMPACT_ITERATION}"
      --output_model "${compact_model}"
      --seed "${SELECTOR_SEED}"
    )
    selector_json="${out_dir}/selector/compaction_candidates.json"
    if [[ -f "${selector_json}" ]]; then
      compaction_args+=(--candidates_json "${selector_json}")
    else
      compaction_args+=(
        --selector_mode csef_adaptive_policy
        --selector_out_dir "${out_dir}/selector"
      )
    fi
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py \
      "${compaction_args[@]}" \
      2>&1 | tee "${out_dir}/logs/compaction.log"
  fi

  if [[ ! -f "${compact_model}/train/ours_${COMPACT_ITERATION}/camera_index.json" || ! -d "${compact_model}/train/ours_${COMPACT_ITERATION}/depths" || ! -f "${compact_model}/test/ours_${COMPACT_ITERATION}/camera_index.json" || ! -d "${compact_model}/test/ours_${COMPACT_ITERATION}/depths" ]]; then
    echo "[M360 compact-ELA] render RGB/depth evidence scene=${scene}"
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_render_evidence_maps.py \
      -s "${source_path}" \
      -i "${image_arg}" \
      -m "${compact_model}" \
      --resolution -1 \
      --eval \
      --iteration "${COMPACT_ITERATION}" \
      --quiet \
      2>&1 | tee "${out_dir}/logs/evidence_render.log"
  fi

  if [[ ! -f "${compact_model}/test/${METHOD_NAME}/ela_report.json" ]]; then
    echo "[M360 compact-ELA] apply train-calibrated ELA scene=${scene}"
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
      --base_model_path "${compact_model}" \
      --iteration "${COMPACT_ITERATION}" \
      --target_split test \
      --method_name "${METHOD_NAME}" \
      --auto_policy \
      --policy_modes residual,color \
      --policy_k_values "${ELA_K_VALUES}" \
      --policy_depth_rel_values "${ELA_DEPTH_REL_VALUES}" \
      --policy_residual_clip_values "${ELA_RESIDUAL_CLIP_VALUES}" \
      --policy_direction_weight_values "${ELA_DIRECTION_WEIGHT_VALUES}" \
      --policy_objective balanced \
      --policy_ssim_weight 20.0 \
      --policy_lpips_weight 20.0 \
      --calib_lpips \
      --benefit_policy \
      --edge_gate \
      --edge_gate_quantile "${ELA_EDGE_QUANTILE}" \
      --edge_gate_dilate "${ELA_EDGE_DILATE}" \
      --alpha_grid "${ELA_ALPHA_GRID}" \
      --calib_stride "${ELA_CALIB_STRIDE}" \
      --calib_max_views "${ELA_CALIB_MAX_VIEWS}" \
      --calib_sampler uniform \
      --wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_group "${WANDB_GROUP}" \
      --wandb_name "compact_ela_${scene}_${POLICY_TAG}_${COMPACT_ITERATION}" \
      2>&1 | tee "${out_dir}/logs/ela.log"
  fi

  echo "[M360 compact-ELA] metrics scene=${scene}"
  "${PYTHON_BIN}" metrics.py -m "${compact_model}" \
    2>&1 | tee "${out_dir}/logs/metrics.log"

  if [[ ! -f "${compact_model}/geometry_eval_colmap/iter_${COMPACT_ITERATION}_max500.json" ]]; then
    echo "[M360 compact-ELA] geometry scene=${scene}"
    "${PYTHON_BIN}" evaluate_geometry_colmap.py \
      -s "${source_path}" \
      -m "${compact_model}" \
      --images "${image_arg}" \
      --resolution -1 \
      --eval \
      --iteration "${COMPACT_ITERATION}" \
      --max_points_per_view 500 \
      --output "${compact_model}/geometry_eval_colmap/iter_${COMPACT_ITERATION}_max500.json" \
      2>&1 | tee "${out_dir}/logs/geometry.log"
  fi
done

echo "[M360 compact-ELA] available-scene queue finished"
