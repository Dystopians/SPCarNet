#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data/peilincai/mesh_datasets/mipnerf360}"
CLEAN_ROOT="${CLEAN_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k}"
OUT_ROOT="${OUT_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/fixedbudget_csef_atr_26kto30k}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/miniconda3/envs/Difix/bin/python}"
GPU_ID="${GPU_ID:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-spcarnet_meshprior}"
WANDB_GROUP="${WANDB_GROUP:-paper_m360_fixedbudget_csef_atr_26kto30k}"
SCENES="${SCENES:-bicycle flowers garden stump treehill room counter kitchen bonsai}"
COMPACT_ITERATION="${COMPACT_ITERATION:-26000}"
FINAL_ITERATION="${FINAL_ITERATION:-30000}"
POLICY_TAG="${POLICY_TAG:-csef_atr_fixedbudget}"
SELECTOR_SEED="${SELECTOR_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
RECOVERY_PRESET="${RECOVERY_PRESET:-compact_sparse_low_lambda}"
SPARSE_LAMBDA="${SPARSE_LAMBDA:-0.001}"
SPARSE_FRACTION="${SPARSE_FRACTION:-0.5}"
LPIPS_LAMBDA="${LPIPS_LAMBDA:-0.00025}"
ENABLE_PARENT_ROLLBACK="${ENABLE_PARENT_ROLLBACK:-1}"
PARENT_ROLLBACK_LAMBDA="${PARENT_ROLLBACK_LAMBDA:-0.5}"
PARENT_ROLLBACK_DSSIM_WEIGHT="${PARENT_ROLLBACK_DSSIM_WEIGHT:-0.15}"
PARENT_ROLLBACK_EDGE_WEIGHT="${PARENT_ROLLBACK_EDGE_WEIGHT:-0.05}"
PARENT_ROLLBACK_EDGE_GUIDANCE_WEIGHT="${PARENT_ROLLBACK_EDGE_GUIDANCE_WEIGHT:-0.10}"

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
  clean_final_ckpt="${clean_model}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud_state_dict.pt"
  if [[ ! -d "${source_path}" ]]; then
    echo "[M360 fixedbudget method] skip missing source scene: ${scene} (${source_path})"
    continue
  fi
  if [[ ! -f "${clean_split_ckpt}" || ! -f "${clean_final_ckpt}" ]]; then
    echo "[M360 fixedbudget method] skip incomplete clean checkpoints for ${scene}: need ${clean_split_ckpt} and ${clean_final_ckpt}"
    continue
  fi

  image_arg="images_4"
  resolution_arg="-1"
  indoor_arg=()
  case "${scene}" in
    room|counter|kitchen|bonsai)
      image_arg="images_2"
      indoor_arg=(--indoor)
      ;;
  esac

  out_dir="${OUT_ROOT}/${scene}/${POLICY_TAG}"
  compact_model="${out_dir}/compact_model"
  recovery_model="${out_dir}/recovery_model"
  contract_dir="${out_dir}/recovery_contract"
  mkdir -p "${out_dir}/logs"

  method_final_ckpt="${recovery_model}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud_state_dict.pt"
  method_results="${recovery_model}/results.json"
  method_geometry="${recovery_model}/geometry_eval_colmap/iter_${FINAL_ITERATION}_max500.json"
  if [[ -f "${method_final_ckpt}" && -f "${method_results}" && -f "${method_geometry}" ]]; then
    echo "[M360 fixedbudget method] skip completed recovery/eval for ${scene}: ${method_final_ckpt}"
    continue
  fi

  if [[ ! -f "${compact_model}/point_cloud/iteration_${COMPACT_ITERATION}/point_cloud_state_dict.pt" ]]; then
    echo "[M360 fixedbudget method] compact scene=${scene} iter=${COMPACT_ITERATION}"
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

  if [[ ! -f "${recovery_model}/point_cloud/iteration_${COMPACT_ITERATION}/point_cloud_state_dict.pt" ]]; then
    rm -rf "${recovery_model}"
    rsync -a "${compact_model}/" "${recovery_model}/"
  fi

  parent_rollback_args=()
  if [[ "${ENABLE_PARENT_ROLLBACK}" == "1" && "${PARENT_ROLLBACK_LAMBDA}" != "0" && "${PARENT_ROLLBACK_LAMBDA}" != "0.0" ]]; then
    parent_render_dir="${clean_model}/train/ours_${COMPACT_ITERATION}/renders"
    if [[ ! -d "${parent_render_dir}" ]]; then
      echo "[M360 fixedbudget method] render parent train cache scene=${scene} iter=${COMPACT_ITERATION}"
      "${PYTHON_BIN}" render.py \
        -s "${source_path}" \
        -i "${image_arg}" \
        -m "${clean_model}" \
        --resolution "${resolution_arg}" \
        --eval \
        --iteration "${COMPACT_ITERATION}" \
        --skip_test \
        --quiet
    fi
    parent_rollback_args=(
      --parent_render_rollback_dir "${parent_render_dir}"
      --parent_render_rollback_lambda "${PARENT_ROLLBACK_LAMBDA}"
      --parent_render_rollback_start_iter "${COMPACT_ITERATION}"
      --parent_render_rollback_warmup_iters 300
      --parent_render_rollback_aggregation cvar
      --parent_render_rollback_cvar_fraction 0.10
      --parent_render_rollback_cvar_min_pixels 1024
      --parent_render_rollback_patch_radius 1
      --parent_render_rollback_patch_reduce max_violation
      --parent_render_rollback_error_space l1_dssim_edge
      --parent_render_rollback_dssim_weight "${PARENT_ROLLBACK_DSSIM_WEIGHT}"
      --parent_render_rollback_edge_weight "${PARENT_ROLLBACK_EDGE_WEIGHT}"
      --parent_render_rollback_edge_guidance_weight "${PARENT_ROLLBACK_EDGE_GUIDANCE_WEIGHT}"
    )
  fi

  echo "[M360 fixedbudget method] recover scene=${scene} ${COMPACT_ITERATION}->${FINAL_ITERATION}"
  "${PYTHON_BIN}" scripts/car_model/meshsplatopt_run_strict_compact_recovery.py \
    --source_path "${source_path}" \
    --output_path "${recovery_model}" \
    --load_iteration "${COMPACT_ITERATION}" \
    --final_iteration "${FINAL_ITERATION}" \
    --images "${image_arg}" \
    --resolution "${resolution_arg}" \
    --preset "${RECOVERY_PRESET}" \
    --sparse_lambda "${SPARSE_LAMBDA}" \
    --sparse_start_iter "${COMPACT_ITERATION}" \
    --sparse_warmup_iters 300 \
    --sparse_min_matches 16 \
    --sparse_sample_mode mixed_low_error \
    --sparse_fraction "${SPARSE_FRACTION}" \
    --lpips_lambda "${LPIPS_LAMBDA}" \
    --lpips_start_iter "${COMPACT_ITERATION}" \
    --lpips_warmup_iters 300 \
    --lpips_max_side 512 \
    "${parent_rollback_args[@]}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_name "fixedbudget_${scene}_${POLICY_TAG}_seed${TRAIN_SEED}_${COMPACT_ITERATION}to${FINAL_ITERATION}" \
    --train_seed "${TRAIN_SEED}" \
    "${indoor_arg[@]}" \
    --contract_out_dir "${contract_dir}" \
    --python "${PYTHON_BIN}" \
    --execute \
    2>&1 | tee "${out_dir}/logs/recovery_eval.log"
done

echo "[M360 fixedbudget method] available-scene queue finished"
