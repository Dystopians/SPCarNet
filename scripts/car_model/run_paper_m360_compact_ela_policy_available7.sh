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
SPARSE_OCCLUDER_POLICY="${SPARSE_OCCLUDER_POLICY:-0}"
SPARSE_NUM_VIEWS="${SPARSE_NUM_VIEWS:-48}"
SPARSE_BASE_PRUNE_FRACTION="${SPARSE_BASE_PRUNE_FRACTION:-0.10}"
SPARSE_MAX_OCCLUDER_FRACTION="${SPARSE_MAX_OCCLUDER_FRACTION:-0.01}"
SPARSE_FRONT_REL_MARGIN="${SPARSE_FRONT_REL_MARGIN:-0.04}"
SPARSE_LOW_ABSREL_SUPPORT="${SPARSE_LOW_ABSREL_SUPPORT:-0.03}"
SPARSE_MIN_FACE_HITS="${SPARSE_MIN_FACE_HITS:-1}"
SPARSE_MAX_POINTS_PER_VIEW="${SPARSE_MAX_POINTS_PER_VIEW:-500}"
SPARSE_SAMPLE_MODE="${SPARSE_SAMPLE_MODE:-mixed_low_error}"
SPARSE_LOW_ERROR_FRACTION="${SPARSE_LOW_ERROR_FRACTION:-0.5}"
SPARSE_ID_PATCH_RADIUS="${SPARSE_ID_PATCH_RADIUS:-1}"
SPARSE_ADAPTIVE_GEOMETRY_BUDGET="${SPARSE_ADAPTIVE_GEOMETRY_BUDGET:-0}"
SPARSE_ADAPTIVE_FRONT_OCCLUDER_RATE_THRESHOLD="${SPARSE_ADAPTIVE_FRONT_OCCLUDER_RATE_THRESHOLD:-0.025}"
SPARSE_ADAPTIVE_MEAN_ABSREL_THRESHOLD="${SPARSE_ADAPTIVE_MEAN_ABSREL_THRESHOLD:-0.015}"
SPARSE_ADAPTIVE_HIGH_CONF_BASE_PRUNE_FRACTION="${SPARSE_ADAPTIVE_HIGH_CONF_BASE_PRUNE_FRACTION:-0.015}"
SPARSE_ADAPTIVE_HIGH_CONF_MAX_SPARSE_OCCLUDER_FRACTION="${SPARSE_ADAPTIVE_HIGH_CONF_MAX_SPARSE_OCCLUDER_FRACTION:-0.000005}"
SPARSE_ADAPTIVE_ULTRA_STABLE_MEAN_ABSREL_THRESHOLD="${SPARSE_ADAPTIVE_ULTRA_STABLE_MEAN_ABSREL_THRESHOLD:-0.010}"
SPARSE_ADAPTIVE_ULTRA_STABLE_LOW_ERROR_RATIO_FLOOR="${SPARSE_ADAPTIVE_ULTRA_STABLE_LOW_ERROR_RATIO_FLOOR:-0.95}"
SPARSE_ADAPTIVE_RENDER_STABILITY_VERTEX_REDUCTION_THRESHOLD="${SPARSE_ADAPTIVE_RENDER_STABILITY_VERTEX_REDUCTION_THRESHOLD:-0.001}"
SPARSE_ADAPTIVE_RENDER_STABILITY_BASE_PRUNE_FRACTION="${SPARSE_ADAPTIVE_RENDER_STABILITY_BASE_PRUNE_FRACTION:-0.001}"
SPARSE_ADAPTIVE_RENDER_STABILITY_FRONT_OCCLUDER_RATE_FLOOR="${SPARSE_ADAPTIVE_RENDER_STABILITY_FRONT_OCCLUDER_RATE_FLOOR:-0.02}"
SPARSE_ADAPTIVE_RENDER_STABILITY_LOW_ERROR_RATIO_CEILING="${SPARSE_ADAPTIVE_RENDER_STABILITY_LOW_ERROR_RATIO_CEILING:-0.95}"
INDOOR_POLICY_IMAGE_ARG="${INDOOR_POLICY_IMAGE_ARG:-}"

ELA_ALPHA_GRID="${ELA_ALPHA_GRID:-0,0.125,0.25,0.5,0.75,1.0}"
ELA_POLICY_MODES="${ELA_POLICY_MODES:-residual,color}"
ELA_K_VALUES="${ELA_K_VALUES:-4,8}"
ELA_DEPTH_REL_VALUES="${ELA_DEPTH_REL_VALUES:-0.06,0.12}"
ELA_RESIDUAL_CLIP_VALUES="${ELA_RESIDUAL_CLIP_VALUES:-0.20,0.25}"
ELA_DIRECTION_WEIGHT_VALUES="${ELA_DIRECTION_WEIGHT_VALUES:-0.20,0.35}"
ELA_CALIB_MAX_VIEWS="${ELA_CALIB_MAX_VIEWS:-16}"
ELA_CALIB_STRIDE="${ELA_CALIB_STRIDE:-12}"
ELA_EDGE_QUANTILE="${ELA_EDGE_QUANTILE:-0.70}"
ELA_EDGE_DILATE="${ELA_EDGE_DILATE:-1}"
ELA_UPSCALE_AUTO_ALPHA="${ELA_UPSCALE_AUTO_ALPHA:-1}"
ELA_UPSCALE_ALPHA_GRID="${ELA_UPSCALE_ALPHA_GRID:-0,0.125,0.25,0.5,0.75,1.0}"
ELA_UPSCALE_CALIB_MAX_VIEWS="${ELA_UPSCALE_CALIB_MAX_VIEWS:-16}"
ELA_UPSCALE_CALIB_LPIPS="${ELA_UPSCALE_CALIB_LPIPS:-1}"
ELA_UPSCALE_STRICT_ALL_AXIS="${ELA_UPSCALE_STRICT_ALL_AXIS:-1}"
ELA_UPSCALE_MIN_PSNR_GAIN="${ELA_UPSCALE_MIN_PSNR_GAIN:-0.0}"
ELA_UPSCALE_MIN_SSIM_GAIN="${ELA_UPSCALE_MIN_SSIM_GAIN:-0.0}"
ELA_UPSCALE_MIN_LPIPS_GAIN="${ELA_UPSCALE_MIN_LPIPS_GAIN:-0.0}"
ELA_UPSCALE_SSIM_PEAK_TOLERANCE="${ELA_UPSCALE_SSIM_PEAK_TOLERANCE:-0.0005}"
INDOOR_EVIDENCE_IMAGE_ARG="${INDOOR_EVIDENCE_IMAGE_ARG:-}"
EVIDENCE_SKIP_FAILED_VIEWS="${EVIDENCE_SKIP_FAILED_VIEWS:-0}"
EVIDENCE_FRUSTUM_CULL="${EVIDENCE_FRUSTUM_CULL:-0}"
EVIDENCE_FRUSTUM_MARGIN="${EVIDENCE_FRUSTUM_MARGIN:-0.5}"

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
  evidence_image_arg="${image_arg}"
  policy_image_arg="${image_arg}"
  if [[ -n "${INDOOR_EVIDENCE_IMAGE_ARG}" ]]; then
    case "${scene}" in
      room|counter|kitchen|bonsai)
        evidence_image_arg="${INDOOR_EVIDENCE_IMAGE_ARG}"
        ;;
    esac
  fi
  if [[ -n "${INDOOR_POLICY_IMAGE_ARG}" ]]; then
    case "${scene}" in
      room|counter|kitchen|bonsai)
        policy_image_arg="${INDOOR_POLICY_IMAGE_ARG}"
        ;;
    esac
  fi
  evidence_method="ours_${COMPACT_ITERATION}"
  lowres_ela_method="${METHOD_NAME}"
  use_lowres_evidence=0
  if [[ "${evidence_image_arg}" != "${image_arg}" ]]; then
    use_lowres_evidence=1
    evidence_method="ours_${COMPACT_ITERATION}_${evidence_image_arg}_evidence"
    lowres_ela_method="${METHOD_NAME}_${evidence_image_arg}_lowres"
  fi

  out_dir="${OUT_ROOT}/${scene}/${POLICY_TAG}"
  compact_model="${out_dir}/compact_model"
  mkdir -p "${out_dir}/logs"
  selector_json="${out_dir}/selector/compaction_candidates.json"

  if [[ "${SPARSE_OCCLUDER_POLICY}" == "1" && ! -f "${selector_json}" ]]; then
    echo "[M360 compact-ELA] build train-only sparse occluder candidates scene=${scene}"
    sparse_args=(
      -s "${source_path}"
      -m "${clean_model}"
      -i "${policy_image_arg}"
      --resolution -1
      --eval
      --source_model "${clean_model}"
      --iteration "${COMPACT_ITERATION}"
      --split train
      --num_views "${SPARSE_NUM_VIEWS}"
      --base_prune_fraction "${SPARSE_BASE_PRUNE_FRACTION}"
      --max_sparse_occluder_fraction "${SPARSE_MAX_OCCLUDER_FRACTION}"
      --front_rel_margin "${SPARSE_FRONT_REL_MARGIN}"
      --low_absrel_support "${SPARSE_LOW_ABSREL_SUPPORT}"
      --min_face_hits "${SPARSE_MIN_FACE_HITS}"
      --max_points_per_view "${SPARSE_MAX_POINTS_PER_VIEW}"
      --sample_mode "${SPARSE_SAMPLE_MODE}"
      --low_error_fraction "${SPARSE_LOW_ERROR_FRACTION}"
      --id_patch_radius "${SPARSE_ID_PATCH_RADIUS}"
      --seed "${SELECTOR_SEED}"
      --output_dir "${out_dir}/selector"
    )
    if [[ "${SPARSE_ADAPTIVE_GEOMETRY_BUDGET}" == "1" ]]; then
      sparse_args+=(
        --adaptive_geometry_budget
        --adaptive_front_occluder_rate_threshold "${SPARSE_ADAPTIVE_FRONT_OCCLUDER_RATE_THRESHOLD}"
        --adaptive_mean_absrel_threshold "${SPARSE_ADAPTIVE_MEAN_ABSREL_THRESHOLD}"
        --adaptive_high_conf_base_prune_fraction "${SPARSE_ADAPTIVE_HIGH_CONF_BASE_PRUNE_FRACTION}"
        --adaptive_high_conf_max_sparse_occluder_fraction "${SPARSE_ADAPTIVE_HIGH_CONF_MAX_SPARSE_OCCLUDER_FRACTION}"
        --adaptive_ultra_stable_mean_absrel_threshold "${SPARSE_ADAPTIVE_ULTRA_STABLE_MEAN_ABSREL_THRESHOLD}"
        --adaptive_ultra_stable_low_error_ratio_floor "${SPARSE_ADAPTIVE_ULTRA_STABLE_LOW_ERROR_RATIO_FLOOR}"
        --adaptive_render_stability_vertex_reduction_threshold "${SPARSE_ADAPTIVE_RENDER_STABILITY_VERTEX_REDUCTION_THRESHOLD}"
        --adaptive_render_stability_base_prune_fraction "${SPARSE_ADAPTIVE_RENDER_STABILITY_BASE_PRUNE_FRACTION}"
        --adaptive_render_stability_front_occluder_rate_floor "${SPARSE_ADAPTIVE_RENDER_STABILITY_FRONT_OCCLUDER_RATE_FLOOR}"
        --adaptive_render_stability_low_error_ratio_ceiling "${SPARSE_ADAPTIVE_RENDER_STABILITY_LOW_ERROR_RATIO_CEILING}"
      )
    fi
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_build_sparse_occluder_prune_candidates.py \
      "${sparse_args[@]}" \
      2>&1 | tee "${out_dir}/logs/sparse_candidates.log"
  fi

  if [[ ! -f "${compact_model}/point_cloud/iteration_${COMPACT_ITERATION}/point_cloud_state_dict.pt" ]]; then
    echo "[M360 compact-ELA] compact scene=${scene} iter=${COMPACT_ITERATION}"
    compaction_args=(
      --source_model "${clean_model}"
      --iteration "${COMPACT_ITERATION}"
      --output_model "${compact_model}"
      --seed "${SELECTOR_SEED}"
    )
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

  if [[ ! -f "${compact_model}/train/${evidence_method}/camera_index.json" || ! -d "${compact_model}/train/${evidence_method}/depths" || ! -f "${compact_model}/test/${evidence_method}/camera_index.json" || ! -d "${compact_model}/test/${evidence_method}/depths" ]]; then
    echo "[M360 compact-ELA] render RGB/depth evidence scene=${scene} images=${evidence_image_arg} method=${evidence_method}"
    evidence_args=(
      -s "${source_path}" \
      -i "${evidence_image_arg}" \
      -m "${compact_model}" \
      --resolution -1 \
      --eval \
      --iteration "${COMPACT_ITERATION}" \
      --method_name "${evidence_method}" \
      --quiet
    )
    if [[ "${EVIDENCE_SKIP_FAILED_VIEWS}" == "1" ]]; then
      evidence_args+=(--skip_failed_views)
    fi
    if [[ "${EVIDENCE_FRUSTUM_CULL}" == "1" ]]; then
      evidence_args+=(--frustum_cull --frustum_margin "${EVIDENCE_FRUSTUM_MARGIN}")
    fi
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_render_evidence_maps.py \
      "${evidence_args[@]}" \
      2>&1 | tee "${out_dir}/logs/evidence_render.log"
  fi

  if [[ "${use_lowres_evidence}" == "1" ]]; then
    full_base_common_args=(
      -s "${source_path}" \
      -i "${image_arg}" \
      -m "${compact_model}" \
      --resolution -1 \
      --eval \
      --iteration "${COMPACT_ITERATION}" \
      --no_depth \
      --method_name "ours_${COMPACT_ITERATION}" \
      --quiet
    )
    if [[ "${EVIDENCE_SKIP_FAILED_VIEWS}" == "1" ]]; then
      full_base_common_args+=(--skip_failed_views)
    fi
    if [[ "${EVIDENCE_FRUSTUM_CULL}" == "1" ]]; then
      full_base_common_args+=(--frustum_cull --frustum_margin "${EVIDENCE_FRUSTUM_MARGIN}")
    fi
    if [[ ! -d "${compact_model}/test/ours_${COMPACT_ITERATION}/renders" || ! -d "${compact_model}/test/ours_${COMPACT_ITERATION}/gt" ]]; then
      echo "[M360 compact-ELA] render full-res compact test base scene=${scene} images=${image_arg}"
      "${PYTHON_BIN}" scripts/car_model/meshsplatopt_render_evidence_maps.py \
        "${full_base_common_args[@]}" \
        --skip_train \
        2>&1 | tee "${out_dir}/logs/full_base_test_render.log"
    fi
    if [[ "${ELA_UPSCALE_AUTO_ALPHA}" == "1" && ( ! -d "${compact_model}/train/ours_${COMPACT_ITERATION}/renders" || ! -d "${compact_model}/train/ours_${COMPACT_ITERATION}/gt" ) ]]; then
      echo "[M360 compact-ELA] render full-res compact train base for alpha calibration scene=${scene} images=${image_arg}"
      "${PYTHON_BIN}" scripts/car_model/meshsplatopt_render_evidence_maps.py \
        "${full_base_common_args[@]}" \
        --skip_test \
        2>&1 | tee "${out_dir}/logs/full_base_train_render.log"
    fi
  fi

  if [[ ! -f "${compact_model}/test/${lowres_ela_method}/ela_report.json" ]]; then
    echo "[M360 compact-ELA] apply train-calibrated ELA scene=${scene} base=${evidence_method} out=${lowres_ela_method}"
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
      --base_model_path "${compact_model}" \
      --iteration "${COMPACT_ITERATION}" \
      --base_method_name "${evidence_method}" \
      --target_split test \
      --method_name "${lowres_ela_method}" \
      --auto_policy \
      --policy_modes "${ELA_POLICY_MODES}" \
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

  if [[ "${use_lowres_evidence}" == "1" && "${ELA_UPSCALE_AUTO_ALPHA}" == "1" && ! -f "${compact_model}/train/${lowres_ela_method}/ela_report.json" ]]; then
    echo "[M360 compact-ELA] apply train-calibrated ELA for upsample-alpha calibration scene=${scene} base=${evidence_method} out=${lowres_ela_method}"
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
      --base_model_path "${compact_model}" \
      --iteration "${COMPACT_ITERATION}" \
      --base_method_name "${evidence_method}" \
      --target_split train \
      --method_name "${lowres_ela_method}" \
      --auto_policy \
      --policy_modes "${ELA_POLICY_MODES}" \
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
      --wandb_name "compact_ela_train_${scene}_${POLICY_TAG}_${COMPACT_ITERATION}" \
      2>&1 | tee "${out_dir}/logs/ela_train.log"
  fi

  if [[ "${use_lowres_evidence}" == "1" && ! -f "${compact_model}/test/${METHOD_NAME}/ela_upscale_report.json" ]]; then
    echo "[M360 compact-ELA] upscale low-res ELA residual scene=${scene} low=${lowres_ela_method} full=${METHOD_NAME}"
    upscale_args=(
      --model_path "${compact_model}" \
      --split test \
      --lowres_base_method "${evidence_method}" \
      --lowres_ela_method "${lowres_ela_method}" \
      --full_base_method "ours_${COMPACT_ITERATION}" \
      --output_method "${METHOD_NAME}" \
      --resize_mode bilinear
    )
    if [[ "${ELA_UPSCALE_AUTO_ALPHA}" == "1" ]]; then
      upscale_args+=(
        --auto_alpha
        --alpha_grid "${ELA_UPSCALE_ALPHA_GRID}"
        --calib_split train
        --calib_max_views "${ELA_UPSCALE_CALIB_MAX_VIEWS}"
        --policy_ssim_weight 20.0
        --policy_lpips_weight 20.0
        --strict_alpha_min_psnr_gain "${ELA_UPSCALE_MIN_PSNR_GAIN}"
        --strict_alpha_min_ssim_gain "${ELA_UPSCALE_MIN_SSIM_GAIN}"
        --strict_alpha_min_lpips_gain "${ELA_UPSCALE_MIN_LPIPS_GAIN}"
        --alpha_ssim_peak_tolerance "${ELA_UPSCALE_SSIM_PEAK_TOLERANCE}"
      )
      if [[ "${ELA_UPSCALE_CALIB_LPIPS}" == "1" ]]; then
        upscale_args+=(--calib_lpips)
      fi
      if [[ "${ELA_UPSCALE_STRICT_ALL_AXIS}" == "1" ]]; then
        upscale_args+=(--strict_all_axis_alpha)
      fi
    fi
    "${PYTHON_BIN}" scripts/car_model/meshsplatopt_upscale_ela_residual_to_full.py \
      "${upscale_args[@]}" \
      2>&1 | tee "${out_dir}/logs/ela_upscale.log"
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
