#!/usr/bin/env bash
set -euo pipefail

SCENES_CSV="${SCENES:-bicycle,flowers,garden,treehill,room}"
MAX_VIEWS="${MAX_VIEWS:-8}"
VIEW_STRIDE="${VIEW_STRIDE:-6}"
VIEW_OFFSET="${VIEW_OFFSET:-0}"
ITERATION="${ITERATION:-26000}"
OUT_DIR="${OUT_DIR:-outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence}"
METHOD_ROOT="${METHOD_ROOT:-outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k}"
POLICY_TAG="${POLICY_TAG:-sor_adaptive_geo}"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/miniconda3/envs/Difix/bin/python}"

IFS=',' read -r -a SCENES <<< "$SCENES_CSV"

for scene in "${SCENES[@]}"; do
  model_path="${METHOD_ROOT}/${scene}/${POLICY_TAG}/compact_model"
  "${PYTHON_BIN}" scripts/car_model/ecsr_build_surface_evidence_cache.py \
    -m "${model_path}" \
    --iteration "${ITERATION}" \
    --split train \
    --scene_name "${scene}" \
    --out_dir "${OUT_DIR}" \
    --max_views "${MAX_VIEWS}" \
    --view_stride "${VIEW_STRIDE}" \
    --view_offset "${VIEW_OFFSET}" \
    --save_view_npz
done
