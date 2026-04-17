#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
DATASET_ROOT="${DATASET_ROOT:-/data2/car_meshes/MeshFleet_TRELLIS_RECONSTRUCTED}"
MESH_ROOT="${MESH_ROOT:-$DATASET_ROOT}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python -m ss3dm_prior.tools.build_car_mesh_patch_cache \
  --dataset_root "$DATASET_ROOT" \
  --mesh_root "$MESH_ROOT" \
  --out_dir "$OUT_DIR" \
  --val_fraction 0.1 \
  --clean_sample_count 2048 \
  --observed_sample_count 768 \
  --normalized_radius 1.0 \
  --observed_view_count 3 \
  --min_visibility_cosine 0.05 \
  --num_workers "$NUM_WORKERS" \
  --seed 0 \
  --skip_existing
