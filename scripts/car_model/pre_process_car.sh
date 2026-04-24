#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"

DATASET_ROOT="${DATASET_ROOT:-/data2/car_meshes/MeshFleet_TRELLIS}"
RECON_ROOT="${RECON_ROOT:-/data2/car_meshes/MeshFleet_TRELLIS_RECONSTRUCTED_v4}"
CAR_CACHE_DIR="${CAR_CACHE_DIR:-$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$RECON_ROOT/train" "$RECON_ROOT/test"

python "$DATASET_ROOT/reconstruct_data.py" \
  --shard_dir "$DATASET_ROOT/train" \
  --output_dir "$RECON_ROOT/train"

python "$DATASET_ROOT/reconstruct_data.py" \
  --shard_dir "$DATASET_ROOT/test" \
  --output_dir "$RECON_ROOT/test"

if [ -f "$DATASET_ROOT/train/metadata.csv" ]; then
  cp "$DATASET_ROOT/train/metadata.csv" "$RECON_ROOT/train/metadata.csv"
fi

if [ -f "$DATASET_ROOT/test/metadata.csv" ]; then
  cp "$DATASET_ROOT/test/metadata.csv" "$RECON_ROOT/test/metadata.csv"
fi

python -m ss3dm_prior.tools.build_car_mesh_patch_cache \
  --dataset_root "$RECON_ROOT" \
  --mesh_root "$RECON_ROOT" \
  --out_dir "$CAR_CACHE_DIR" \
  --val_fraction 0.1 \
  --clean_sample_count 2048 \
  --observed_sample_count 768 \
  --normalized_radius 1.0 \
  --observed_view_count 3 \
  --min_visibility_cosine 0.05 \
  --clean_visibility_cosine 0.5 \
  --surface_query_count 512 \
  --free_query_count 512 \
  --unknown_query_count 256 \
  --query_ball_radius 1.1 \
  --surface_query_eps 0.025 \
  --free_query_eps 0.04 \
  --hard_negative_count 128 \
  --num_workers "$NUM_WORKERS" \
  --seed 0 \
  --skip_existing
