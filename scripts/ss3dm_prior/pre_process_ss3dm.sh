#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/peilincai/mesh-splatting"
RAW_ROOT="/path/to/SS3DM_raw"

MANIFEST="$ROOT/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest_v4.json"
OBSERVED_CACHE_DIR="$ROOT/outputs/ss3dm_prior/observed_cache_v4"
TOWN_MESH_CACHE_DIR="$ROOT/outputs/ss3dm_prior/town_mesh_cache"
PATCH_CACHE_DIR="$ROOT/outputs/ss3dm_prior/teacher_patch_cache_v3_v4"

python -m ss3dm_prior.tools.build_manifest \
  --root "$RAW_ROOT" \
  --out "$MANIFEST"

python -m ss3dm_prior.tools.build_observed_cache \
  --manifest "$MANIFEST" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/default_town_split.yaml" \
  --config "$ROOT/configs/ss3dm_prior/observed_cache_default.yaml" \
  --out_dir "$OBSERVED_CACHE_DIR" \
  --subsets train val test

python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 \
  --manifest "$MANIFEST" \
  --split_config "$ROOT/configs/ss3dm_prior/splits/default_town_split.yaml" \
  --config "$ROOT/configs/ss3dm_prior/teacher_patch_v3.yaml" \
  --observed_cache_dir "$OBSERVED_CACHE_DIR" \
  --town_mesh_cache_root "$TOWN_MESH_CACHE_DIR" \
  --out_dir "$PATCH_CACHE_DIR" \
  --subsets train val test \
  --num_workers 8 \
  --seed 0

python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 \
  --patch_cache_dir "$PATCH_CACHE_DIR" \
  --num_visualizations 8 \
  --seed 0
