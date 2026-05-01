#!/usr/bin/env bash
# Per-corruption + residual probe sweep on v0.8.2 best_recon.pt.
# Mirrors the v0.7 ablation in docs/car_model/carnet_v0_8_diagnosis.md so the two are directly comparable.

set -euo pipefail

ROOT="/data/peilincai/mesh-splatting"
CKPT="$ROOT/outputs/carnet/v0_8_2/full/checkpoints/best_recon.pt"
CACHE="$ROOT/outputs/ss3dm_prior_car/meshfleet_car_cache_v5"
SPLIT="$CACHE/split_meshfleet_car.yaml"
OUT="$ROOT/outputs/carnet/v0_8_2/eval_ablation"
PYTHON_BIN="${PYTHON_BIN:-/home/peilincai/micromamba/envs/mesh_splatting/bin/python}"
GPU="${GPU:-5}"

mkdir -p "$OUT"
echo "[diag] checkpoint=$CKPT" >&2
echo "[diag] output_dir=$OUT" >&2

PROFILES=(zero default only_point_dropout only_gaussian_jitter only_normal_noise only_local_hole_mask only_outlier_cluster only_density_imbalance)

for p in "${PROFILES[@]}"; do
  echo "[diag] profile=$p" >&2
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m ss3dm_prior.tools.diagnose_carnet \
    --checkpoint "$CKPT" \
    --patch_cache_dir "$CACHE" \
    --split_config "$SPLIT" \
    --output_dir "$OUT" \
    --eval_name "$p" \
    --profile "$p"
done

# Residual direction probe on default (50 patches).
echo "[diag] probe=default_probe" >&2
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m ss3dm_prior.tools.diagnose_carnet \
  --checkpoint "$CKPT" \
  --patch_cache_dir "$CACHE" \
  --split_config "$SPLIT" \
  --output_dir "$OUT" \
  --eval_name default_probe \
  --profile default \
  --probe_points \
  --probe_max_patches 50

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m ss3dm_prior.tools.analyze_probe \
  --probe_npz "$OUT/default_probe/probe_points.npz" \
  --output_dir "$OUT/default_probe"

echo "[diag] DONE -> $OUT" >&2
