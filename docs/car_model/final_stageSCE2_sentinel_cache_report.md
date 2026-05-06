# Final Stage SCE2 - Sparse-Depth Sentinel Cache Report

Date: 2026-05-06

Decision: `PASS`

## Implementation

SCE2 adds a deterministic train/calibration sparse-depth sentinel cache builder.

Files added:

- `utils/sparse_depth_sentinel_cache.py`
- `scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py`
- `scripts/car_model/smoke_test_stageSCE2_sentinel_cache.py`
- `docs/car_model/final_stageSCE2_sentinel_cache_design.md`

The builder rejects `--split test`, stores `no_test_leakage=true` in the manifest, keeps only parent-valid sparse correspondences, and writes sentinel weights plus cluster IDs for later one-sided rollback loss.

## Smoke Test

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE2_sentinel_cache.py
```

Result: `SCE2 sparse-depth sentinel cache smoke test PASS`.

The smoke test verifies deterministic weights under a fixed seed, candidate-regression flags, cluster-balanced weights, required output files, and test-split rejection.

## Real Courtyard Train Cache

Command:

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py \
  --source_path /data/peilincai/mesh_datasets/eth3d_colmap/courtyard \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/courtyard/adaptive_global_policy_v5_seed0/recovery_model \
  --parent_iteration 26000 \
  --candidate_model_path outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard/adaptive_global_policy_v5_teacher0p001_sparse0p001_rendergeom0p01_27000_seed0/recovery_model \
  --candidate_iteration 27000 \
  --split train \
  --num_views 32 \
  --prefer_hard_views \
  --prefer_observable_views \
  --max_points_per_view 500 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --point_error_max 2.0 \
  --seed 7 \
  --cluster_balance \
  --output outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_cache.npz
```

Outputs:

- `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_cache.npz`
- `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_manifest.json`
- `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_view_summary.csv`
- `outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_report.md`

Manifest summary:

- split: `train`
- no_test_leakage: `true`
- selected views: `32`
- parent-valid sentinels: `13630`
- candidate-regressed sentinels: `4985`
- seed: `7`
- cluster balancing: `true`

## Gate

SCE2 passes because the cache generation is deterministic in smoke testing, the real cache explicitly records `no_test_leakage=true`, and test-split cache construction is rejected. The next stage is SCE3: implement the opt-in one-sided parent rollback sparse-depth loss that consumes this cache.

