# Final Stage SCE1 - Sparse-Depth Regression Analyzer Report

Date: 2026-05-06

Decision: `PASS`

## Implementation

SCE1 adds the sparse-depth parent-vs-candidate regression analyzer required by the SCE-Repair plan.

Files added or modified:

- `utils/sparse_depth_regression.py`
- `scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py`
- `scripts/car_model/smoke_test_stageSCE1_sparse_depth_regression.py`
- `utils/prism_geometry_proxy.py`
- `docs/car_model/final_stageSCE1_sparse_depth_regression_analyzer_design.md`

The existing `collect_view_sparse_depth_correspondences` helper now returns `point3D_id`, which is required for point-level aggregation and later sentinel cache construction.

## Smoke Test

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE1_sparse_depth_regression.py
```

Result: `SCE1 sparse-depth regression smoke test PASS`.

The smoke test verifies synthetic deltas, absolute/relative regression masks, invalid candidate accounting, per-view and per-point aggregation, cluster construction, and all required output files.

## Real Diagnostic: Courtyard F82 vs F95

GPU selection before the real run showed GPU 7 as the lowest-use device (`2657 / 49140 MiB`, `0%` util), so the analyzer was run with `CUDA_VISIBLE_DEVICES=7`.

Command:

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py \
  --source_path /data/peilincai/mesh_datasets/eth3d_colmap/courtyard \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/courtyard/adaptive_global_policy_v5_seed0/recovery_model \
  --parent_iteration 26000 \
  --candidate_model_path outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard/adaptive_global_policy_v5_teacher0p001_sparse0p001_rendergeom0p01_27000_seed0/recovery_model \
  --candidate_iteration 27000 \
  --split test \
  --max_points_per_view 500 \
  --point_error_max 2.0 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --seed 7 \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE1_sparse_depth_regression/courtyard
```

Output directory:

`outputs/carnet/meshsplatopt/final_stageSCE1_sparse_depth_regression/courtyard`

Required files were produced:

- `correspondence_regressions.csv`
- `correspondence_regressions.npz`
- `per_view_regression_summary.csv`
- `point_regression_summary.csv`
- `cluster_regression_summary.csv`
- `sentinel_candidate_mask.npz`
- `regression_report.md`
- `regression_summary.json`

## Result

The real diagnostic reproduces the known F95-vs-F82 sparse-depth failure direction.

| metric | F82 parent | F95 candidate | candidate - parent |
|---|---:|---:|---:|
| AbsRel | 0.324888045 | 0.325786638 | +0.000898593 |
| Depth MAE | 3.516864341 | 3.533150427 | +0.016286086 |

Analyzer sample summary:

- correspondences: `2500`
- both-valid correspondences: `1961`
- candidate-invalid correspondences: `8`
- gate-critical correspondences: `98`
- regressed AbsRel correspondences at zero margin: `1315`
- regressed MAE correspondences at zero margin: `1315`

Per-view summary shows that the regression is not uniform:

- `dsc_0286`: candidate improves AbsRel and MAE.
- `dsc_0310`: candidate improves AbsRel and MAE.
- `dsc_0294`, `dsc_0302`, and especially `dsc_0318`: candidate regresses sparse depth.

This confirms the SCE0 diagnosis: the blocker is localized correspondence-level sparse-depth regression, not a global RGB or per-view render failure.

## Gate

SCE1 passes because:

1. the smoke test passes;
2. local F82/F95 courtyard artifacts exist;
3. the real analyzer reproduces the known sparse-depth / AbsRel direction: F95 is worse than F82 on sparse depth even though it is visually stronger.

The next required stage is SCE2: build a deterministic train/calibration sentinel cache without test leakage. The SCE1 test output is diagnostic only and must not be used for training.

