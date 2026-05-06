# Final Stage SCE1 - Sparse-Depth Regression Analyzer Design

Date: 2026-05-06

Decision target: compare a parent checkpoint and a candidate checkpoint on the exact COLMAP sparse correspondence distribution before launching more recovery runs.

## Motivation

F95 improves courtyard RGB, per-view PSNR, LPIPS, and sparse-normal proxy over F82, but fails parent-Pareto because sparse AbsRel and Depth MAE regress. Aggregate geometry JSON cannot tell whether this is a broad depth drift or a small number of gate-critical correspondences. SCE1 makes sparse correspondences first-class evidence.

## Interface

```bash
python scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py \
  --source_path <scene_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_model_path> \
  --parent_iteration 26000 \
  --candidate_model_path <F95_model_path> \
  --candidate_iteration 27000 \
  --split test \
  --max_points_per_view 500 \
  --point_error_max 2.0 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --seed 7 \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE1_sparse_depth_regression/<scene>
```

## Data Flow

1. Load parent and candidate `TriangleModel` checkpoints independently.
2. Build the same COLMAP geometry proxy context used by `evaluate_geometry_colmap.py`.
3. Select train, test, calibration, or all views.
4. For each selected view, collect sparse correspondences once through `collect_view_sparse_depth_correspondences`.
5. Render parent and candidate on the same view and sample `surf_depth` at the same `(px, py)` correspondence pixels.
6. Compute parent/candidate absolute error, AbsRel, candidate-parent deltas, validity flags, regression masks, boundary/depth/error bins, and simple 2D grid clusters for gate-critical points.

## Regression Masks

- `regressed_abs`: `candidate_abs_error - parent_abs_error > margin_abs`, plus parent-valid/candidate-invalid points.
- `regressed_rel`: `candidate_abs_rel - parent_abs_rel > margin_rel`, plus parent-valid/candidate-invalid points.
- `gate_critical`: union of top positive `delta_abs_error`, top positive `delta_abs_rel`, and candidate-invalid points. Default top fraction is `10%`.

Positive deltas always mean candidate is worse than parent.

## Outputs

The analyzer writes:

- `correspondence_regressions.csv`
- `correspondence_regressions.npz`
- `per_view_regression_summary.csv`
- `point_regression_summary.csv`
- `cluster_regression_summary.csv`
- `sentinel_candidate_mask.npz`
- `regression_report.md`
- `regression_summary.json`

Every output records `split=train|calibration|test|all`. Test-split outputs are diagnostic only and must not be used as a training sentinel cache.

## Smoke Test

`scripts/car_model/smoke_test_stageSCE1_sparse_depth_regression.py` constructs synthetic correspondences with known parent/candidate depths, including an invalid candidate point. It verifies:

- candidate-parent deltas,
- one absolute and relative regression,
- invalid candidate accounting,
- per-view and per-point aggregation,
- gate-critical cluster construction,
- all required output files.

