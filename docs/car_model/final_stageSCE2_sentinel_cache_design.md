# Final Stage SCE2 - Train/Calibration Sparse-Depth Sentinel Cache Design

Date: 2026-05-06

## Goal

SCE2 builds a deterministic sparse-depth sentinel cache from train or calibration views only. It is the data interface for later one-sided parent rollback loss. Test views are explicitly rejected by the builder to prevent leakage.

## Interface

```bash
python scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py \
  --source_path <scene_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_model_path> \
  --parent_iteration 26000 \
  --candidate_model_path <optional_candidate_model_path> \
  --candidate_iteration <optional_candidate_iter> \
  --split train \
  --num_views 32 \
  --prefer_hard_views \
  --prefer_observable_views \
  --max_points_per_view 500 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --point_error_max 2.0 \
  --output outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/<scene>/sentinel_cache.npz
```

## Cache Content

The `.npz` cache stores:

- `image_name`, `image_key`
- `px`, `py`
- `gt_depth`
- `point3D_id`
- `parent_pred_depth`
- `parent_abs_error`
- `parent_abs_rel`
- optional candidate prediction and regression deltas
- `sentinel_weight`
- `cluster_id`
- `is_regressed_candidate`
- `manifest_json`

The sidecar `sentinel_manifest.json` records source path, split, parent/candidate checkpoint identity, view-selection reasons, seed, sampling parameters, and `no_test_leakage=true`.

## Selection Policy

1. Only train or calibration views are accepted.
2. Observable views can be preferred using COLMAP sparse match counts.
3. If a train/calibration regression report is supplied, hard views can be oversampled through `--prefer_hard_views`.
4. Within selected views, correspondence sampling uses the same deterministic `GeometryProxyConfig` options as the geometry evaluator.
5. Candidate regressions increase sentinel weights.
6. Cluster balancing rescales weights by local grid-cluster density rather than silently dropping hard points.

## Determinism

The cache builder stores the seed and uses deterministic view ordering, correspondence sampling, cluster IDs, and weight construction. Re-running the smoke test with the same synthetic inputs produces identical weights and masks.

## Leakage Rule

`--split test` raises an error. Test outputs from SCE1 are diagnostic only and cannot become training sentinel caches.

