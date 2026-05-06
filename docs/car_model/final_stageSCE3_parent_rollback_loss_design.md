# Final Stage SCE3 - One-Sided Parent Rollback Loss Design

Date: 2026-05-06

## Goal

SCE3 adds an opt-in sparse-depth parent rollback loss. It consumes a train/calibration sentinel cache and penalizes only current sparse-depth regressions relative to the parent checkpoint. It does not pull depth toward the parent everywhere and it does not penalize improvements over the parent.

## Loss

For each cached sentinel correspondence:

```text
parent_abs = abs(parent_pred_depth - gt_depth)
current_abs = abs(current_pred_depth - gt_depth)
parent_rel = parent_abs / max(gt_depth, eps)
current_rel = current_abs / max(gt_depth, eps)
```

The active violation is one-sided:

```text
absrel:   ReLU(current_rel - parent_rel - margin_rel)
mae:      ReLU(current_abs - parent_abs - margin_abs)
combined: ReLU(current_rel - parent_rel - margin_rel)
        + ReLU(current_abs - parent_abs - margin_abs)
```

The final loss is a weighted Smooth-L1 average over sentinel points, multiplied by `lambda_sparse_depth_parent_rollback`.

## Safety Rules

- Disabled by default.
- Cache split `test` raises `RuntimeError` unless explicitly overridden by diagnostic flag.
- Missing camera key returns zero loss and logs `missing_camera_key`.
- Missing `surf_depth` returns zero loss unless strict mode is enabled.
- Parent-valid filtering is performed by SCE2 cache construction.
- The loss is compatible with the existing sparse COLMAP depth loss and strict topology-frozen recovery.

## New Flags

```text
--enable_sparse_depth_parent_rollback_loss
--sparse_depth_parent_rollback_cache <path>
--lambda_sparse_depth_parent_rollback <float>
--sparse_depth_parent_rollback_start_iter <int>
--sparse_depth_parent_rollback_warmup_iters <int>
--sparse_depth_parent_rollback_margin_abs <float>
--sparse_depth_parent_rollback_margin_rel <float>
--sparse_depth_parent_rollback_huber_delta <float>
--sparse_depth_parent_rollback_cluster_balance
--sparse_depth_parent_rollback_max_points_per_view <int>
--sparse_depth_parent_rollback_loss_space absrel|mae|combined
--sparse_depth_parent_rollback_allow_test_cache
--sparse_depth_parent_rollback_strict
```

The strict recovery wrapper exposes the main training flags and records them in `recovery_summary.json` and `exact_train_command.txt`.

