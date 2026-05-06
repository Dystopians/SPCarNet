# Final Stage SCE3 - One-Sided Parent Rollback Loss Report

Date: 2026-05-06

Decision: `PASS`

## Implementation

Files added or modified:

- `utils/sparse_depth_parent_rollback.py`
- `scripts/car_model/smoke_test_stageSCE3_parent_rollback_loss.py`
- `arguments/__init__.py`
- `train.py`
- `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`
- `docs/car_model/final_stageSCE3_parent_rollback_loss_design.md`

The implementation is opt-in and zero-cost when disabled. When enabled, `train.py` loads a SCE2 train/calibration sentinel cache, samples the current `surf_depth` at cached sparse correspondence pixels, and adds a one-sided loss only where current error exceeds parent error by configured margins.

## Smoke Test

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE3_parent_rollback_loss.py
```

Result: `SCE3 sparse-depth parent rollback loss smoke test PASS`.

The smoke test verifies:

- equal-to-parent current depth gives zero loss;
- improved current depth gives zero loss;
- worse current depth gives positive loss;
- sentinel weights change the weighted mean;
- missing camera key skips cleanly;
- test-split cache loading raises `RuntimeError`.

## Wrapper Contract Smoke

The strict recovery wrapper was run in non-execute mode against the F95 courtyard checkpoint with the SCE2 train cache. It wrote a command contract under:

`outputs/carnet/meshsplatopt/final_stageSCE3_parent_rollback_loss/contract_smoke`

The generated `exact_train_command.txt` includes:

- `--enable_sparse_depth_parent_rollback_loss`
- `--sparse_depth_parent_rollback_cache outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_cache.npz`
- `--lambda_sparse_depth_parent_rollback 0.001`
- `--sparse_depth_parent_rollback_loss_space combined`

## Logging

The training loop logs:

- `loss_components/loss_sparse_parent_rollback`
- `loss_components/loss_sparse_parent_rollback_pure`
- `sparse_parent_rollback/lambda`
- `sparse_parent_rollback/active_points`
- `sparse_parent_rollback/total_points`
- `sparse_parent_rollback/active_fraction`
- `sparse_parent_rollback/mean_violation_rel`
- `sparse_parent_rollback/max_violation_rel`
- `sparse_parent_rollback/mean_violation_abs`
- `sparse_parent_rollback/max_violation_abs`

## Gate

SCE3 passes because the loss is opt-in, disabled by default, refuses test caches by default, and unit tests prove the one-sided behavior. The next stage is SCE4: sentinel-aware parent-Pareto gate.

