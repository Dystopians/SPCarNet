# Final Stage SCE7 Automatic SCE Policy Design

Date: 2026-05-06

Decision: `SCE7_INTERFACE_IMPLEMENTED_PENDING_MULTISCENE_VALIDATION`

## Goal

SCE7 converts the manual dense-sentinel repair lesson into a reusable policy. The policy is scene-agnostic: scene-specific sentinel caches are allowed, but thresholds and recovery knobs are fixed.

## Policy v1

The policy uses the following fixed behavior:

1. Run or load a train/calibration sentinel gate.
2. If sentinel aggregate AbsRel or MAE regresses, activate one-sided parent rollback.
3. Use a short high-vertex-LR geometry phase, because SCE6 showed default late-stage vertex LR is too small to move geometry.
4. Use absrel rollback by default; MAE-only rollback is rejected by SCE6 evidence.
5. Use early stopping to pick the first parent-Pareto-safe knee instead of blindly extending training.

Default knobs:

- rollback loss: `absrel`
- rollback lambda: `1.0`
- sparse COLMAP lambda: `0.003`
- render normal anchor: `0.01`
- render depth anchor: `0.0`
- vertex LR init: `0.015`
- phase length: `500`

## Interfaces

- `utils/sce_recovery_policy.py`
- `scripts/car_model/meshsplatopt_run_sce_policy_recovery.py`
- `scripts/car_model/smoke_test_stageSCE7_sce_policy.py`

The runner writes:

- `sce_policy_decision.json`
- `exact_train_command.txt`
- `policy_report.md`

## Current Limitation

SCE7 is an interface and policy decision layer. It does not claim final multiscene success until SCE8 runs the fixed policy across the selected scenes. The current courtyard evidence is strong partial: RGB, AbsRel, and normal pass F82, while Depth MAE remains slightly above F82.

## Verification

- Compile: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q`
- Smoke: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE7_sce_policy.py`
- Contract run: `outputs/carnet/meshsplatopt/final_stageSCE7_automatic_sce_policy/courtyard/contract`

The contract run consumed the dense F82-vs-F95 sentinel gate and correctly selected `run_targeted_rollback` with `activate_rollback=true`, writing a strict recovery command with fixed SCE v1 defaults.
