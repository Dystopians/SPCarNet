# Stage SCE19 Policy Guard Report

Date: 2026-05-06

Decision: `SCE_POLICY_GUARD_IMPLEMENTED_BONSAI_NEGATIVE_CAUGHT`

## Motivation

The first SCE8 bonsai fixed-policy probe improved sparse depth but worsened RGB, SSIM, LPIPS, and slightly worsened normal. That failure exposes a policy weakness: sentinel geometry can improve while the candidate is still not acceptable as a general replacement for F82.

SCE19 hardens the automatic policy with opt-in recovery guards:

- require a sentinel gate before executing guarded recovery
- require measured parent/candidate render metrics before executing guarded recovery
- reject candidates whose PSNR or SSIM drop, LPIPS increases, or combined render score falls below the configured threshold
- optionally require full parent-Pareto non-regression across PSNR, SSIM, LPIPS, AbsRel, Depth MAE, and normal before accepting a measured candidate
- return `accept_parent_noop` instead of launching another recovery run when guards fail

The original `sce_v1` defaults remain unchanged. The new behavior is opt-in through policy config or CLI flags.

## Implementation

Updated files:

- `utils/sce_recovery_policy.py`
- `scripts/car_model/meshsplatopt_run_sce_policy_recovery.py`
- `scripts/car_model/smoke_test_stageSCE7_sce_policy.py`

New policy fields:

- `require_sentinel_gate_for_recovery`
- `require_measured_candidate_for_recovery`
- `max_psnr_drop`
- `max_ssim_drop`
- `max_lpips_increase`
- `min_render_score_delta`
- `require_parent_pareto_for_acceptance`

New decision behavior:

- missing required sentinel gate -> `accept_parent_noop`
- measured render guard failure -> `accept_parent_noop`
- measured full-metric parent-Pareto failure -> `accept_parent_noop`
- sentinel degradation with render pass -> `run_targeted_rollback`
- sentinel non-degradation with render pass -> `continue_or_accept_visual_recovery`

## Bonsai Guard Check

Artifact:

`outputs/carnet/meshsplatopt/final_stageSCE19_policy_guard/bonsai_render_guard_v1/policy_contract/sce_policy_decision.json`

Input comparison:

- parent: F82 bonsai seed0 iteration 26000
- candidate: SCE8 bonsai fixed-policy probe iteration 26200
- sentinel gate for this dry-run was set to passing to isolate the render guard

Decision:

- action: `accept_parent_noop`
- execute recovery: `false`
- reason: `render_guard_failed`

Measured deltas:

- PSNR: `-0.176259`
- SSIM: `-0.044556`
- LPIPS: `+0.030025`
- render score: `-0.250839`

This means the hardened policy would not accept or continue the bonsai SCE8 v1 candidate, even though sparse depth improved. The failure becomes a controlled no-op rather than a damaging recovery.

## Verification

Commands:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE7_sce_policy.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q
```

Both passed.

## Remaining Work

This is a policy reliability fix, not a new quality win. It prevents known negative transfer on bonsai, but it does not yet solve the remaining courtyard held-out Depth MAE gap or create a universal SCE improvement over F82. The next load-bearing step is a guarded multiscene SCE v2 validation where each scene either improves under strict metrics or safely no-ops to F82.
