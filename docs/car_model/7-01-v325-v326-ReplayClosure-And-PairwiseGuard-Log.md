# v325/v326 Replay Closure and Pairwise Guard Log

## Status

This is a reproducibility and safety milestone, not a paper-level improvement.
It fixes the immediate blocker that made v323/v324 comparisons unreliable:
current-code runs could fail to replay archived v322C because the effective
policy/evaluation profile was not fully captured by the representative command.

## What Changed

- Added `--policy_profile v322c_incumbent` to
  `scripts/car_model/apply_source_heldout_support_transport_calibrator.py`.
- The profile pins the archived v322C incumbent-preserving policy:
  candidate ladder, base-only KNN fallback, source reliability thresholds,
  calibrated raw-incumbent LCB, OOD guard, `evidence_max_side=256`, and
  `ssim_max_side=256`.
- The final v325b profile also pins the archived KNN fallback details:
  `per_view_knn_forbid_fixed_when_scene_nonfixed=True` and
  `per_view_knn_min_score_delta_vs_scene=0.0005`.
- Source reliability reports now always record the predictive/source gates that
  determine whether the policy was enabled, even when the policy is disabled.
- Added a hard pairwise safety gate: if pairwise dominance accepts no source
  leave-one-out views, it is disabled before target inference. This prevents
  target-time full-model overrides without source evidence.
- Added `scripts/car_model/audit_v322c_replay_consistency.py` for archived
  v322C replay audits.

## Evidence

Treehill exact replay:

- output: `outputs/carnet/spcarnet_v325_profile_replay_treehill_20260701`
- audit: `docs/car_model/results/v325_treehill_v322c_profile_replay_audit.json`
- PSNR gain delta vs archived v322C: `0.0`
- SSIM gain delta vs archived v322C: `0.0`
- per-view output mismatches: `0`
- source LOO choices match exactly.

Pairwise negative evidence before the guard:

- output: `outputs/carnet/spcarnet_v326_profile_pairwise_strict_treehill_20260701`
- audit: `docs/car_model/results/v326_pairwise_strict_treehill_vs_v322c_audit.json`
- PSNR gain delta vs archived v322C: `-0.002158558035077557`
- SSIM gain delta vs archived v322C: `-0.0000026060475243462698`
- root cause: source LOO accepted `0/11` pairwise candidates, but target full
  model still changed two views to `mix0250`.

Pairwise guard after the fix:

- output: `outputs/carnet/spcarnet_v326b_pairwise_zeroaccept_guard_treehill_20260701`
- audit: `docs/car_model/results/v326b_zeroaccept_guard_treehill_vs_v322c_audit.json`
- pairwise verdict: `pairwise dominance accepted no source views`
- PSNR gain delta vs archived v322C: `0.0`
- SSIM gain delta vs archived v322C: `0.0`
- per-view output mismatches: `0`

Full9 replay closure:

- initial full9 output:
  `outputs/carnet/spcarnet_v325_profile_replay_full9_20260701`
- KNN-fix reruns for the three mismatched scenes:
  `outputs/carnet/spcarnet_v325b_profile_replay_knnfix_delta3_20260701`
- merged report-only audit root:
  `outputs/carnet/spcarnet_v325b_profile_replay_full9_reportmerge_20260701`
- full9 audit:
  `docs/car_model/results/v325b_full9_v322c_profile_replay_audit.json`
- scenes: `9/9`
- missing archive scenes: `0`
- missing replay scenes: `0`
- PSNR gain delta vs archived v322C: `0.0`
- SSIM gain delta vs archived v322C: `0.0`
- candidate PSNR delta vs archived v322C: `0.0`
- candidate SSIM delta vs archived v322C: `0.0`

## Lesson

The current bottleneck is not candidate capacity. The strict oracle still shows
that some views can be improved without lowering SSIM, but target-GT-free
selection is not yet strong enough to capture that gap. The v326 guard makes
the method safer and restores fair incumbent replay, but it does not create a
new positive method result over v322C.

## Next Required Step

Use `--policy_profile v322c_incumbent` as the fixed incumbent baseline for any
v327+ selector changes. Do not compare a new selector against v322C unless this
profile remains unchanged and replay-audited.
