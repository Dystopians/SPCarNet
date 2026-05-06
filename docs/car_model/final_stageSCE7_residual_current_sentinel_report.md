# Final Stage SCE7 Residual Current Sentinel Report

Date: 2026-05-06

Decision: `RESIDUAL_CURRENT_SENTINEL_DID_NOT_BEAT_28600_KNEE`

## Goal

Test whether rebuilding train-only sentinels from the current best SCE7 candidate, rather than the older F95 candidate, can close the last courtyard Depth MAE gap.

## Train Evidence

Analyzer:

`outputs/carnet/meshsplatopt/final_stageSCE7_residual_current_sentinel/courtyard/train_regression_best28600_vs_f82`

Current best 28600 vs F82 on train split:

- AbsRel: `0.389850 -> 0.385679`
- Depth MAE: `4.864911 -> 4.844891`
- regressed candidate sentinels: `13274`
- gate-critical sentinels: `1906`

Cache:

`outputs/carnet/meshsplatopt/final_stageSCE7_residual_current_sentinel/courtyard/sentinel_cache_current_residual_dense1500/sentinel_cache.npz`

- sentinels: `42245`
- current-regressed sentinels: `14945`
- views: `33`
- no test leakage: `true`

## Recovery Result

Run:

`outputs/carnet/meshsplatopt/final_stageSCE7_residual_current_sentinel/courtyard/currentres_beta0p02_28600to28650_seed0/recovery_model`

W&B: `hfkzouma`

Configuration:

- start: current best `28600`
- final: `28650`
- rollback cache: current residual dense1500
- rollback: `combined`, `beta=0.02`, `lambda=0.2`, `regressed_only=true`
- vertex LR: `0.002`

Result:

- PSNR `12.608293`
- SSIM `0.338111`
- LPIPS `0.559888`
- AbsRel `0.299108`
- Depth MAE `3.343615`
- Normal `39.370048`

This is worse than the 28600 knee on Depth MAE (`3.341660`) and does not beat F82 (`3.339872`).

## ECG/Planner Diagnosis

Train ECG:

`outputs/carnet/meshsplatopt/final_stageSCE12_evidence_conflict_graph/courtyard/best28600_train_policy`

Planner:

`outputs/carnet/meshsplatopt/final_stageSCE13_certificate_edit_planner/courtyard/best28600_train_policy_plan`

The train ECG top conflict is `cluster 876`, and the planner emits only `ROLLBACK_ONLY` actions. There is no train-certified evidence for snap/split/fill/delete. This means local topology surgery should not be promoted from this evidence.

## Conclusion

Current-residual sentinels are useful diagnostically but do not close the last held-out `DSC_0318` MAE gap through additional short recovery. The best accepted candidate remains SCE7 `28600`.

