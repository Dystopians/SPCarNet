# Stage SCE21 Tail-Risk Sentinel Report

Date: 2026-05-06

Decision: `SCE21_COURTYARD_ALL_METRIC_PASS_VS_F82`

## Summary

SCE21 is the first courtyard candidate in this line that beats F82 on all six tracked independent metrics under unchanged topology.

The mechanism is Conditional Tail-Risk Sentinel Envelope (CTR-SCE): one-sided parent rollback optimized with cluster-CVaR over the worst sparse-depth certificate violations, plus a local 1px pixel envelope.

## Runs

First tail-risk run:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/courtyard/cluster_cvar_patch1_28600to28780_seed0/recovery_model`

- W&B: `uhbivqf7`
- start: SCE7 best iteration 28600
- final: 28780
- topology unchanged: `true`
- regressed-only train sentinel cache
- rollback aggregation: `cluster_cvar`
- CVaR fraction: `0.2`
- pixel radius: `1`
- patch reduce: `max_violation`

Second all-sentinel continuation:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/courtyard/all_sentinel_cvar_patch1_28780to28880_seed0/recovery_model`

- W&B: `i4eewtbz`
- start: SCE21 iteration 28780
- final: 28880
- topology unchanged: `true`
- all train sentinels, not only regressed sentinels
- rollback aggregation: `cluster_cvar`
- CVaR fraction: `0.1`
- pixel radius: `1`
- patch reduce: `max_violation`

## Main Result: max500 Geometry Gate

| method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| F82 26000 | 12.198611 | 0.308649 | 0.566687 | 0.301884 | 3.339872 | 40.215702 |
| SCE7 28600 | 12.610288 | 0.338174 | 0.560068 | 0.298901 | 3.341660 | 39.368305 |
| SCE21 28780 | 12.612520 | 0.338573 | 0.559891 | 0.298388 | 3.337240 | 39.329123 |
| SCE21 28880 | 12.616089 | 0.338898 | 0.559881 | 0.298215 | 3.336610 | 39.339078 |

SCE21 28880 minus F82:

- PSNR: `+0.417478`
- SSIM: `+0.030249`
- LPIPS: `-0.006806`
- AbsRel: `-0.003668`
- Depth MAE: `-0.003262`
- Normal: `-0.876624`

Collector artifact:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/courtyard/f82_vs_sce21_28880_table/stageSCE8_multiscene_policy_report.md`

## Robustness Check: max1000 Geometry

Additional independent geometry evaluation with `--max_points_per_view 1000`:

| method | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|
| F82 26000 max1000 | 0.306570 | 3.353679 | 39.744123 |
| SCE21 28880 max1000 | 0.295966 | 3.280159 | 38.343424 |

SCE21 remains better under a denser sparse-geometry gate:

- AbsRel: `-0.010604`
- Depth MAE: `-0.073519`
- Normal: `-1.400699`

## Diagnostic-Only Test Correspondence Analyzer

Diagnostic output:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/courtyard/test_regression_f82_vs_sce21_28880`

This test-split analyzer is not used for training or policy selection.

It shows:

- AbsRel improves: `-0.004612`
- sampled correspondence MAE remains slightly worse: `+0.017089`
- gate-critical count decreases from the earlier SCE7-style failure but does not vanish
- `DSC_0318` remains a local diagnostic weak view

Interpretation: SCE21 solves the aggregate independent geometry gate and the max1000 robustness check, but it does not prove every sparse correspondence or every held-out view is locally non-regressing.

## First Multiscene Probe

Bonsai fixed-policy probe:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/bonsai/all_sentinel_cvar_patch1_26000to26200_seed0/recovery_model`

- W&B: `5eg8309n`
- source settings from F82 contract: `images_4`, resolution `4`
- train-only parent sentinel cache: `16136` sentinels, `32` train views, `no_test_leakage=true`
- topology unchanged: `true`

Result vs F82 bonsai:

- PSNR: `+0.001048`
- SSIM: `-0.000914`
- LPIPS: `+0.000470`
- AbsRel: `-0.000101`
- Depth MAE: `-0.001859`
- Normal: `+0.000579`

This is not an all-metric pass. It is a much safer result than the earlier SCE8 bonsai v1 negative transfer, but CTR-SCE is still not a universal F82 replacement.

Current two-scene collector:

`outputs/carnet/meshsplatopt/final_stageSCE21_tail_risk_sentinel/current_courtyard_bonsai_table/stageSCE8_multiscene_policy_report.md`

- rows: `2`
- all-pass rows: `1`

## Verification

Commands passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE21_tail_risk_rollback.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE7_sce_policy.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q
```

## Decision

`SCE21_COURTYARD_ALL_METRIC_PASS_VS_F82`

This is a real milestone over SCE7/SCE20: the remaining courtyard Depth MAE gap is closed while preserving RGB, perceptual, AbsRel, normal, and unchanged topology.

The claim should still be scoped: SCE21 currently proves a courtyard breakthrough and a general mechanism implementation. A first fair bonsai probe is mixed, so SCE21 does not yet prove multiscene universal superiority over F82.
