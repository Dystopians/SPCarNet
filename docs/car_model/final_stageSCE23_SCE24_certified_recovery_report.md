# Stage SCE23/SCE24 Certified Recovery Report

Date: 2026-05-06

Decision: `SCE24_CERTIFIED_POLICY_REJECTS_BONSAI_RECOVERY`

## Summary

The new innovation path is a dual-certificate recovery policy:

1. CTR-SCE geometry certificate: train-only sparse-depth parent rollback with cluster-CVaR over the worst sentinel violations.
2. ATR appearance certificate: one-sided parent render rollback that penalizes only pixels where the current render is worse than the parent against GT.
3. Certified render selection: a train-only Pareto guard accepts a recovered checkpoint only if PSNR/SSIM/LPIPS do not regress against the parent.

This is a stronger and more honest method than parameter scanning. It does not force every scene to accept a risky update. On bonsai, the certified policy rejects recovery and selects the parent; on courtyard, SCE21 remains the real all-metric improvement row.

## Literature Basis

- CVaR tail-risk optimization: Rockafellar and Uryasev, "Optimization of Conditional Value-at-Risk", DOI `10.21314/jor.2000.038`.
- Conformal risk control motivates separating model improvement from a calibrated accept/reject risk certificate: `https://arxiv.org/abs/2208.02814`.
- 3D Gaussian Splatting motivates fast differentiable radiance-field optimization from sparse camera-calibration points: `https://arxiv.org/abs/2308.04079`.
- COLMAP/SfM sparse correspondences are used as geometry certificates, following the sparse-depth supervision idea also used by DS-NeRF: `https://arxiv.org/abs/2107.02791`.

## Implemented Interfaces

Training:

- `--enable_parent_render_rollback_loss`
- `--parent_render_rollback_dir`
- `--lambda_parent_render_rollback`
- warmup/decay controls
- absolute/relative margin controls
- `mean` or `cvar` aggregation
- CVaR fraction and minimum tail pixels
- optional patch radius and patch reduction
- `l1`, `l2`, or `channel_max` residual space

Runners:

- `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`
- `scripts/car_model/meshsplatopt_run_sce_policy_recovery.py`

New certification tools:

- `scripts/car_model/evaluate_render_split_metrics.py`
- `scripts/car_model/select_certified_recovery.py`
- `utils/certified_model_selection.py`

Smoke:

- `scripts/car_model/smoke_test_stageSCE23_parent_render_tail_rollback.py`

## Bonsai Experiments

Parent:

`outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/bonsai/adaptive_global_policy_v5_seed0/recovery_model`, iteration `26000`.

### Test RGB Metrics

| run | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| F82 parent 26000 | 11.069180 | 0.241154 | 0.572932 | 0 | 0 | 0 |
| SCE22 teacher 26200 | 11.070197 | 0.240247 | 0.573415 | +0.001017 | -0.000907 | +0.000483 |
| SCE23 ATR cvar 26800 | 11.073215 | 0.239113 | 0.574592 | +0.004035 | -0.002041 | +0.001659 |
| SCE24 ATR mean LPIPS 26200 | 11.070174 | 0.240256 | 0.573038 | +0.000994 | -0.000898 | +0.000105 |

### Test Geometry Metrics

| run | AbsRel | Depth MAE | Normal deg | dAbsRel | dMAE | dNormal |
|---|---:|---:|---:|---:|---:|---:|
| F82 parent 26000 | 0.181340 | 1.805841 | 42.147364 | 0 | 0 | 0 |
| SCE22 teacher 26200 | 0.181224 | 1.804218 | 42.151583 | -0.000116 | -0.001624 | +0.004218 |
| SCE23 ATR cvar 26800 | 0.181389 | 1.804270 | 42.166360 | +0.000049 | -0.001572 | +0.018996 |
| SCE24 ATR mean LPIPS 26200 | 0.181230 | 1.803978 | 42.148668 | -0.000110 | -0.001864 | +0.001304 |

## Train-Only Certified Selection

Train split metrics for SCE24:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| parent `ours_26000` | 11.508512 | 0.290483 | 0.549275 |
| candidate `ours_26200` | 11.511419 | 0.289623 | 0.549478 |

Guard decision:

`outputs/carnet/meshsplatopt/final_stageSCE24_balanced_appearance_risk/bonsai/atr_mean_lpips_26000to26200_seed0/certified_selection_train_guard.json`

The guard rejects the candidate:

- `delta_PSNR = +0.002907`
- `delta_SSIM = -0.000861`
- `delta_LPIPS = +0.000203`
- reasons: `ssim_regression`, `lpips_regression`
- selected: `parent`

This is the desired behavior. Bonsai should not be forced into a recovery update when train-only evidence already predicts appearance regression.

## Interpretation

SCE23/SCE24 did not produce a bonsai all-metric improvement. The important upgrade is methodological reliability:

- The old behavior kept trying to tune a weak positive PSNR signal while silently accepting SSIM/LPIPS regressions.
- The new behavior treats recovery as a certified decision: improve all protected objectives or revert to parent.
- This avoids test-set cherry-picking because the guard can be run on train/calibration renders before final test reporting.

## Current Paper Claim

Safe claim:

> SPCarNet introduces certified tail-risk recovery for mesh splatting: sparse-geometry CVaR certificates plus one-sided appearance rollback and train-only Pareto acceptance. It can produce strict all-metric improvements on courtyard and safely no-op on bonsai where recovery evidence is insufficient.

Unsafe claim:

> The method universally outperforms F82 on every scene.

That claim remains false under current bonsai evidence.

## Next Work

The next real improvement should not be another manual parameter sweep. It should add a stronger train-calibrated appearance certificate, likely patch-SSIM or feature-space parent rollback, because pixel residual and LPIPS-on-GT did not prevent SSIM regression on bonsai.
