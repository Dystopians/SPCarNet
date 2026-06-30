# v285-v286 Source-Heldout Calibration v169 Gate Log

Date: 2026-06-30

Prompt authority: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Gate

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

Full9 remains blocked until flowers exact beats all three axes.

## Motivation

v283/v284 showed that removing the low-rank PCA bottleneck was not enough. The direct `view_feature_ridge_texture` decoder passed policy-val, but target exact still had negative tails and remained about `0.462 dB` below Phase-J PSNR.

The next suspected weakness was calibration: v284 self-error used the same source samples that fitted the ridge model, so it could not reliably detect source-to-target extrapolation risk.

## Method Change

v285 adds target-free source-heldout residual-direction calibration to `view_feature_ridge_texture`.

New CLI:

- `--view_feature_ridge_holdout_beta`
- `--view_feature_ridge_holdout_floor`
- `--view_feature_ridge_holdout_min_sources`

For every target row:

1. collect same-face/UV-neighborhood train-fit source residual slots;
2. split slots by source-view parity, falling back to slot parity when `source_view_id` is unavailable;
3. fit a second ridge decoder on one split;
4. predict heldout source residuals on the other split;
5. compute heldout residual error ratio and residual-direction cosine;
6. shrink target residual blend by a confidence derived from heldout error and direction.

This is target-free: it uses only train-fit source slots and target no-GT render features during apply.

v286 then addresses the v284c weakness by dropping inherited checkpoint policy fields and recalibrating policy reliability/gain for the current decoder:

- `--drop_checkpoint_policy_fields`
- `--policy_reliability_mode patch_perceptual_v1`
- `--policy_reliability_alpha 1.0`
- `--policy_gain_mode positive_soft`

## Artifacts

Shared evidence:

- fit: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT apply: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval after apply: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`
- loaded bank: `outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz`

Run dirs:

- `outputs/carnet/v285_holdout_smoke_20260630`
- `outputs/carnet/v285a_holdout_policy_loadedbank_20260630`
- `outputs/carnet/v285b_holdout_flowers_exact_alpha1_20260630`
- `outputs/carnet/v286a_holdout_recalibrated_policyval_20260630`
- `outputs/carnet/v286b_holdout_recalibrated_flowers_exact_alpha1_20260630`

Machine-readable summary: `docs/car_model/results/v285_v286_holdout_calibration_summary.json`.

All medium/exact runs used GPU `1` and `WANDB_MODE=offline`.

## Results

| run | stage | candidate | gains vs parent | positive fractions | Phase-J PSNR gap | verdict |
|---|---|---:|---:|---:|---:|---|
| v285 smoke | policy-val | 21.161609 / 0.743768 / n/a | -0.000064 / -0.000150 / n/a | 0.667 / 0.667 / n/a | n/a | interface pass, quality fail |
| v285a | policy-val | 20.664685 / 0.719837 / 0.152346 | +0.058248 / +0.002310 / +0.000970 | 1.000 / 1.000 / 1.000 | n/a | promote exact |
| v285b | target exact | 19.842752 / 0.620126 / 0.180018 | +0.010698 / +0.000215 / +0.000317 | 0.864 / 0.682 / 0.727 | -0.461606 | fail |
| v286a | recalibrated policy-val | 20.650730 / 0.719454 / 0.152572 | +0.044293 / +0.001928 / +0.000745 | 1.000 / 1.000 / 1.000 | n/a | promote exact |
| v286b | recalibrated target exact | 19.840910 / 0.620183 / 0.180100 | +0.008856 / +0.000272 / +0.000235 | 0.864 / 0.682 / 0.773 | -0.463448 | fail |

Target tail comparison:

| run | PSNR tail CVaR | SSIM tail CVaR | LPIPS tail CVaR |
|---|---:|---:|---:|
| v284b | -0.002786 | -0.000306 | -0.000421 |
| v285b | -0.002830 | -0.000307 | -0.000419 |
| v286b | -0.000542 | -0.000206 | -0.000244 |

Heldout stats:

- v285a policy mean holdout confidence: `0.847217`
- v285a policy mean holdout cosine: `0.225993`
- v285a policy p10 holdout cosine: `-0.704934`
- v285b target mean holdout confidence: `0.845896`
- v285b target mean holdout error ratio: `2.078181`
- v286b target mean holdout confidence: `0.845896`

## Interpretation

v285 proves the heldout calibration machinery is real and target-free. It detects unstable residual directions: the p10 heldout cosine is strongly negative on policy-val. However, simple shrinkage does not improve the target exact frontier. It slightly improves LPIPS but loses PSNR and does not fix target tails.

v286 is more important scientifically. It removes inherited v265 policy fields and recalibrates policy reliability/gain for the current decoder. This fixes the v284c story weakness: the method can pass policy-val without relying on old checkpoint policy priors. But the price is lower residual injection. v286 improves target tail risk substantially, especially PSNR tail CVaR, but it further reduces mean PSNR and remains farther from Phase-J.

## Verdict

Status: `NOT COMPLETE`.

Source-heldout calibration and current-decoder policy recalibration are valid engineering/research additions, but they do not solve the core paper blocker. Conservative calibration can make target tails safer; it cannot create the missing high-fidelity RGB correction energy needed to beat Phase-J.

Do not run full9 from v285/v286. The next route should move beyond per-row local ridge and train a global or patch-aware view-dependent surface decoder with a stronger residual target, likely involving patch/perceptual teacher supervision and explicit source-heldout validation.
