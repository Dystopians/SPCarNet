# v283-v284 View-Feature Ridge Texture v169 Gate Log

Date: 2026-06-30

Prompt authority: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Gate

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

The candidate can be promoted only if flowers exact beats all three axes. Full9 remains blocked before that.

## Method Change

`view_feature_ridge_texture` is a real representation-level change in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.

Compared with v282 low-rank baked texture, this mode removes the PCA coefficient bottleneck. For each target surface pixel it gathers train-fit Phase-J teacher residual slots from the same face/UV neighborhood and fits a tiny weighted ridge RGB decoder. The source features include:

- source view direction;
- source normal;
- source parent RGB;
- parent edge;
- residual-edge support;
- teacher-better support;
- relative UV-bin offset;
- source gain and support count;
- source-target view/normal/color/edge compatibility.

The target feature vector uses only target no-GT render evidence plus train-fit source statistics. Target/test RGB is not read during apply.

v284 adds `--view_feature_ridge_self_error_beta` and `--view_feature_ridge_self_error_floor`. This computes source residual self-reconstruction error from the fitted ridge model and shrinks unreliable row blends without using target/test GT.

An audit-only switch `--drop_checkpoint_policy_fields` was also added. It removes inherited `policy_reliability`, `policy_gain`, and `policy_tail_risk` from loaded source-bank checkpoints so we can measure how much the new decoder depends on prior policy calibration.

## Commands And Artifacts

All runs used GPU `1`, `WANDB_MODE=offline`, and wrote outputs under `/data/peilincai/mesh-splatting/outputs/carnet`. `/dev/shm` was read-only for existing low-copy flowers evidence because it had only about `1.7G` free during preflight.

Shared evidence:

- fit: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- target no-GT apply: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- target eval after apply: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`
- loaded bank: `outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_bank.npz`

Key output dirs:

- v283 policy: `outputs/carnet/v283_view_feature_ridge_texture_policy_loadedbank_20260630`
- v283 exact: `outputs/carnet/v283b_view_feature_ridge_texture_flowers_exact_alpha1_20260630`
- v284 policy: `outputs/carnet/v284_view_feature_ridge_selferr_policy_loadedbank_20260630`
- v284 exact: `outputs/carnet/v284b_view_feature_ridge_selferr_flowers_exact_alpha1_20260630`
- v284 policy-prior ablation: `outputs/carnet/v284c_view_feature_ridge_selferr_drop_policy_policyval_20260630`

Machine-readable summary: `docs/car_model/results/v283_v284_view_feature_ridge_texture_summary.json`.

## Results

| run | stage | policy-val / target candidate | gains vs parent | positive fractions | Phase-J PSNR gap | verdict |
|---|---|---:|---:|---:|---:|---|
| v283 | policy-val | 20.664684 / 0.719834 / 0.152349 | +0.058247 / +0.002308 / +0.000967 | 1.000 / 1.000 / 1.000 | n/a | promote to exact |
| v283b | target exact | 19.842806 / 0.620127 / 0.180020 | +0.010752 / +0.000217 / +0.000315 | 0.864 / 0.682 / 0.682 | -0.461552 | fail Phase-J |
| v284 | policy-val | 20.664695 / 0.719835 / 0.152348 | +0.058258 / +0.002309 / +0.000968 | 1.000 / 1.000 / 1.000 | n/a | promote to exact |
| v284b | target exact | 19.842785 / 0.620127 / 0.180019 | +0.010731 / +0.000216 / +0.000316 | 0.864 / 0.682 / 0.727 | -0.461573 | fail Phase-J |
| v284c | policy-val without checkpoint policy fields | 20.615103 / 0.713699 / 0.155407 | +0.008666 / -0.003828 / -0.002090 | 0.667 / 0.000 / 0.333 | n/a | fail policy-val |

Tail risk on target exact remains unresolved:

- v283b target PSNR/SSIM/LPIPS CVaR gains: `-0.002749 / -0.000305 / -0.000421`
- v284b target PSNR/SSIM/LPIPS CVaR gains: `-0.002786 / -0.000306 / -0.000421`

v284 self-error stats:

- policy mean texture self-confidence: `0.912442`
- policy mean self-error ratio: `0.203886`
- target mean texture self-confidence: `0.910285`
- target mean self-error ratio: `0.210121`

## Comparison To v282

Best v282 target exact remains v282b fixed alpha 0.50:

- candidate: `19.850666 / 0.619745 / 0.180620`
- gains: `+0.018612 / -0.000165 / -0.000286`
- Phase-J PSNR gap: `-0.453692`

v283/v284 improve SSIM/LPIPS over v282 but lose PSNR:

- v283b vs v282b fixed alpha 0.50: `-0.007860 PSNR / +0.000382 SSIM / +0.000600 LPIPS`
- v284b vs v282b fixed alpha 0.50: `-0.007881 PSNR / +0.000382 SSIM / +0.000601 LPIPS`

This means the new direct ridge carrier is not a breakthrough. It changes the quality tradeoff toward perceptual metrics but does not close the Phase-J PSNR gap.

## No-GT Audit

v283b and v284b target exact both passed stripped target no-GT audit. Forbidden keys checked:

`rgb_gt`, `residual_rgb`, `residual_l1`, `teacher_residual_rgb`, `teacher_residual_l1`, `teacher_residual_rgb_raw`, `teacher_parent_delta_l1`, `teacher_gain_l1`, `teacher_better_mask`.

Target/test GT was loaded only after apply for metrics.

## Candid Verdict

Status: `NOT COMPLETE`.

The v283-v284 route answered a useful question: the v282 bottleneck was not only the PCA low-rank coefficient bottleneck. A more expressive per-pixel local ridge decoder gives strong policy-val gains, but it still fails target exact and remains about `0.462 dB` below Phase-J flowers PSNR.

The v284c ablation is important: dropping checkpoint policy fields makes policy-val fail all-axis. The current stable gain depends on inherited policy reliability/gain priors from the loaded bank. This is allowed under the current train/policy protocol, but it weakens the claim that `view_feature_ridge_texture` alone is a robust final method.

Do not run full9 from v283/v284. The next useful route should be a stronger learned view-dependent surface decoder with explicit source-heldout calibration, or patch/perceptual teacher objectives that can carry more correct RGB energy while preserving target tails. Further alpha/threshold scans are not recommended.
