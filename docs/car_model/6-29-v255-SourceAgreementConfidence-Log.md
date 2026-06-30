# SPCarNet v255 Source Agreement Confidence Log

Date: 2026-06-29

v255 tests the next logical step after v253-v254: suppress source-bank residuals
whose top-k train-view residuals disagree with each other. This is target-blind
and uses the frozen v253b source bank, so it does not tune on target/test GT.

## Implementation

Updated:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New options:

```text
--source_agreement_mode {off,soft,hard}
--source_agreement_beta
--source_agreement_min_confidence
```

For each target pixel, the renderer computes the weighted top-k residual mean,
then computes source residual variance around that mean. In `soft` mode, high
variance lowers confidence and scales the predicted residual. This is meant to
reduce out-of-trajectory perceptual artifacts.

## Command

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline \
WANDB_DIR=/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/wandb \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_deferred_source_residual_renderer.py \
  --bank_checkpoint /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_bank.npz \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --source_agreement_mode soft \
  --source_agreement_beta 0.25 \
  --compute_lpips \
  --target_eval_mode auto \
  --enable_wandb \
  --output_dir /tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact
```

## Result

| stage | alpha | PSNR gain | SSIM gain | LPIPS gain | mean confidence | all-axis |
|---|---:|---:|---:|---:|---:|---|
| policy-val | 0.046875 | +0.001655 | +0.000018 | +0.000001 | 0.655315 | pass |
| target exact | 0.046875 | +0.001395 | +0.000036 | -0.000008 | 0.651719 | fail |

Artifacts:

```text
docs/car_model/results/v255_source_agreement_confidence_summary.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/v253_deferred_source_renderer_audit.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/target_exact_fixed_policy
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/wandb/offline-run-20260629_200707-e3wmgpr9
```

## Interpretation

Soft agreement confidence is not enough. It reduces mean residual confidence to
about `0.655`, but target LPIPS becomes more negative than v253b/v253d. The
method still improves PSNR and SSIM, which confirms that the correction is not
random, but it does not solve perceptual transfer.

This rules out a simple source residual variance gate as the main fix. The next
step should use a learned or calibrated perceptual reliability predictor, not a
single hand-designed agreement scalar.

Final status: **NOT COMPLETE for paper-level all-axis win**.
