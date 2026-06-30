# v275-v277 Learned Surface Decoder v169 Gate Log

Date: 2026-06-30

Authoritative prompt:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

Hard gate:

```text
Do not run full9 until flowers exact beats Phase-J all-axis:
PSNR > 20.304358, SSIM > 0.557770, LPIPS < 0.329222.
```

## Storage Preflight

The latest preflight before documenting this milestone was:

```text
/data:   116G available
/dev/shm: 1.6G available
/tmp:    6.0T available
root quota: 99138M of 100G
```

This is enough for flowers checkpoint/eval artifacts but not safe for duplicate
full9 evidence materialization. Full9 remains blocked by the metric gate anyway.

## Implemented Method Changes

Main implementation files:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

The v275-v277 route turns the old diagnostic MLP into a real train/eval
candidate:

1. Surface-attached residual decoder.
   - Per selected face embedding.
   - Fourier/barycentric/view/normal/parent-color features.
   - Tiny MLP predicts teacher-parent residual.
   - Target/test apply uses stripped no-GT evidence; GT is loaded only after
     apply for evaluation.

2. Target exact verifier.
   - `--target_evidence_dir` points to no-GT evidence.
   - `--target_eval_evidence_dir` is used only after apply.
   - The audit records forbidden-key checks and target exact metrics.

3. Structure-safe apply gate.
   - `--apply_gate_mode parent_luma_gradient`.
   - Uses only parent luma gradients and predicted residual gradients.
   - Shrinks residuals likely to create unsupported high-frequency changes.

4. Gain-soft learned confidence.
   - `--confidence_target_mode gain_soft`.
   - Builds confidence labels from train-fit `teacher_gain_l1`, not the mostly
     all-one `teacher_better_mask`.
   - `--sample_weight_confidence_power` downweights low-gain residual pixels.

5. Deploy-time confidence threshold.
   - `--apply_confidence_threshold_grid` is selected on policy-val only.
   - Target exact applies the fixed threshold without target GT.

## Key Commands

The full command line for each run is stored in its audit JSON under `command`.
Representative commands:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_eval_mode always \
  --confidence_head --confidence_target_mode gain_soft \
  --apply_gate_mode parent_luma_gradient \
  --compute_lpips --enable_wandb
```

Checkpoint reuse / threshold scan form:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --init_checkpoint outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277a_mid_gainsoft_structure_gate_targetexact/v180_perceptual_surface_decoder.pt \
  --skip_training \
  --alpha_grid 0.5 \
  --apply_confidence_threshold_grid 0.85 \
  --apply_gate_strength_grid 1 \
  --target_eval_mode always \
  --compute_lpips --enable_wandb
```

## Flowers Exact Results

All rows below use target no-GT apply and load target GT only after apply.

| run | method delta | selected policy | target PSNR gain | target SSIM gain | target LPIPS gain | changed | verdict |
|---|---|---|---:|---:|---:|---:|---|
| v275b | learned surface decoder, confidence head | alpha 0.25 | +0.009091 | -0.000808 | -0.000724 | 0.139362 | fail |
| v276a | parent-luma structure gate | alpha 0.75, gate 2.0 | +0.009069 | -0.001037 | -0.000304 | 0.139342 | fail |
| v277a | gain-soft confidence training | alpha 0.5, gate 1.0 | +0.010690 | -0.001008 | -0.000456 | 0.139102 | fail |
| v277c | confidence threshold scan | alpha 0.5, threshold 0.7, gate 1.0 | +0.009657 | -0.000896 | -0.000488 | 0.132016 | fail |
| v277d | conservative confidence threshold | alpha 0.5, threshold 0.85, gate 1.0 | +0.000945 | -0.000138 | -0.000284 | 0.004060 | fail |

Phase-J flowers reference:

```text
PSNR 20.304358 / SSIM 0.557770 / LPIPS 0.329222
```

Best target PSNR in this route is v277a:

```text
candidate PSNR 19.842744, gap to Phase-J PSNR = -0.461614
```

The conservative v277d policy proves the confidence threshold can strongly
reduce target damage:

```text
changed fraction: 0.004060
target SSIM gain: -0.000138
target LPIPS gain: -0.000284
```

However, it still does not become all-axis positive.

## Evidence Artifacts

Machine-readable summary:

```text
docs/car_model/results/v275_v277_learned_surface_decoder_summary.json
```

Primary audit JSON files:

```text
outputs/carnet/spcarnet_v275_learned_surface_feature_decoder_20260630/v275b_mid_confidence_head_fourier_targetexact/v180_perceptual_surface_decoder_audit.json
outputs/carnet/spcarnet_v276_structure_safe_decoder_20260630/v276a_reuse_v275b_gate_scan_targetexact/v180_perceptual_surface_decoder_audit.json
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277a_mid_gainsoft_structure_gate_targetexact/v180_perceptual_surface_decoder_audit.json
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277c_reuse_v277a_conf_threshold_small_targetexact/v180_perceptual_surface_decoder_audit.json
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277d_reuse_v277a_forced_conf085_targetexact/v180_perceptual_surface_decoder_audit.json
```

W&B offline roots:

```text
outputs/carnet/spcarnet_v276_structure_safe_decoder_20260630/v276a_reuse_v275b_gate_scan_targetexact/wandb/offline-run-20260630_043406-uwrpvh55
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277a_mid_gainsoft_structure_gate_targetexact/wandb/offline-run-20260630_050007-ycfceevh
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277c_reuse_v277a_conf_threshold_small_targetexact/wandb/offline-run-20260630_053906-wbayddh6
outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277d_reuse_v277a_forced_conf085_targetexact/wandb/offline-run-20260630_055426-97w4ctfq
```

## Interpretation

This is a real method upgrade, not only parameter tuning. The train/eval pipeline
now supports learned surface residual decoding, no-target-GT exact evaluation,
structure-aware application, gain-soft confidence learning, and fixed
confidence-threshold deployment.

The result is still negative under the v169 prompt. Policy-val all-axis success
does not transfer to target exact. The residual representation mostly improves
MSE/PSNR, but the correction direction still harms target SSIM and LPIPS. A very
conservative threshold can make the harm small, but then the visual/metric gain
nearly collapses.

## Current Bottleneck

The remaining weakness is not just alpha selection. The carrier can learn a
teacher residual that is locally useful on train-policy-val, but it cannot
certify that the same residual direction is structure/perceptually correct on
target views. This suggests the next useful route should change the teacher
target or representation again, for example:

1. Train a residual representation against a structure/perceptual teacher target,
   not raw teacher-parent RGB residual.
2. Add multi-view consistency or source-view agreement as a first-class learned
   input, not only face/local features.
3. Evaluate a true view-dependent low-rank basis with confidence calibrated on
   held-out views before target exact.

## Verdict

```text
Final status: NOT COMPLETE.
Full9 allowed: false.
Reason: flowers exact still fails target SSIM/LPIPS vs parent and fails Phase-J
PSNR by about 0.46 dB.
```
