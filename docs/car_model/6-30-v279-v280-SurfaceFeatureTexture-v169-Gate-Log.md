# v279-v280 Calibration Gate and Surface Feature Texture Log

Date: 2026-06-30

Prompt: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Status: `NOT COMPLETE`. No full9 was launched because flowers exact still does not beat Phase-J all-axis.

## Gate

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

Candidate passes only if PSNR is higher, SSIM is higher, and LPIPS is lower.

## Storage / Runtime Preflight

- `/data`: `116G` available at launch.
- `/dev/shm`: `1.7G` available; no new evidence cache was copied.
- `/tmp`: `6.0T` available.
- v279a ran on GPU1 with W&B offline.
- v280a ran on GPU1 with W&B offline; late target exact views slowed because another user later occupied GPU1.
- v280b/v280c fixed-alpha exact reruns were interrupted because shared GPUs made single policy/exact views take minutes; they are not quality results.

## Method Changes

### v279 Calibration Face Reliability

Implemented in `scripts/car_model/train_perceptual_surface_residual_decoder.py`:

- disjoint fit / calibration / policy-val split;
- per-face calibration score from calibration-view local L1 gain plus structure gain;
- face-reliability threshold grid in policy-val;
- fixed threshold applied to target no-GT apply;
- W&B/audit fields for selected face-reliability threshold.

This is useful as a diagnostic but is not treated as the main representation innovation because the 6-28 prompt explicitly warns against `face reliability gate only` loops.

### v280 Surface Feature Texture v1

Implemented in `scripts/car_model/train_perceptual_surface_residual_decoder.py`:

- train-fit-only surface texture over `face x UV-bin`;
- 18 feature channels per bin: support, teacher residual mean/energy/variance, teacher gain, parent RGB, alpha, normal/view statistics;
- uncovered UV bins fall back to train-fit face-level means while keeping support flags;
- `_load_feature_rows` appends texture features during train, policy-val, and target no-GT apply;
- checkpoint saves `surface_feature_texture`, and `surface_feature_texture_v1.npz` is emitted for audit;
- target exact now computes adapted output from stripped no-GT evidence before loading eval GT for metrics.

This is the real representation-level attempt in this round: it bakes teacher residual statistics into a MeshSplatting-compatible surface-attached feature texture rather than adding another scalar gate.

## Commands

v279a reused the v277a checkpoint and enabled calibration reliability:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_eval_mode always --init_checkpoint outputs/carnet/spcarnet_v277_gain_soft_confidence_20260630/v277a_mid_gainsoft_structure_gate_targetexact/v180_perceptual_surface_decoder.pt \
  --skip_training --enable_calibration_face_reliability --calibration_stride 4 \
  --face_reliability_threshold_grid=-0.0005,0,0.0005 --alpha_grid 0.5 \
  --apply_confidence_threshold_grid 0,0.7 --apply_gate_strength_grid 0,1 \
  --compute_lpips --enable_wandb \
  --output_dir outputs/carnet/spcarnet_v279_calibration_face_reliability_20260630/v279a_reuse_v277a_calib_face_reliability_targetexact
```

v280a trained the new surface feature texture decoder from scratch:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_eval_mode always --policy_val_stride 4 \
  --surface_texture_mode v1 --surface_texture_uv_bins 4 --surface_texture_max_samples_per_view 120000 \
  --max_candidate_faces 65536 --steps 300 --batch_size 65536 \
  --embedding_dim 24 --hidden_dim 192 --layers 4 --confidence_head \
  --confidence_target_mode gain_soft --feature_mode fourier_v1 \
  --alpha_grid 0.25,0.5,0.75 --apply_confidence_threshold_grid 0,0.7 \
  --apply_gate_mode parent_luma_gradient --apply_gate_strength_grid 0,1 \
  --compute_lpips --enable_wandb \
  --output_dir outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_surface_texture_v1_mid_targetexact
```

## Results

| run | change | policy-val gains PSNR / SSIM / LPIPS | target gains PSNR / SSIM / LPIPS | target candidate | changed | verdict |
|---|---|---:|---:|---:|---:|---|
| v277a ref | learned decoder + gain confidence | +0.013926 / +0.000037 / +0.000090 | +0.010690 / -0.001008 / -0.000456 | 19.842744 / 0.618902 / 0.180791 | 0.139102 | fail target SSIM/LPIPS |
| v279a | calibration face reliability | +0.002628 / +0.000018 / +0.000135 | +0.000292 / -0.000277 / -0.000150 | 19.832346 / 0.619633 / 0.180485 | 0.009105 | conservative no-breakthrough |
| v280a | surface feature texture v1 | +0.031057 / +0.000530 / +0.000980 | +0.008885 / -0.001514 / -0.000743 | 19.840939 / 0.618396 / 0.181078 | 0.121150 | stronger policy-val, target fail |

v280a surface texture coverage:

- candidate faces: `65536`
- UV bins: `1048576`
- covered faces: `65430` (`0.998383`)
- covered bins: `426286` (`0.406538`)
- train-fit samples used: `3788244`

No-target-GT audit passed for v279a and v280a: `22` checked views, no forbidden target keys.

## Alpha Rescale Diagnostic

Using v280a saved alpha-0.75 target PNGs, I rescaled `candidate-parent` offline to estimate whether a more conservative fixed alpha could recover target SSIM/LPIPS. This is a diagnostic, not a replacement for exact model reapply.

| alpha | PSNR gain | SSIM gain | LPIPS gain | all-axis |
|---:|---:|---:|---:|---|
| 0.025 | +0.001384 | +0.000035 | -0.000007 | false |
| 0.050 | +0.002694 | +0.000062 | -0.000023 | false |
| 0.075 | +0.003928 | +0.000082 | -0.000047 | false |
| 0.100 | +0.005088 | +0.000094 | -0.000079 | false |
| 0.125 | +0.006173 | +0.000099 | -0.000120 | false |
| 0.150 | +0.007182 | +0.000098 | -0.000165 | false |
| 0.200 | +0.008977 | +0.000074 | -0.000253 | false |
| 0.250 | +0.010470 | +0.000025 | -0.000334 | false |
| 0.300 | +0.011663 | -0.000047 | -0.000405 | false |

Interpretation: lowering alpha can make SSIM non-negative, but LPIPS stays negative across the tested range. The failure is not only over-strong deployment; the target-view residual direction still hurts perceptual similarity.

## Artifacts

- v279a JSON: `outputs/carnet/spcarnet_v279_calibration_face_reliability_20260630/v279a_reuse_v277a_calib_face_reliability_targetexact/v180_perceptual_surface_decoder_audit.json`
- v279a renders: `outputs/carnet/spcarnet_v279_calibration_face_reliability_20260630/v279a_reuse_v277a_calib_face_reliability_targetexact/target_exact_fixed_policy`
- v280a JSON: `outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_surface_texture_v1_mid_targetexact/v180_perceptual_surface_decoder_audit.json`
- v280a surface texture: `outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_surface_texture_v1_mid_targetexact/surface_feature_texture_v1.npz`
- v280a renders: `outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_surface_texture_v1_mid_targetexact/target_exact_fixed_policy`
- alpha diagnostic JSON: `outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_alpha_rescale_diagnostic/alpha_rescale_sweep.json`
- W&B offline:
  - `outputs/carnet/spcarnet_v279_calibration_face_reliability_20260630/v279a_reuse_v277a_calib_face_reliability_targetexact/wandb/offline-run-20260630_070416-8yyhbdj9`
  - `outputs/carnet/spcarnet_v280_surface_feature_texture_20260630/v280a_surface_texture_v1_mid_targetexact/wandb/offline-run-20260630_072354-w70xwtxv`

## Verdict

v280 is a real representation upgrade and improves policy-val much more than v279/v277, but target exact still fails the 6-28 prompt's hard Phase-J flowers gate. The current bottleneck is now sharper:

1. The surface feature texture can carry more train/policy-val teacher signal.
2. The learned residual direction still does not transfer perceptually to held-out target views.
3. Conservative alpha cannot recover LPIPS, so the next step should change the supervision/representation, not alpha.

Recommended next direction: train the residual carrier against an explicit perceptual/patch teacher target or learn a target-free uncertainty model that predicts residual direction validity from source-view disagreement, not only per-face/UV train-fit statistics.
