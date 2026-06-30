# v246-v247 Source Evidence Bank and Projection Loss Log

Date: 2026-06-29

Authoritative prompt: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

## Verdict

This round is a real train/eval pipeline change, but it is not a quality success.

The new source-evidence-bank surface U-Net and teacher residual projection losses did not pass the v169 flowers gate. v247a slightly increases residual energy retention over v246a, but it worsens the teacher-image projection metrics and fails the policy-val all-axis tail guard. v247b, the teacher-only ablation, selects alpha `0.0` and therefore collapses to no-op under the safe policy.

Therefore this branch must not be promoted to full9.

## Hard Gate

The prompt requires flowers exact to beat Phase-J all-axis before any full9 promotion:

| reference | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J flowers gate | 20.304358 | 0.557770 | 0.329222 |

v246/v247 did not reach target apply/exact because the policy-val all-axis gate failed. This is the intended early stop behavior from the v169 prompt.

## What Changed In Code

Files touched for this round:

- `scripts/car_model/train_surface_conditioned_residual_unet.py`
- `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py`

Implementation details:

- Added source-bank evidence conditioning for `SurfaceTextureConditionedUNet`.
- The model can consume compact per-surface source evidence, top-k source views, a residual prior, and a view-direction gate.
- Added teacher residual projection losses:
  - cosine alignment between predicted residual and `teacher - parent`;
  - RMS/energy matching between predicted residual and `teacher - parent`;
  - active residual mask controlled by `--teacher_residual_projection_min_l1`.
- Added W&B metrics for the new losses and diagnostic values:
  - `train/teacher_residual_cosine_loss`
  - `train/teacher_residual_energy_loss`
  - `train/teacher_residual_cosine`
  - `train/teacher_residual_projection_active_fraction`
- Updated eval checkpoint loading so `surface_texture_unet` checkpoints with evidence stats can be applied without target/test GT.

This is not a pure parameter scan. It changes the representation input path and the training objective. The result is still negative.

## Runs

| run | role | W&B/offline log | result |
|---|---|---|---|
| v246a | source-bank, no residual prior, GT-assisted | `/tmp/peilincai_spcarnet_v246a_source_bank_flowers_20260629/v246a_source_bank_native1256/wandb/offline-run-20260629_173326-i9vpkf4q` | weak policy-val gain, failed strict all-axis |
| v246b | source-bank weak prior | none complete | failed during checkpoint save because quota was nearly full |
| v247a | source-bank + residual prior + projection losses, GT-assisted | `/tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_sourcebank_gtassist_native1256/wandb/offline-run-20260629_174751-fah6w8ta` | failed policy-val all-axis |
| v247b | teacher-only ablation + stronger projection losses | `/tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_sourcebank_teacheronly_native1256/wandb/offline-run-20260629_174751-uq328sla` | selected no-op |

Storage/quota snapshot after these runs:

- `quota -s`: `/dev/nvme0n1p4 98430M used of 100G`
- `/tmp/peilincai_spcarnet_v246a_source_bank_flowers_20260629`: `1.2G`
- `/tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629`: `1.3G`
- `/tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629`: `1.3G`

## Policy-Val Metrics

All rows compare candidate vs parent on held-out train-policy-val views. Positive LPIPS gain means LPIPS went down.

| run | selected alpha | PSNR gain | SSIM gain | LPIPS gain | min PSNR | min SSIM | min LPIPS | CVaR SSIM | CVaR LPIPS | changed fraction | all-axis row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v246a | 0.5 | +0.052707 | +0.002253 | +0.001792 | +0.016502 | +0.000603 | -0.000030 | +0.000933 | +0.000588 | 0.246993 | none |
| v247a | 0.5 | +0.037900 | +0.000707 | +0.001625 | +0.002806 | -0.000346 | -0.001074 | -0.000051 | -0.000045 | 0.273611 | none |
| v247b | 0.0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.000000 | none |

Interpretation:

- v246a is the least bad row, but its LPIPS min-view gain is negative and it did not satisfy the strict gate.
- v247a is worse under the actual tail criteria. The projection loss increases changed fraction and energy retention, but makes SSIM/LPIPS tails unsafe.
- v247b proves the teacher-only objective cannot currently produce a safe transferable correction; policy selection correctly falls back to alpha `0.0`.

## Residual Projection Audit

The audit compares the checkpoint residual against the Phase-J teacher residual on policy-val. If the carrier could really bake Phase-J, energy retention and cosine should be high and candidate vs teacher should improve.

| run | energy retention | cosine | changed fraction | sign agreement | PSNR gain vs teacher | SSIM gain vs teacher | LPIPS gain vs teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| v246a | 0.074658 | 0.124085 | 0.265788 | 0.544963 | -0.020431 | -0.001119 | +0.000257 |
| v247a | 0.085702 | 0.121128 | 0.294451 | 0.552884 | -0.054637 | -0.001954 | -0.000028 |
| v247b | 0.083270 | 0.115199 | 0.288069 | 0.537097 | -0.062839 | -0.002009 | -0.000039 |

The best energy retention is still only `8.57%`. The predicted residual direction is weakly aligned with the teacher (`cosine ~= 0.12`), and the projected candidate is worse than the parent relative to the teacher in PSNR/SSIM. This confirms the main bottleneck is still carrier/objective underfitting, not missing target evidence.

## No-GT Status

For v247a and v247b:

- target no-GT precheck passed;
- checked target evidence view count: `22`;
- `target_gt_visible_to_apply = false`;
- `target_residual_visible_to_apply = false`;
- target apply was skipped because policy-val all-axis failed.

So this negative result is not caused by target/test GT leakage.

## Commands And Artifact Paths

Primary machine-readable summary:

- `docs/car_model/results/v246_v247_sourcebank_projection_loss_summary.json`

Important reports:

- v246a report: `/tmp/peilincai_spcarnet_v246a_source_bank_flowers_20260629/v246a_source_bank_native1256/v246a_source_bank_noprior_gtassist_flowers_report.json`
- v246a projection audit: `/tmp/peilincai_spcarnet_v246a_source_bank_flowers_20260629/v246a_projection_audit.json`
- v247a report: `/tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_sourcebank_gtassist_native1256/v247a_projection_sourcebank_gtassist_flowers_report.json`
- v247a projection audit: `/tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_audit.json`
- v247b report: `/tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_sourcebank_teacheronly_native1256/v247b_projection_sourcebank_teacheronly_flowers_report.json`
- v247b projection audit: `/tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_audit.json`

Audit command used for v247a:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v247a_audit \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/audit_surface_checkpoint_residual_projection.py \
  --run_name v247a_projection_sourcebank_gtassist_flowers \
  --checkpoint /tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_sourcebank_gtassist_native1256/v247a_projection_sourcebank_gtassist_flowers.pt \
  --fit_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/teacher_surface_evidence_phasej_to_phasef_native1256 \
  --residual_rgb_key teacher_residual_rgb_raw \
  --residual_l1_key teacher_parent_delta_l1 \
  --alpha 0.5 --policy_val_stride 4 --max_policy_views 12 \
  --eval_tile 512 --eval_overlap 32 --ssim_max_side 512 --lpips_max_side 256 --compute_lpips --region_grid 4 \
  --output_json /tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_audit.json \
  --output_md /tmp/peilincai_spcarnet_v247a_projection_loss_flowers_20260629/v247a_projection_audit.md
```

Audit command used for v247b:

```bash
CUDA_VISIBLE_DEVICES=2 TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v247b_audit \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/audit_surface_checkpoint_residual_projection.py \
  --run_name v247b_projection_sourcebank_teacheronly_flowers \
  --checkpoint /tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_sourcebank_teacheronly_native1256/v247b_projection_sourcebank_teacheronly_flowers.pt \
  --fit_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/teacher_surface_evidence_phasej_to_phasef_native1256 \
  --residual_rgb_key teacher_residual_rgb_raw \
  --residual_l1_key teacher_parent_delta_l1 \
  --alpha 0.5 --policy_val_stride 4 --max_policy_views 12 \
  --eval_tile 512 --eval_overlap 32 --ssim_max_side 512 --lpips_max_side 256 --compute_lpips --region_grid 4 \
  --output_json /tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_audit.json \
  --output_md /tmp/peilincai_spcarnet_v247b_projection_loss_flowers_20260629/v247b_projection_audit.md
```

Training reports contain the exact serialized `args` blocks for reproducibility. The v247 launch profile used:

- `--model_type surface_texture_unet`
- `--base_channels 32`
- `--surface_texture_size 8`
- `--surface_feature_dim 16`
- `--enable_surface_source_evidence_bank`
- `--surface_source_evidence_top_k 4`
- `--surface_evidence_residual_prior_weight 0.1`
- `--surface_evidence_view_gate_power 1.0`
- `--teacher_residual_cosine_weight 0.12` and `--teacher_residual_energy_weight 1.0` for v247a
- `--teacher_residual_cosine_weight 0.18` and `--teacher_residual_energy_weight 1.2` for v247b
- W&B project `spcarnet_meshprior`, offline mode

## Lessons

1. Adding a source-bank and projection loss is not enough if the prediction is still a small image-space residual through a weak surface carrier.
2. Energy retention around `8%` is a hard warning. It is too low to justify exact/full9 promotion.
3. Projection loss can improve residual amplitude while damaging perceptual tails. The hard gate should remain tail-aware.
4. Teacher-only training is currently not viable for this carrier. It needs either stronger supervised structural anchors or a different representation.
5. The next attempt should not be another alpha, footprint, source-bank top-k, or scalar residual sweep.

## Next Recommendation

Stop the source-bank/projection-loss branch unless a materially different representation is introduced.

The next route should be one of:

- a stronger surface-attached feature texture with a decoder that predicts final selected-alpha correction directly and validates on cross-view held-out policy splits;
- a patch/gradient-aware deferred surface decoder with explicit SSIM/LPIPS tail objective and no posthoc alpha contract mismatch;
- a diagnostic proof that Phase-J-style render-time correction cannot be faithfully baked into the current MeshSplatting-compatible carrier.

Exact next command/prompt if continuing:

```text
Use docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md and docs/car_model/6-29-v246-v247-SourceEvidenceBank-ProjectionLoss-Log.md as constraints. Do not continue source-bank/top-k/alpha sweeps. Implement one materially stronger surface-attached decoder that directly optimizes held-out policy-val PSNR/SSIM/LPIPS tails on flowers, with target/test GT stripped, and run projection audit before any exact/full9 promotion.
```
