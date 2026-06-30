# 2026-06-30 v290-v292 PatchViewMoE + View-Support Gate Log

This log follows the hard protocol in:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

The key rule is still binding: **do not run full9 until flowers exact beats
Phase-J all-axis**:

| reference | PSNR must be > | SSIM must be > | LPIPS must be < |
|---|---:|---:|---:|
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 |

## Storage And GPU Preflight

Latest preflight before documenting this milestone:

```text
/data:   101G available
/dev/shm: 1.7G available
/tmp (/): 6.0T available
quota:   99304M used / 100G limit
```

Runs reused the low-copy v168 flowers evidence under:

```text
/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers
```

No full9 was launched because the flowers Phase-J gate did not pass.

GPU selection followed the low/mid-occupancy rule. The latest visible low
occupancy devices were GPU1, GPU2, GPU3, and GPU5.

## Real Method Change

The implementation was added to:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

This is not an alpha-only or footprint-only iteration. The v290-v292 line adds
four concrete method pieces:

1. `lowrank_view_v2` surface texture rows.
   Each surface bin stores low-rank teacher residual bases plus source camera
   mean, source camera concentration, and target-source camera cosine.

2. `patch_view_moe` decoder output.
   The decoder predicts low-rank coefficients plus a view/patch-conditioned
   direct expert residual mixture. This is a stronger carrier than a scalar RGB
   residual atlas.

3. Policy-val tail-safe certificate.
   Policy rows now record all-axis pass, positive-view fraction, min-view gain,
   CVaR/tail gains, and a `tail_safe_pass` flag. `target_eval_mode=auto` now
   requires tail safety instead of mean all-axis only.

4. `lowrank_view_cos` target view-support gate.
   At target apply time, the residual and confidence are attenuated when the
   target camera is poorly supported by the training source-view distribution.
   This directly addresses the v169 concern that out-of-trajectory target views
   can turn a policy-val residual into a target LPIPS failure.

## Core Commands

The main v290 policy training command was:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_eval_mode never --policy_val_stride 4 \
  --surface_texture_mode lowrank_view_v2 --surface_texture_uv_bins 4 \
  --decoder_output_mode patch_view_moe --moe_expert_count 3 --moe_direct_scale 0.35 \
  --residual_target_mode raw --image_loss_mode patch_edge_v1 \
  --alpha_grid 0.125,0.25,0.5,0.75 \
  --compute_lpips --enable_wandb \
  --wandb_project spcarnet-v290-patch-view-moe \
  --output_dir outputs/carnet/spcarnet_v290_patch_view_moe_20260630/v290a_patch_view_moe_policy_20260630
```

The selected v292 frontier used the trained v290 checkpoint and added
view-support gating:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --target_eval_mode always --policy_val_stride 4 \
  --init_checkpoint outputs/carnet/spcarnet_v290_patch_view_moe_20260630/v290a_patch_view_moe_policy_20260630/v180_perceptual_surface_decoder.pt \
  --skip_training \
  --surface_texture_mode lowrank_view_v2 --surface_texture_uv_bins 4 \
  --decoder_output_mode patch_view_moe --moe_expert_count 3 --moe_direct_scale 0.35 \
  --alpha_grid 0.5,0.75 \
  --view_support_gate_mode lowrank_view_cos \
  --view_support_min_cos 0.95 --view_support_min_concentration 0.15 \
  --view_support_power 1.0 --view_support_floor 0.35 \
  --policy_val_min_positive_view_fraction 0.75 \
  --policy_val_min_ssim_positive_view_fraction 0.75 \
  --policy_val_min_lpips_positive_view_fraction 0.75 \
  --policy_val_min_psnr_cvar_gain 0.0 \
  --policy_val_min_ssim_cvar_gain 0.0 \
  --policy_val_min_lpips_cvar_gain 0.0 \
  --compute_lpips --enable_wandb \
  --wandb_project spcarnet-v292-view-support \
  --output_dir outputs/carnet/spcarnet_v292_view_support_20260630/v292d_v290a_viewsupport_floor035_forced_exact_20260630
```

W&B was enabled in offline mode for the medium/target exact runs. Important
offline run paths:

```text
outputs/carnet/spcarnet_v290_patch_view_moe_20260630/v290a_patch_view_moe_policy_20260630/wandb/offline-run-20260630_141013-ifmeenke
outputs/carnet/spcarnet_v292_view_support_20260630/v292d_v290a_viewsupport_floor035_forced_exact_20260630/wandb/offline-run-20260630_145015-e6h3o7f1
```

## Results

All target exact runs used stripped target evidence first. Target GT was used
only after apply for final evaluation. The no-target-GT audit passed for all
listed exact runs.

| run | target PSNR | target SSIM | target LPIPS | PSNR gain | SSIM gain | LPIPS gain | changed frac. | no-GT | Phase-J gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| v266c old LPIPS reference | 19.845698 | 0.620201 | 0.179915 | +0.013644 | +0.000290 | +0.000419 | n/a | pass | fail |
| v289c target-compatible source weighting | 19.841839 | 0.620214 | 0.180080 | +0.009785 | +0.000303 | +0.000255 | n/a | pass | fail |
| v290a raw PatchViewMoE | 19.850131 | 0.619529 | 0.180489 | +0.018077 | -0.000381 | -0.000155 | 0.119689 | pass | fail |
| v291b structure-gain tail-safe | 19.845900 | 0.619911 | 0.180715 | +0.013846 | +0.000000 | -0.000380 | 0.101699 | pass | fail |
| v292b view-support floor 0.25 | 19.850320 | 0.620385 | 0.180131 | +0.018266 | +0.000474 | +0.000204 | 0.106106 | pass | fail |
| v292c view-support floor 0.15 | 19.848830 | 0.620398 | 0.180057 | +0.016777 | +0.000487 | +0.000278 | 0.092727 | pass | fail |
| **v292d view-support floor 0.35** | **19.851452** | **0.620343** | **0.180212** | **+0.019398** | **+0.000432** | **+0.000123** | **0.111852** | **pass** | **fail** |
| v292e view-support floor 0.00 | 19.845929 | 0.620358 | 0.180018 | +0.013875 | +0.000447 | +0.000317 | 0.049314 | pass | fail |

v292d is the current frontier because it gives the highest target exact PSNR and
still keeps SSIM/LPIPS positive versus the parent. Compared with v290a, the
view-support gate fixes the main failure mode:

```text
v290a target gains: +0.018077 PSNR / -0.000381 SSIM / -0.000155 LPIPS
v292d target gains: +0.019398 PSNR / +0.000432 SSIM / +0.000123 LPIPS
```

The gate still fails Phase-J flowers because PSNR remains too low:

```text
v292d PSNR gap to Phase-J flowers: -0.452906
v292d SSIM margin over Phase-J flowers: +0.062573
v292d LPIPS margin over Phase-J flowers: +0.149010 lower-is-better margin
```

## Qualitative Outputs

Target exact qualitative outputs:

```text
outputs/carnet/spcarnet_v292_view_support_20260630/v292d_v290a_viewsupport_floor035_forced_exact_20260630/target_exact_fixed_policy
```

Compact report panel:

```text
docs/car_model/assets/v292d_view_support_flowers_exact_panel.png
docs/car_model/assets/v292d_view_support_flowers_exact_panel_manifest.json
```

The panel shows `parent / v292d / |delta| x8 / GT`. The difference image is
only a visualization aid; metrics above remain the evidence.

## Bottleneck Diagnosis

The v169 prompt correctly identified two failure modes:

1. A stronger surface carrier is needed; scalar and local RGB residuals were
   too weak.
2. Policy-val success can fail on target if source-view support is not modeled.

v290-v292 addresses both partially. The new representation can apply a much
larger residual than v168-era near-no-op carriers, with target changed fraction
around `0.09-0.12`. The view-support gate turns v290a's target SSIM/LPIPS
failure into an all-axis parent win.

The remaining hard bottleneck is not no-GT leakage or storage. It is that the
baked surface representation is still roughly `0.45 dB` below the Phase-J
flowers PSNR endpoint. That means the carrier is useful but still under-capacity
or under-aligned for the teacher endpoint. Under the v169 prompt, this blocks
full9.

## Machine-Readable Summary

```text
docs/car_model/results/v290_v292_patch_view_moe_view_support_summary.json
```

## Verdict

```text
Final status: NOT COMPLETE.
```

This is a real milestone because v292d is the first PatchViewMoE variant that
keeps the high PSNR of v290a while repairing target SSIM and LPIPS. It is not a
closed paper-level result because the Phase-J flowers PSNR gate still fails.
