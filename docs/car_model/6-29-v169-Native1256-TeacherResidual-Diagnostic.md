# SPCarNet v169 Native1256 Teacher-Residual Diagnostic

Date: 2026-06-29

## Verdict

**NOT a flowers hard-gate success.**  The v169 route was repaired to the correct native1256 evaluation protocol and trained with a real Phase-J-to-Phase-F teacher residual target, but exact test still fails badly against Phase-J.

The useful milestone is negative but important: the earlier v168/v240 evidence path had a resolution mismatch (`1600x1054` evidence vs `1256x828` Phase-J gate). This run fixes that protocol issue and shows the current surface-texture carrier only recovers a tiny fraction of the Phase-J gain.

## Prompt Contract

The active prompt is:

`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`

Hard flowers gate:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Prompt Phase-J threshold | > 20.304358 | > 0.557770 | < 0.329222 |

No full9 should be launched until this flowers exact gate passes all three axes.

## What Was Fixed

Previous v168/v240 evidence used raw-resolution surface evidence (`1600x1054`) while the Phase-J/Phase-F reference renders and GT used `1256x828`. This made exact comparison invalid.

This run rebuilt a native1256 evidence chain:

1. Rebased train-fit evidence to Phase-F parent at native1256 while preserving real train GT for policy-val.
2. Built true teacher residual evidence: `teacher_residual_rgb = Phase-J train render - Phase-F train render`.
3. Rebased target evidence to Phase-F parent at native1256 and stripped all target/test GT and residual fields.
4. Trained the surface-texture U-Net carrier with teacher-only losses and W&B offline logging.
5. Evaluated v169, Phase-F parent, and Phase-J reference under the same exact evaluator and GT.

## Storage / Runtime Preflight

At launch:

- `/data`: about `202M` free, unsuitable for new large artifacts.
- `/dev/shm`: about `1.7G` free, insufficient for another large evidence cache.
- `/tmp` user quota: about `96.75G / 100G` used after native1256 evidence creation.
- GPU selection: GPU1 was selected because it was essentially idle (`~501 MiB`, `0%` util).

Artifacts were kept under:

`/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629`

## Commands

Train-fit native1256 rebase:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v169_native1256_rebase PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_rebase_evidence_rgb_render_from_renders.py \
  --input_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence \
  --render_dir /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders \
  --gt_render_dir /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/gt \
  --output_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/fit_evidence_phasef_native1256 \
  --audit_path /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/fit_evidence_phasef_native1256_audit.json \
  --allow_resize --match_render_resolution --minimal_fields --recompute_residual_from_gt --force
```

Teacher residual cache:

```bash
TMPDIR=/tmp PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v169_native1256_teacher PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py \
  --base_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/fit_evidence_phasef_native1256 \
  --teacher_render_dir /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasej_trainval_gate/renders \
  --parent_render_dir /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders \
  --out_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/teacher_surface_evidence_phasej_to_phasef_native1256 \
  --copy_mode auto_link --rewrite_rgb_render_to_parent --selection_mode better_masked_residual \
  --teacher_parent_delta_min 0.005 --top_support_min_alpha 0.03 --top_support_limit 8192 --force
```

Target no-GT native1256 rebase:

```bash
TMPDIR=/tmp PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v169_native1256_target PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_rebase_evidence_rgb_render_from_renders.py \
  --input_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence \
  --render_dir /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_phasef_extra_compact_base/renders \
  --output_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/target_evidence_no_gt_phasef_native1256 \
  --audit_path /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/target_evidence_no_gt_phasef_native1256_audit.json \
  --allow_resize --match_render_resolution --minimal_fields --strip_target_gt_and_residuals --force
```

Training and target apply:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp WANDB_MODE=offline PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v169_native1256_train PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/train_surface_conditioned_residual_unet.py \
  --fit_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/teacher_surface_evidence_phasej_to_phasef_native1256 \
  --target_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/target_evidence_no_gt_phasef_native1256 \
  --surface_target_visible_evidence_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/target_evidence_no_gt_phasef_native1256 \
  --residual_rgb_key teacher_residual_rgb --policy_val_stride 4 --train_max_side 512 --patch_size 256 --steps 3200 --lr 0.00024 \
  --model_type surface_texture_unet --surface_texture_size 8 --surface_feature_dim 8 --surface_face_max_unique 8192 \
  --surface_face_min_alpha 0.03 --surface_face_min_residual_l1 0.0 --enable_surface_support_gate --lowrank_min_bin_support 8 \
  --base_channels 24 --max_delta 0.08 --confidence_mode sigmoid --confidence_bias -1.0 --confidence_min 0.0 --confidence_max 1.0 \
  --alpha_conditioned_residual --teacher_l1_weight 0.82 --teacher_ssim_weight 0.26 --teacher_lpips_weight 0.16 \
  --teacher_grad_weight 0.08 --teacher_highfreq_weight 0.10 --gt_l1_weight 0.0 --gt_ssim_weight 0.0 --gt_lpips_weight 0.0 \
  --gt_grad_weight 0.0 --gt_highfreq_weight 0.0 --lpips_loss_max_side 224 --grad_loss_max_side 256 --highfreq_loss_max_side 256 \
  --highfreq_loss_levels 3 --delta_l1_weight 0.00080 --alpha_grid 0,0.25,0.5,0.75,1 --policy_select_mode tail_guard \
  --policy_tail_fraction 0.35 --policy_min_psnr_gain -0.25 --policy_min_ssim_gain -0.01 --policy_min_lpips_gain -0.02 \
  --policy_cvar_psnr_gain -0.08 --policy_cvar_ssim_gain -0.003 --policy_cvar_lpips_gain -0.006 \
  --method_name ours_26000_v169_native1256_phasej_to_phasef_surface_texture_unet_teacheronly_flowers --scene_name flowers \
  --eval_tile 512 --eval_overlap 32 --ssim_max_side 512 --lpips_max_side 256 --compute_lpips --skip_policy_val_renders \
  --output_dir /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_phasej_to_phasef_surface_texture_unet_teacheronly_native1256 \
  --artifact_prefix v169_native1256_phasej_to_phasef_surface_texture_unet_teacheronly_flowers --seed 241 --enable_wandb \
  --wandb_run_name v169-native1256-phasej-to-phasef-surface-texture-unet-teacheronly-flowers
```

Exact evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v169_native1256_eval PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py \
  -m /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_phasej_to_phasef_surface_texture_unet_teacheronly_native1256/flowers_exact_target_apply \
  --split test \
  --methods ours_26000_v169_native1256_phasej_to_phasef_surface_texture_unet_teacheronly_flowers phasej_reference_native1256 phasef_parent_native1256 \
  --output /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_native1256_flowers_exact_results.json \
  --per_view_output /tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_native1256_flowers_exact_per_view.json \
  --merge_model_results
```

## Diagnostics

Teacher signal is real and nonzero:

| diagnostic | value |
|---|---:|
| processed train views | 46 |
| mean teacher-active fraction | 0.334879 |
| mean teacher target L1 | 0.009554 |
| mean raw Phase-J minus Phase-F L1 | 0.016889 |
| mean positive teacher gain L1 | 0.008468 |

Policy-val passed but the gains were tiny:

| item | value |
|---|---:|
| selected alpha | 0.25 |
| PSNR gain over parent | +0.002417 |
| SSIM gain over parent | +0.000117 |
| LPIPS gain over parent | +0.000101 |
| PSNR positive-view fraction | 1.000 |
| SSIM positive-view fraction | 1.000 |
| LPIPS positive-view fraction | 0.833 |
| mean policy-val changed fraction | 0.017808 |

Target no-GT apply changed too little:

| item | value |
|---|---:|
| target no-GT verifier | passed |
| selected alpha | 0.25 |
| mean target changed fraction | 0.023900 |
| mean known-face fraction | 0.112010 |
| mean active-support fraction | 0.062901 |
| mean active-support changed fraction | 0.432044 |

Exact image-delta recovery against Phase-J is only about `0.7%`:

| delta | mean image L1 |
|---|---:|
| v169 - Phase-F parent | 0.000137 |
| Phase-J - Phase-F parent | 0.019999 |
| v169 - Phase-J | 0.020009 |
| recovered teacher-parent delta fraction | 0.007001 |

## Exact Flowers Result

Same evaluator, same `1256x828` GT, 22 test views:

| method | PSNR | SSIM | LPIPS | vs Phase-F |
|---|---:|---:|---:|---|
| Phase-F parent native1256 | 19.668695 | 0.511678 | 0.394788 | baseline parent |
| v169 native1256 teacher-only | 19.670961 | 0.511814 | 0.394431 | tiny win |
| Phase-J reference native1256 | 20.300608 | 0.557458 | 0.329505 | far stronger |

Against Phase-F parent, v169 improves:

- PSNR: `+0.002266`
- SSIM: `+0.000136`
- LPIPS: `-0.000357`

Against Phase-J, v169 is worse by:

- PSNR: `-0.629646`
- SSIM: `-0.045644`
- LPIPS: `+0.064925`

It also fails the prompt's absolute Phase-J threshold (`20.304358 / 0.557770 / 0.329222`).

## Interpretation

The current carrier is not failing because the teacher signal is zero, because target GT leaked, or because the resolution protocol is wrong. Those were checked and fixed.

The failure is representation and coverage:

- The teacher residual exists on about one third of train-policy pixels, but target apply changes only about `2.39%` of pixels.
- Target known-face fraction is only `11.2%`; after support gating, active support is only `6.29%`.
- The selected alpha is conservative (`0.25`) because policy-val gains are real but tiny.
- The candidate recovers about `0.7%` of the Phase-J minus Phase-F image delta on exact test.
- Therefore, this surface-texture U-Net carrier is currently a small parent repair, not a successful Phase-J distillation mechanism.

## Required Next Step

Do **not** launch full9 from this checkpoint.

The next real representation change should address target coverage and view-dependent residual capacity directly. The strongest next experiment is:

1. Build a target-visible face expansion that is learned from train residual statistics but allocates more capacity to target-visible unseen/low-support faces without target RGB.
2. Replace the sparse changed-pixel carrier with a view-conditioned low-rank residual texture whose output is dense over target-known faces, not only current active supports.
3. Add a policy-val gate that explicitly predicts target exact delta coverage: require a minimum teacher-parent delta recovery fraction on held-out policy-val, not just small all-axis metric gains.
4. Re-run flowers exact only; full9 remains blocked until the Phase-J all-axis gate is passed.

## Artifact Index

- Train-fit native1256 audit: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/fit_evidence_phasef_native1256_audit.json`
- Teacher residual summary: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/teacher_surface_evidence_phasej_to_phasef_native1256/teacher_surface_evidence_summary.json`
- Target no-GT audit: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/target_evidence_no_gt_phasef_native1256_audit.json`
- W&B offline run: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_phasej_to_phasef_surface_texture_unet_teacheronly_native1256/wandb/offline-run-20260629_155935-vd406zb4`
- Train/apply report: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_phasej_to_phasef_surface_texture_unet_teacheronly_native1256/v169_native1256_phasej_to_phasef_surface_texture_unet_teacheronly_flowers_report.json`
- Exact metrics: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_native1256_flowers_exact_results.json`
- Exact per-view metrics: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_native1256_flowers_exact_per_view.json`
- Exact delta recovery summary: `/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629/v169_native1256_flowers_exact_delta_summary.json`
