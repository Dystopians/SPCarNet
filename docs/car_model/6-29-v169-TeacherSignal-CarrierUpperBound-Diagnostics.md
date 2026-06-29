# 6-29 v169 Teacher Signal And Carrier Upper-Bound Diagnostics

Date: 2026-06-29

This note follows `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.
The goal was not to launch another full9 run. The goal was to answer the sharper
v169 question:

> Is the Phase-J teacher residual real, and can the current baked surface carrier
> project it strongly enough to justify exact target promotion without target/test
> GT leakage?

## Storage And Execution Preflight

Storage at the start of this diagnostic:

| mount | status |
|---|---|
| `/data` | about `800K-860K` free, effectively full |
| `/dev/shm` | about `13G` free |
| `/tmp` | about `6.1T` free |

Because `/data` is full, all new experiment outputs were written to `/dev/shm`.
No large image artifacts were copied into the repository in this step.

GPU choice:

- GPU 2 and GPU 3 were essentially free before the diagnostic.
- The diagnostic is mostly CPU/IO plus LPIPS inference; commands used
  `CUDA_VISIBLE_DEVICES=2` or `CUDA_VISIBLE_DEVICES=3`.
- These were diagnostics, not medium/long training runs, so W&B was not used.

## Code Changes

New diagnostic script:

- `scripts/car_model/analyze_v169_teacher_signal_audit.py`

It reads train-fit / policy-val teacher surface evidence and writes JSON plus
Markdown. It measures:

- whether `teacher_residual_rgb_raw` and `teacher_residual_rgb` are nonzero;
- raw-vs-masked residual L1/L2 retention;
- clipping retention under `max_abs_delta_rgb`;
- active visible pixel and active face fractions;
- luma-gradient residual energy;
- RGB sign consistency;
- parent vs raw-teacher and masked-teacher PSNR/L1 gains on policy-val GT;
- explicit `target_or_test_gt_usage = none`.

Reliability fix:

- `scripts/car_model/train_surface_conditioned_residual_unet.py`
  now returns `TrainBatch(features, face_ids, parent, teacher, gt)` in the
  no-crop fallback branch of `_sample_patch`.
- This fixes a real interface bug for `patch_size <= 0` or tiny images. The main
  v191/v192 runs used cropped patches, so this is a robustness fix rather than
  a reinterpretation of those metrics.

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/train_surface_conditioned_residual_unet.py \
  scripts/car_model/analyze_v169_teacher_signal_audit.py
```

Result: passed.

## Teacher Signal Audit

### Flowers

Artifacts:

- JSON: `/dev/shm/peilincai_spcarnet_v169_teacher_signal_flowers/v169_flowers_teacher_signal_audit.json`
- Markdown: `/dev/shm/peilincai_spcarnet_v169_teacher_signal_flowers/v169_flowers_teacher_signal_audit.md`

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TMPDIR=/dev/shm/peilincai_tmp_v169_teacher_audit \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_v169_teacher_signal_audit.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --output_json /dev/shm/peilincai_spcarnet_v169_teacher_signal_flowers/v169_flowers_teacher_signal_audit.json \
  --output_md /dev/shm/peilincai_spcarnet_v169_teacher_signal_flowers/v169_flowers_teacher_signal_audit.md \
  --policy_val_stride 4 \
  --min_alpha 0.03 \
  --max_abs_delta_rgb 0.12
```

Policy-val result:

| scene | views | raw L1 | masked L1 | mask L1 retention | clip L2 retention | active valid frac | active face frac | parent PSNR | raw teacher PSNR gain | masked teacher PSNR gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | 12 | 0.014603 | 0.008392 | 0.574638 | 0.970615 | 0.329769 | 0.540518 | 20.464215 | +0.428486 | +0.841581 |

Verdict:

- The teacher-parent residual is real.
- The masked teacher residual is also real and improves parent PSNR on policy-val.
- About `57.46%` of raw residual L1 survives the better-mask target. This is
  useful but also shows substantial teacher-signal dilution before projection.

### Counter

Artifacts:

- JSON: `/dev/shm/peilincai_spcarnet_v169_teacher_signal_counter/v169_counter_teacher_signal_audit.json`
- Markdown: `/dev/shm/peilincai_spcarnet_v169_teacher_signal_counter/v169_counter_teacher_signal_audit.md`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TMPDIR=/dev/shm/peilincai_tmp_v169_teacher_counter \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_v169_teacher_signal_audit.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/teacher_surface_evidence \
  --output_json /dev/shm/peilincai_spcarnet_v169_teacher_signal_counter/v169_counter_teacher_signal_audit.json \
  --output_md /dev/shm/peilincai_spcarnet_v169_teacher_signal_counter/v169_counter_teacher_signal_audit.md \
  --policy_val_stride 4 \
  --min_alpha 0.03 \
  --max_abs_delta_rgb 0.12
```

Policy-val result:

| scene | views | raw L1 | masked L1 | mask L1 retention | clip L2 retention | active valid frac | active face frac | parent PSNR | raw teacher PSNR gain | masked teacher PSNR gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 12 | 0.015501 | 0.009802 | 0.632333 | 0.942966 | 0.295403 | 0.521092 | 27.152916 | +1.881452 | +2.651850 |

Verdict:

- Counter has an even stronger teacher signal than flowers under this diagnostic.
- Therefore the counter failure is not because Phase-J provides no residual.
  It is more likely carrier/representation capacity, view transfer, or target
  generalization.

## Carrier Projection Upper Bound

### Flowers, fast proxy and full-image PSNR rescan

Artifacts:

- fast proxy JSON: `/dev/shm/peilincai_spcarnet_v169_upper_bound_flowers/v169_flowers_upper_bound.json`
- fast proxy Markdown: `/dev/shm/peilincai_spcarnet_v169_upper_bound_flowers/v169_flowers_upper_bound.md`
- full-image PSNR JSON: `/dev/shm/peilincai_spcarnet_v169_upper_bound_flowers_fullpsnr/v169_flowers_upper_bound_fullpsnr.json`
- full-image PSNR Markdown: `/dev/shm/peilincai_spcarnet_v169_upper_bound_flowers_fullpsnr/v169_flowers_upper_bound_fullpsnr.md`

Full-image PSNR command:

```bash
CUDA_VISIBLE_DEVICES=2 TMPDIR=/dev/shm/peilincai_tmp_v169_ub_flowers_fullpsnr \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_v169_policy_val_upper_bound.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --output_json /dev/shm/peilincai_spcarnet_v169_upper_bound_flowers_fullpsnr/v169_flowers_upper_bound_fullpsnr.json \
  --output_md /dev/shm/peilincai_spcarnet_v169_upper_bound_flowers_fullpsnr/v169_flowers_upper_bound_fullpsnr.md \
  --texture_sizes 8,16 \
  --alpha_grid 0,0.03125,0.0625,0.125 \
  --policy_val_stride 4 \
  --max_faces 2048 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_samples_per_view 60000 \
  --teacher_distilled_basis_mode low_rank_view_texture_rich_k4 \
  --teacher_distilled_low_rank_texture_rank 4 \
  --policy_val_ssim_max_size 512 \
  --policy_val_l1_max_size 512 \
  --policy_val_lpips_max_size 256 \
  --enable_full_image_psnr_rescan
```

Key result:

| scene | carrier | best alpha | PSNR gain | SSIM gain | LPIPS gain | SSIM positive frac | LPIPS positive frac | robust pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| flowers | low-rank rich K4, texture 16 | 0.03125 | +0.000168 | +0.000000402 | +0.00000244 | 0.500000 | 0.500000 | false |

Interpretation:

- The fast proxy run with larger alpha values failed SSIM/LPIPS outright.
- The stricter full-image PSNR rescan finds a formally positive all-axis row,
  but the gain is microscopic and not tail-robust.
- SSIM and LPIPS positive-view fractions are only `0.5`, and their CVaR/min-view
  gains are negative.
- This does not justify a flowers exact/full9 promotion from the current carrier.
  It just proves the carrier is not entirely dead.

### Counter policy-val upper bound

Artifacts:

- JSON: `/dev/shm/peilincai_spcarnet_v169_upper_bound_counter/v169_counter_upper_bound.json`
- Markdown: `/dev/shm/peilincai_spcarnet_v169_upper_bound_counter/v169_counter_upper_bound.md`

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TMPDIR=/dev/shm/peilincai_tmp_v169_ub_counter \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_v169_policy_val_upper_bound.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_v115_counter_v106anchor_20260626_1555/counter/teacher_surface_evidence \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json \
  --output_json /dev/shm/peilincai_spcarnet_v169_upper_bound_counter/v169_counter_upper_bound.json \
  --output_md /dev/shm/peilincai_spcarnet_v169_upper_bound_counter/v169_counter_upper_bound.md \
  --texture_sizes 8,16 \
  --alpha_grid 0,0.03125,0.0625,0.125 \
  --policy_val_stride 4 \
  --max_faces 2048 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_samples_per_view 60000 \
  --teacher_distilled_basis_mode low_rank_view_texture_rich_k4 \
  --teacher_distilled_low_rank_texture_rank 4 \
  --policy_val_ssim_max_size 512 \
  --policy_val_l1_max_size 512 \
  --policy_val_lpips_max_size 256
```

Key result:

| scene | carrier | best robust alpha | PSNR gain | SSIM gain | LPIPS gain | SSIM positive frac | LPIPS positive frac | robust pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| counter | low-rank rich K4, texture 16 | 0.0625 | +0.076027 | +0.00005296 | +0.00005631 | 1.000000 | 1.000000 | true |

Important caveat:

- Counter PSNR here is `policy_val_residual_sample_proxy`, not full-image PSNR.
- This is a policy-val upper-bound diagnostic, not a held-out target/test win.
- It explains why counter looked promising in training, but it does not overturn
  the official target result: v192 counter still loses Phase-J on PSNR/SSIM even
  though it wins LPIPS.

## Integrated Diagnosis

The v169 diagnosis is now sharper than before:

1. The Phase-J teacher residual exists on both flowers and counter.
2. Direct teacher correction improves the parent strongly on policy-val PSNR.
3. Current masking keeps only about `57%` to `63%` of raw residual L1.
4. The current carrier can project some residual, but flowers projection is
   almost noise-scale and non-robust.
5. Counter policy-val projection is more promising, but previous target exact
   evidence shows that policy-val success does not automatically transfer to
   held-out target/test views or beat Phase-J.

Therefore the remaining bottleneck is not the absence of a teacher signal. The
bottleneck is a representation and certification gap:

- the carrier is too weak on flowers to support a paper-quality exact promotion;
- the train/policy-val certificate is not strong enough to guarantee counter
  target/test all-axis gains against Phase-J;
- GT-assisted U-Net can pass flowers, but teacher-only U-Net failed, so the
  current successful result cannot be claimed as pure teacher distillation.

## Method Direction After This Diagnostic

Do not continue with alpha scans, footprint expansion, or simple scalar RGB atlas
as the main contribution.

The next real method change should be a surface-attached representation with
stronger target/test transfer:

1. per-face or face-group feature texture with a compact decoder;
2. view-dependent low-rank residual bases with explicit tail-risk regularization;
3. policy-val gate trained to reject views/faces that are Phase-J-positive on
   average but structurally unsafe in the lower tail;
4. teacher-only and GT-assisted variants reported separately.

The exact gate for promotion remains unchanged:

- flowers exact must beat Phase-J all-axis;
- target/test apply must not see target/test RGB GT;
- fixed policy must then run full9 before any paper-final claim.

## Status

Final status: NOT COMPLETE.

Reason:

- v191 flowers passes Phase-J, but it is GT-assisted.
- v194 teacher-only flowers fails exact.
- current flowers carrier upper-bound is not robust.
- counter policy-val carrier looks promising, but official target exact still
  does not beat Phase-J all-axis.

Exact next command if continuing from this diagnostic:

```bash
WANDB_MODE=offline \
WANDB_DIR=/dev/shm/peilincai_spcarnet_v195_surface_feature_decoder/wandb \
CUDA_VISIBLE_DEVICES=<low_or_mid_load_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_conditioned_residual_unet.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt \
  --target_eval_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented \
  --scene_name flowers \
  --steps 3200 \
  --train_max_side 640 \
  --patch_size 320 \
  --base_channels 32 \
  --teacher_l1_weight 1.0 \
  --teacher_ssim_weight 0.20 \
  --teacher_grad_weight 0.08 \
  --gt_l1_weight 0 \
  --gt_ssim_weight 0 \
  --gt_lpips_weight 0 \
  --gt_grad_weight 0 \
  --compute_lpips \
  --enable_wandb
```

This command is a teacher-only continuation baseline, not a guaranteed final
method. The actual research upgrade should add surface-attached features or
view-dependent low-rank bases rather than only changing loss weights.
