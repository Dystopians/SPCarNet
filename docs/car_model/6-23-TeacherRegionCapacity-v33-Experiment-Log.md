# 6-23 Teacher Region Capacity v33 Experiment Log

日期：2026-06-23

## Motivation

v31 teacher-surface basis 证明 `teacher_residual_rgb / teacher_residual_l1`
可以进入 surface evidence cache，并被 face-local SH1 residual operator 读取。
v32 teacher-region carrier 进一步把候选从 face-only 扩展到 render-visible
region evidence，但 Bonsai full-res 最好结果仍只是 `+0.00091` PSNR
vs compact base，SSIM/LPIPS 微退。

v33 的问题很明确：

> 如果瓶颈只是 face-local 表达容量太小，那么提高 SH 阶数、加入
> train-only validation gain、或把多个 face 约束到 shared residual field
> 应该能带来超过噪声级的 held-out full-res 改善。

因此本轮跑两个 Bonsai full-resolution pilot：

1. independent face-local `SH3 + global_gain`；
2. shared RBF residual field `SH3 + global_gain`。

两者都继续使用 train-only policy gate，不使用 held-out test GT 做选择。

## Inputs

```text
source compact model:
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model

teacher-region evidence:
outputs/carnet/meshsplatopt/ecsr_phase_v32_teacher_patch_carrier/bonsai_teacher_region_evidence

teacher render-visible carrier:
outputs/carnet/meshsplatopt/ecsr_phase_v32_teacher_patch_carrier/bonsai_teacher_render_visible_region_carriers.json
```

## Code / Method Settings

No new code edit was required for v33. This experiment uses existing
operator capacity already implemented in:

```text
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
```

Key settings shared by both pilots:

```text
--top_k 4096
--sh_degree 3
--strength 0.28
--max_abs_delta_rgb 0.12
--validation_shrink_mode global_gain
--validation_gain_max_scale 1.35
--region_carrier_json bonsai_teacher_render_visible_region_carriers.json
--render_region_objective
--render_region_core_weight 2.0
--render_region_context_weight 0.5
--render_region_outside_penalty 0.03
--render_region_tail_cvar_weight 0.05
--bystander_zero_delta_weight 0.015
--direction_luma_safety_weight 0.005
--direction_cosine_weight 0.005
--max_faces_to_apply 512
--residual_rgb_key teacher_residual_rgb
--residual_l1_key teacher_residual_l1
```

The shared-field pilot additionally uses:

```text
--shared_residual_field
--shared_residual_field_anchors 96
--shared_residual_field_lr 0.020
--shared_residual_field_weight_l2 0.00008
--shared_residual_field_view_hinge_weight 0.04
--shared_residual_field_duplicate_smooth_weight 0.008
```

## Commands and Artifacts

### Independent SH3 + global gain

Fit log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/fit_bonsai_teacher_region_sh3_globalgain_gpu2.log
```

Output model:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/bonsai_teacher_region_sh3_globalgain
```

Full-res render/metrics:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/render_bonsai_teacher_region_sh3_globalgain_fullres_gpu5_retry.log
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/metrics_bonsai_teacher_region_sh3_globalgain_fullres_gpu5_retry.log
```

Note: the first full-res render attempt on GPU2 OOMed because only about
`650MB` was free after model loading. The successful fair full-res render
was rerun on GPU5.

### Shared-field SH3 + global gain

Fit log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/fit_bonsai_teacher_region_sharedfield_sh3_gpu3.log
```

Output model:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/bonsai_teacher_region_sharedfield_sh3
```

Full-res render/metrics:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/render_bonsai_teacher_region_sharedfield_sh3_fullres_gpu5_retry.log
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/logs/metrics_bonsai_teacher_region_sharedfield_sh3_fullres_gpu5_retry.log
```

Note: the first full-res render attempt on GPU3 OOMed for the same reason.
The successful fair full-res render was rerun on GPU5.

## Train-Only Gate Results

| pilot | accepted | selected faces | accepted faces | vertices added | policy-val final relative gain | validation shrink |
|---|---:|---:|---:|---:|---:|---|
| SH3 global-gain | true | 3470 | 46 | 138 | 0.536536 | global scale 0.833691 |
| shared-field SH3 | true | 3470 | 49 | 147 | 0.660131 | global scale 0.578035 |

Shared-field has the stronger train-only proxy and higher accepted-face count,
so if the route were capacity-limited only, it should have improved held-out
metrics more clearly.

## Full-Resolution Held-Out Metrics

Fair Bonsai full-resolution metrics:

| method | PSNR | SSIM | LPIPS | dPSNR compact | dSSIM compact | dLPIPS compact | dPSNR clean | dSSIM clean | dLPIPS clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected clean `ours_26000` | 28.895233 | 0.896400 | 0.259493 | +0.030893 | +0.000388 | +0.000153 | +0.000000 | +0.000000 | +0.000000 |
| compact base | 28.864340 | 0.896012 | 0.259340 | +0.000000 | +0.000000 | +0.000000 | -0.030893 | -0.000388 | -0.000153 |
| v31 face-local SH1 | 28.864361 | 0.896006 | 0.259352 | +0.000021 | -0.000006 | +0.000012 | -0.030872 | -0.000394 | -0.000141 |
| v32 region minpilot | 28.865252 | 0.896005 | 0.259347 | +0.000912 | -0.000007 | +0.000008 | -0.029982 | -0.000395 | -0.000145 |
| v33 SH3 global-gain | 28.864204 | 0.895998 | 0.259350 | -0.000135 | -0.000014 | +0.000010 | -0.031029 | -0.000402 | -0.000143 |
| v33 shared-field SH3 | 28.865301 | 0.895999 | 0.259353 | +0.000961 | -0.000013 | +0.000013 | -0.029932 | -0.000401 | -0.000140 |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 | +2.966772 | +0.033879 | -0.086937 |

## Interpretation

v33 is an honest negative result.

The route did improve train-only proxy substantially, but that did not convert
to non-noise held-out full-resolution gains:

- independent SH3 is slightly worse than compact base on PSNR and SSIM;
- shared-field SH3 is only `+0.00096` PSNR vs compact and has worse SSIM/LPIPS;
- neither comes close to selected clean, and both are far from Phase-J ELA;
- therefore “raise SH degree / add global validation gain / share RBF field”
  is not enough to solve representation-level baking.

The bottleneck is likely not only coefficient capacity. More likely causes:

1. teacher residual is image/render-time and not directly representable by
   sparse face-local color deltas at this checkpoint resolution;
2. selected surface samples cover very small visible residual support compared
   with ELA's image-space support aggregation;
3. policy-val proxy is sample-local and does not fully predict full-frame
   SSIM/LPIPS;
4. applying tiny local face edits to a 9.55M-triangle mesh changes too little of
   the held-out render to matter.

## Status

`NOT PROMOTED`.

v33 should be cited as bottleneck evidence, not as a method result. The current
paper-facing endpoint remains Phase-J guarded adaptive ELA.

Next method direction should not be another small face-local SH capacity
increase. The next credible route is one of:

1. larger support representation: patch/region-level residual texture or
   surface atlas carrier, not per-face sparse deltas;
2. differentiable render-level training that optimizes final held-out-style
   image loss through train-only split views after applying teacher-region
   priors;
3. a formal hybrid method position where compact checkpoint + train-evidence
   render-time ELA is the core contribution, and representation baking is an
   optional diagnostic branch.

