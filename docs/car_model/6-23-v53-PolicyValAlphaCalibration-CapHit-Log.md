# v53 Policy-Val Alpha Calibration Cap-Hit Probe

Date: 2026-06-23

Status: `NOT_PROMOTED`. v53 is a real representation-level method change, but the cap-hit validation does not justify replacing v52.

## Motivation

v48/v52 often select the maximum hand-specified alpha in the conservative grid (`0.125`). This suggests that the surface residual atlas may be under-applying reliable residuals. v53 tests whether alpha can be estimated from train policy-val evidence instead of manually extending the grid.

The method adds a closed-form policy-val least-squares alpha candidate:

```text
alpha* = argmin_alpha || teacher_residual - alpha * atlas_residual||^2
```

It uses only train policy-val residual samples. The generated alpha candidates are then passed through the existing relative-gain, SSIM, image-L1, tail, and min-view gates. Held-out test metrics are used only after the policy has made its decision.

## Implementation

Modified scripts:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New apply flags:

```text
--enable_policy_val_alpha_calibration
--alpha_calibration_max_alpha
--alpha_calibration_multipliers
--alpha_calibration_min_denominator
```

Runner additions:

```text
--enable_policy_val_alpha_calibration
--alpha_calibration_max_alpha
--alpha_calibration_multipliers
--wandb_project
--wandb_run_name
--wandb_group
```

The implementation is default-off, so existing v48/v51/v52 commands are unchanged.

## Validation Commands

The probe was run on the three v52 cap-hit scenes with the v51 support ladder fixed. This isolates alpha calibration from texture/fill/support parameter scanning.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 6 \
  --output_root /dev/shm/peilincai_spcarnet_v53_alpha_calib_counter_20260623 \
  --tag v53_policyval_alpha_calib_support_ladder_tex32_nearest_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 2048,4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_alpha_calibration \
  --alpha_calibration_max_alpha 0.5 \
  --alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --min_policy_val_l1_positive_view_fraction 1.0 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v53_policyval_alpha_calibration \
  --wandb_run_name v53_alpha_calib_counter_20260623 \
  --force
```

The same command was run for `kitchen` on GPU `3` and `bonsai` on GPU `2`, using:

```text
/dev/shm/peilincai_spcarnet_v53_alpha_calib_caphit_20260623
```

## W&B Runs

| scene | W&B run |
|---|---|
| counter | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/9dwsrs2n` |
| kitchen | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/6l9dcx6s` |
| bonsai | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/pxobsvd0` |

## Results vs v52

| scene | accepted | selected alpha | changed | dPSNR | dSSIM | dLPIPS | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| counter | 1 | `0.0625` | `6.5362%` | `-0.001730` | `-0.00002551` | `+0.00007340` | worse than v52 |
| kitchen | 1 | `0.5000` | `3.9361%` | `+0.004459` | `-0.00009871` | `-0.00024672` | PSNR/LPIPS up, SSIM down |
| bonsai | 0 | rejected/no-op | `0.0000%` | `-0.004087` | `-0.00007844` | `+0.00012958` | correctly rejected but fallback below v52 |

## Policy Diagnostics

### counter

The calibrated alpha estimate hits the configured cap:

```text
raw alpha = 1.5017878538291516
calibrated alpha = 0.5
generated candidates = 0.25, 0.375, 0.5
```

The risk gate selects `0.0625`, not the calibrated high-alpha candidate. This is train-policy safe, but held-out metrics are worse than v52, so counter should remain v52.

### kitchen

The calibrated high-alpha candidate passes all train gates:

```text
raw alpha = 1.3038363506271733
selected alpha = 0.5
policy-val SSIM gain = +0.0004470497
policy-val L1 gain = +0.0000589479
```

Held-out PSNR and LPIPS improve clearly, but SSIM decreases. This means the current train-policy SSIM proxy is not sufficient to guarantee held-out SSIM preservation under high-alpha residual application.

### bonsai

The calibrated high-alpha candidate is rejected:

```text
raw alpha = 2.148449977604493
calibrated alpha = 0.5
safe alpha count = 0
```

The risk reasons are meaningful: negative min-view SSIM and image-L1 tail failures. A code-level audit fix was added after this run so future rejected candidates report final `selected_alpha=0.0` instead of the unsafe best alpha.

## Decision

Do not promote v53 into the main policy.

Reasons:

- It is not a three-metric strict improvement on the cap-hit set.
- It improves `kitchen` PSNR/LPIPS, but the SSIM regression violates the current paper-facing standard.
- It makes `counter` worse than v52.
- `bonsai` is correctly rejected, but the fallback result is below v52 because v52 already has a valid accepted atlas for that scene.

## Lesson

Policy-val least-squares alpha calibration is useful as a diagnostic: the atlas is indeed under-scaled in some scenes. However, simply increasing alpha is not a reliable paper endpoint. The next representation-level method should be more local than a single scene-level alpha, for example:

- per-face or per-region calibrated alpha with tail-safe gates;
- SSIM-aware local residual clipping rather than global scalar scaling;
- view-consistency and visibility-stratified alpha certification;
- faster candidate evaluation so local policies can be validated without exploding runtime.

This is a constructive failure: it identifies that amplitude is a real bottleneck, but global alpha is too blunt.
