# v90 Adaptive Source-Mixture Implementation Log

Date: 2026-06-25  
Status: running counter validation  
W&B: `SPCarNet/v90_source_mixture_adaptive/v90_source_mixture_adaptive_counter_20260625`, run id `clkmpc8x`

## Motivation

`v87_source_mixture` implemented a real method change: policy-val prior-bin hybrid no longer hard-copies source bins, but fits a continuous source-mixture weight:

```text
mixed_bin = baseline_bin + w * (source_bin - baseline_bin)
```

The mechanism ran correctly but did not beat the existing v84/v82b counter anchor:

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| v84/v82b counter anchor | `26.756137847900` | `0.862126350403` | `0.251690655947` | current anchor |
| v87 source-mixture | `26.756130218506` | `0.862126231194` | `0.251691371202` | not promoted |

The audit showed the likely failure mode:

- candidate bins: `427802`
- allowed bins: `615`
- mean source-mixture weight: `0.0047563986`
- max source-mixture weight: `0.2752484326`

The fitted weights were extremely small because v87 used a fixed absolute ridge of `0.01`, while per-bin denominators were often around `1e-6`. That made the regularizer dominate the actual policy-val signal.

## Method Change

v90 adds a scale-aware ridge mode:

```text
--source_mixture_ridge_mode adaptive_den
```

The old behavior is preserved by:

```text
--source_mixture_ridge_mode absolute
```

In `adaptive_den`, the per-bin ridge term is:

```text
ridge_term = source_mixture_ridge * max(local_denominator, median_positive_denominator)
```

Then the source-mixture weight is:

```text
w = clip(numerator / (denominator + ridge_term), 0, 1)
```

This is a method-level correction rather than per-scene parameter searching: the ridge is normalized by the scale of the policy-val evidence collected in the current candidate pool.

## Code Changes

Updated files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_l1risk_fairnoop_scene.py`

Key additions:

- adapter CLI: `--source_mixture_ridge_mode {absolute,adaptive_den}`
- runner CLI: `--source_mixture_ridge_mode {absolute,adaptive_den}`
- runner-to-adapter forwarding
- W&B/config logging for `source_mixture_ridge_mode`
- W&B scalar logging for:
  - `policy/source_mixture_den_reference`
  - `policy/source_mixture_ridge_term_mean`
- audit logging for:
  - `source_mixture_ridge_mode`
  - `source_mixture_den_reference`
  - `source_mixture_ridge_term_mean/min/max`
  - per-row `source_mixture_ridge_term`

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

Both entrypoints expose the new argument in `--help`.

## Running Command

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v90_source_mixture_adaptive WANDB_MODE=online CUDA_VISIBLE_DEVICES=2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v90_source_mixture_adaptive_20260625 \
  --tag v90_source_mixture_adaptive_counter_region_texture_adapter \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_block_sizes 1,2,3 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend_candidates 0,0.5,1.0 \
  --enable_policy_val_prior_bin_gain_hybrid \
  --enable_policy_val_source_mixture \
  --source_mixture_ridge_mode adaptive_den \
  --source_mixture_ridge 1.0 \
  --source_mixture_min_weight 0.001 \
  --prior_bin_gain_hybrid_min_bin_samples 4 \
  --prior_bin_gain_hybrid_min_views 1 \
  --prior_bin_gain_hybrid_min_positive_view_fraction 0.5 \
  --enable_target_footprint_bin_certificate \
  --target_footprint_min_bin_pixels 8 \
  --target_footprint_min_views 1 \
  --enable_target_footprint_tail_risk_certificate \
  --target_footprint_tail_risk_min_positive_view_fraction 0.75 \
  --target_footprint_tail_risk_min_min_view_gain -0.0000001 \
  --target_footprint_tail_risk_min_cvar20_view_gain 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v90_source_mixture_adaptive \
  --wandb_run_name v90_source_mixture_adaptive_counter_20260625 \
  --wandb_mode online \
  --force
```

## Promotion Rule

v90 should only be promoted if it beats the v84/v82b counter anchor on all three held-out RGB metrics, or if it gives a clear geometry/coverage advantage without RGB regression:

```text
anchor = 26.756137847900 PSNR / 0.862126350403 SSIM / 0.251690655947 LPIPS
```

If it promotes on counter, the next step is a fixed-policy hard-triad run on `counter,kitchen,bonsai`, then full9. If it does not promote, the conclusion is that source-mixture smoothing is not enough and the next method should target coverage/support learning rather than bin interpolation.
