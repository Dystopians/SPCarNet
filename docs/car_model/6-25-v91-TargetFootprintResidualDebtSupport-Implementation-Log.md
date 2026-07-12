# v91 Target-Footprint Residual-Debt Support Implementation Log

Date: 2026-06-25  
Status: counter probe running  
W&B: `SPCarNet/v91_target_debt_support/v91_target_debt_support_counter_20260625`, run id `ac4hggum`

## Motivation

The representation-level atlas line repeatedly hit a small-gain bottleneck. Recent evidence:

- v84/v82b counter anchor: `26.756137847900 / 0.862126350403 / 0.251690655947`
- v87 source-mixture: `26.756130218506 / 0.862126231194 / 0.251691371202`, not promoted
- v87 mean source-mixture weight: `0.0047563986`, showing the surviving certified bins carried very weak signal

This suggests the bottleneck is not only how to blend already-selected bins. The larger issue is support allocation: the atlas may not spend its limited support budget on surfaces that are both train-residual-heavy and visible in target trajectories.

## Method

v91 adds a new support expansion mode:

```text
--support_expansion_mode target_footprint_residual_debt
```

It ranks extra support faces by:

```text
score(face, bin)
  = train_residual_debt(face, bin)
    * log1p(target_pixels(face, bin))
    * target_view_fraction(face, bin)
```

The train residual debt is computed only from fit/train evidence views, skipping policy-val views with `policy_val_stride`:

```text
debt
  = mean_l1
    * log1p(train_samples)
    * sqrt(train_view_count)
    * sign_consistency
    / (sqrt(l1_variance) + 1e-3)
```

The target footprint uses only GT-free target evidence:

- `face_id`
- `barycentric`
- optional `barycentric_valid`
- optional `alpha`

It does **not** read target RGB or held-out GT. This keeps the method target-trajectory-aware without test-label leakage.

The selected faces then pass through the existing pipeline unchanged:

```text
target-debt support faces
  -> fit atlas
  -> face-alpha calibration
  -> local-patch prior candidates
  -> policy-val prior-bin hybrid
  -> target-footprint tail-risk certificate
  -> image SSIM/L1 gates
  -> held-out metric evaluation
```

## Code Changes

Updated files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_l1risk_fairnoop_scene.py`

New function:

- `rank_target_footprint_residual_debt_faces(...)`

CLI changes:

- adapter: `--support_expansion_mode {none,fit_residual_topk,target_footprint_residual_debt}`
- runner: `--support_expansion_mode {none,fit_residual_topk,target_footprint_residual_debt}`

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

Both entrypoints expose the new mode in `--help`.

## Running Command

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_v91_target_debt_support WANDB_MODE=online CUDA_VISIBLE_DEVICES=3 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v91_target_debt_support_20260625 \
  --tag v91_target_debt_support_counter_region_texture_adapter \
  --support_expansion_mode target_footprint_residual_debt \
  --support_expansion_max_extra_faces 4096 \
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
  --wandb_group v91_target_debt_support \
  --wandb_run_name v91_target_debt_support_counter_20260625 \
  --wandb_mode online \
  --force
```

## Promotion Rule

v91 should only be promoted if it strictly beats the counter anchor:

```text
anchor = 26.756137847900 PSNR / 0.862126350403 SSIM / 0.251690655947 LPIPS
```

Additional requirements:

- `accepted_atlas`
- `target_changed_fraction >= 0.001`
- no image L1/SSIM gate regression relative to the accepted candidate
- audit confirms `support_expansion.mode = target_footprint_residual_debt`

If counter promotes, run hard-triad with the same fixed policy on `counter,kitchen,bonsai`, then full9. If it fails, archive as a negative diagnostic and inspect whether target-debt selected different faces, whether target coverage rose, and whether policy-val gates rejected the additional support.
