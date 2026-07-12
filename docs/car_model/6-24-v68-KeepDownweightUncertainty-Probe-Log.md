# v68 Keep-With-Downweight Uncertainty Shrink Probe Log

Date: 2026-06-24

Purpose: fix the v67 uncertainty-shrink semantics and test whether an uncertainty-aware local residual policy can keep normal atlas residual transfer while downweighting only risky face/UV bins.

---

## Summary

v68 is a real train/eval pipeline change. It extends the v67 `policy_val_bin_uncertainty_shrink` mode with an explicit policy mode:

```text
--bin_uncertainty_shrink_policy_mode {sparse_positive,keep_with_downweight}
```

The important semantic change is:

| mode | meaning | expected use |
|---|---|---|
| `sparse_positive` | store only bins with positive policy-val evidence; unknown bins fall back to the configured fallback shrink | v67 behavior |
| `keep_with_downweight` | keep fallback residual strength for unknown bins and write explicit local downweights only for risky observed bins | v68 behavior |

Result: v68 is not promoted. It is better than v67 on `counter/kitchen`, which proves the semantic fix was useful, but it still remains slightly below the current selected v56/v64 references.

| scene | reference | v68 PSNR | v68 SSIM | v68 LPIPS | verdict |
|---|---:|---:|---:|---:|---|
| counter | v56/v64: `26.756130 / 0.862126 / 0.251691` | 26.753967 | 0.862119 | 0.251854 | improves v67, still below selected reference |
| kitchen | v64: `27.822626 / 0.876538 / 0.198849` | 27.819143 | 0.876533 | 0.199032 | improves v67, still below selected reference |

Comparison against v67:

| scene | dPSNR vs v67 | dSSIM vs v67 | dLPIPS vs v67 |
|---|---:|---:|---:|
| counter | +0.004114 | +0.000069 | -0.000144 |
| kitchen | +0.002754 | +0.000090 | -0.000169 |

Interpretation:

> v68 confirms that the v67 failure was partly an overly sparse residual-transfer semantics problem. However, the corrected keep/downweight policy still cannot beat the current best fixed representation-level policy, so the promoted method remains Phase-J for presentation and v64 for representation-level reporting.

---

## Implementation

Changed files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner flag:

```text
--bin_uncertainty_shrink_policy_mode {sparse_positive,keep_with_downweight}
```

For `keep_with_downweight`, v68 estimates a risk confidence per observed face/bin:

```text
risk confidence =
  count confidence
  * max(
      positive-view deficit,
      negative-gain confidence,
      variance penalty,
      sign-inconsistency penalty
    )
```

Then it shrinks only the risky bins away from the fallback residual strength:

```text
shrink = fallback_shrink - risk_confidence * (fallback_shrink - min_shrink)
```

The probe used:

```text
fallback_shrink = 1.0
min_shrink = 0.0
max_profile_bins = 16384
min_positive_view_fraction = 0.75
```

The audit now records:

```text
uncertainty_shrink_policy_mode
downweighted_bin_count
upweighted_bin_count
mean_profile_abs_delta_from_fallback
```

The runner also logs these fields to W&B:

```text
policy/bin_uncertainty_shrink_policy_keep_downweight
policy/bin_uncertainty_shrink_downweighted_count
policy/bin_uncertainty_shrink_upweighted_count
```

---

## Validation

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

CLI exposure was checked on both adapter and runner:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help | rg 'bin_uncertainty_shrink_policy_mode'

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py --help | rg 'bin_uncertainty_shrink_policy_mode'
```

---

## Probe Commands

Output root:

```text
/dev/shm/peilincai_spcarnet_v68_keepdown_probe_20260624
```

Persistent copy:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624
```

W&B:

| scene | GPU | W&B run | status |
|---|---:|---|---|
| counter | 2 | `34xekhu1` | completed |
| kitchen | 3 | `2892evrb` | completed |

Command template:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <SCENE> \
  --gpu <GPU> \
  --output_root /dev/shm/peilincai_spcarnet_v68_keepdown_probe_20260624 \
  --tag v68_keepdown_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --enable_policy_val_bin_uncertainty_shrink \
  --bin_uncertainty_shrink_policy_mode keep_with_downweight \
  --bin_uncertainty_shrink_min_bin_samples 32 \
  --bin_uncertainty_shrink_min_relative_gain 0.0 \
  --bin_uncertainty_shrink_min_positive_view_fraction 0.75 \
  --bin_uncertainty_shrink_max_mean_variance -1 \
  --bin_uncertainty_shrink_min_mean_sign_consistency 0.0 \
  --bin_uncertainty_shrink_count_tau 128 \
  --bin_uncertainty_shrink_gain_tau 0.005 \
  --bin_uncertainty_shrink_variance_scale 0.004 \
  --bin_uncertainty_shrink_sign_power 0.5 \
  --bin_uncertainty_shrink_min_shrink 0.0 \
  --bin_uncertainty_shrink_max_shrink 1.0 \
  --bin_uncertainty_shrink_fallback_shrink 1.0 \
  --bin_uncertainty_shrink_max_profile_bins 16384 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v68_keepdown_probe \
  --wandb_run_name v68_keepdown_<SCENE>_20260624 \
  --wandb_mode online \
  --force
```

---

## Results

### Counter

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/logs/apply_metrics_counter.log
```

| field | value |
|---|---:|
| PSNR | 26.753967 |
| SSIM | 0.862119 |
| LPIPS | 0.251854 |
| accepted | true |
| selected alpha | 0.125 |
| changed fraction | 0.065630 |
| shrink policy mode | `keep_with_downweight` |
| candidate bins | 250224 |
| stored shrink bins | 16384 |
| fallback bins | 233840 |
| downweighted bins | 16384 |
| upweighted bins | 0 |
| mean selected shrink | 0.957678 |
| min/max selected shrink | 0.646176 / 0.994129 |
| mean abs delta from fallback | 0.042322 |

Policy-val gate:

| field | value |
|---|---:|
| selected SSIM gain | 0.000197659 |
| selected SSIM positive fraction | 1.000000 |
| selected image-L1 gain | 0.000017172 |
| selected image-L1 positive fraction | 1.000000 |
| selected image-L1 min-view gain | 0.000000121 |
| selected image-L1 CVaR20 gain | 0.000002678 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64 selected | 26.756130 | 0.862126 | 0.251691 |
| v67 sparse positive | 26.749853 | 0.862050 | 0.251998 |
| v68 keep/downweight | 26.753967 | 0.862119 | 0.251854 |

### Kitchen

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/kitchen/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/kitchen/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/kitchen/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/logs/apply_metrics_kitchen.log
```

| field | value |
|---|---:|
| PSNR | 27.819143 |
| SSIM | 0.876533 |
| LPIPS | 0.199032 |
| accepted | true |
| selected alpha | 0.125 |
| changed fraction | 0.039585 |
| shrink policy mode | `keep_with_downweight` |
| candidate bins | 217409 |
| stored shrink bins | 16384 |
| fallback bins | 201025 |
| downweighted bins | 16384 |
| upweighted bins | 0 |
| mean selected shrink | 0.969303 |
| min/max selected shrink | 0.751766 / 0.995005 |
| mean abs delta from fallback | 0.030697 |

Policy-val gate:

| field | value |
|---|---:|
| selected SSIM gain | 0.000179042 |
| selected SSIM positive fraction | 1.000000 |
| selected image-L1 gain | 0.000019700 |
| selected image-L1 positive fraction | 1.000000 |
| selected image-L1 min-view gain | 0.000007510 |
| selected image-L1 CVaR20 gain | 0.000010682 |

Reference comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v64 selected | 27.822626 | 0.876538 | 0.198849 |
| v67 sparse positive | 27.816389 | 0.876443 | 0.199201 |
| v68 keep/downweight | 27.819143 | 0.876533 | 0.199032 |

---

## Decision

v68 is a valid implementation and a useful diagnostic, but it is not promoted.

What it fixed:

- v67's sparse-positive semantics under-used residual evidence.
- keep-with-downweight makes the target changed fraction meaningful again.
- W&B and audit fields now expose whether shrink is actually acting as local downweighting.

What it did not fix:

- The best selected references are still slightly stronger on both probe scenes.
- Per-bin downweighting alone cannot recover the larger Phase-J render-time gain.
- The bottleneck remains representation capacity and target-view support, not just uncertainty attenuation.

Next research implication:

> The next representation-level step should not be another scalar/RGB/shrink calibration. It should increase the residual field's expressive support while keeping the train-only risk gate, for example through compact surface-conditioned bases or multi-scale residual support with explicit out-of-trajectory protection.

