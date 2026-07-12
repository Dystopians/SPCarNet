# v72 Local Prior Allowlist Negative Log

Date: 2026-06-24

Status: `COMPLETED_NOT_PROMOTED_NEGATIVE`

Persistent artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v72_local_prior_allowlist_20260624
```

## Motivation

v70 and v71a showed that global nonzero count-pyramid prior blending is not selected by the train-only policy on `counter/kitchen`. v72 asks a narrower question: can a nonzero prior become useful if only policy-val-positive local bins are allowed to use it?

This tests a scientific hypothesis rather than a new scalar sweep:

```text
same-face coarse prior is not globally safe
  -> restrict it to bins with local policy-val gain and low uncertainty
  -> check whether the resulting target edit improves held-out metrics
```

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v72_local_prior_allowlist_20260624 \
  --tag v72_localprior_binuncertainty_countpyramid_blend1_support4096_tex16_nearest_region_texture_adapter \
  --force \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode count_pyramid \
  --surface_multiscale_prior_block_sizes 2,4,6 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 1.0 \
  --surface_multiscale_prior_gate_mode evidence_consistent \
  --surface_multiscale_prior_min_prior_weight 0.05 \
  --surface_multiscale_prior_min_direct_samples 1 \
  --surface_multiscale_prior_min_sign_consistency 0.5 \
  --surface_multiscale_prior_max_mean_variance 0.004 \
  --surface_multiscale_prior_min_cosine 0.0 \
  --enable_policy_val_bin_uncertainty_guard \
  --bin_uncertainty_guard_min_bin_samples 8 \
  --bin_uncertainty_guard_min_relative_gain 0.0 \
  --bin_uncertainty_guard_min_positive_view_fraction 0.5 \
  --bin_uncertainty_guard_max_mean_variance 0.004 \
  --bin_uncertainty_guard_min_mean_sign_consistency 0.5 \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --view_conditioned_basis_min_bin_samples 16 \
  --view_conditioned_basis_ridge 0.1 \
  --view_conditioned_basis_ood_mode diag_z \
  --view_conditioned_basis_ood_max_z 2.5 \
  --view_conditioned_basis_ood_min_std 0.05 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v72_local_prior_allowlist \
  --wandb_run_name v72_localprior_counter_binuncertainty_blend1_20260624 \
  --wandb_mode online
```

W&B run:

```text
fnc0ktxk
```

## Result

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v72 local prior allowlist | `26.750389` | `0.862056` | `0.251968` |
| v70/v71a counter | `26.753996` | `0.862119` | `0.251853` |
| selected v64/v56 reference | `26.756130` | `0.862126` | `0.251691` |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v72 vs v70/v71a | `-0.003607` | `-0.000063658` | `+0.000114977` |
| v72 vs selected reference | `-0.005741` | `-0.000070632` | `+0.000276655` |

## Audit Evidence

The run was not a no-op:

```text
accepted: true
effective_policy: accepted_atlas
selected_alpha: 0.125
selected_surface_multiscale_prior_blend: 1.0
target_changed_fraction: 0.003807394
```

The prior and guard were active:

```text
prior blended bins: 201339
prior blended bin fraction: 0.138709077
prior gate rejected bins: 692569
bin guard candidate bins: 250224
bin guard allowed bins: 4791
bin guard rejected bins: 245433
bin guard allowed faces: 741
bin guard allowed sample fraction: 0.090951269
```

Policy-val evidence was positive after the guard:

```text
relative gain: 0.005462999
SSIM gain: 0.000056560
image-L1 gain: 0.000005347
positive view fractions: 1.0 / 1.0 / 1.0
```

## Conclusion

v72 strengthens the negative diagnosis. A train-policy-val bin allowlist can make a nonzero prior candidate locally defensible, but the resulting target edit is too small and still regresses held-out RGB metrics. Continuing to tune `count_pyramid` blend and uncertainty thresholds is therefore unlikely to create the needed representation-level breakthrough.

The next method should add target-support-certified candidate selection:

- compute target-visible support for every policy-val-safe candidate before final selection;
- rank survivors by target changed fraction, per-view changed fraction, CVaR target support, and valid support fraction;
- keep train-policy nonregression as the hard safety gate;
- avoid using target GT for selection.

This is the clean next step because v72 proves policy-val positivity alone is not enough; candidates also need enough target-view action footprint.
