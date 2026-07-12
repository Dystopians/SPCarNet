# v69 Count-Pyramid Multi-Scale Surface Prior Probe Log

Date: 2026-06-24

Status: `NOT_PROMOTED_DIAGNOSTIC`

## Motivation

v65-v68 showed that the remaining representation-level bottleneck is not mostly scalar alpha tuning. The surface residual atlas still suffers from sparse support: many face/UV bins have too few direct residual samples, so the atlas either under-applies useful residuals or becomes locally noisy.

v69 therefore adds a real support/coverage-oriented method change:

```text
direct face/bin residual
  -> keep high-support bins
  -> blend low-support bins with same-face coarse residual block priors
```

The goal is to give low-support bins a multi-scale same-surface prior without reading held-out test GT.

## Implementation

Primary files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner flags:

```text
--surface_multiscale_prior_mode {none,count_pyramid}
--surface_multiscale_prior_block_sizes 2,4,8
--surface_multiscale_prior_min_bin_samples 8
--surface_multiscale_prior_count_tau 32.0
--surface_multiscale_prior_blend 1.0
```

Core logic:

- build coarse residual block means from `sum_grid / counts` on the same face;
- select a per-bin coarse prior from candidate block sizes;
- identify low-support bins with `count < min_bin_samples`;
- blend only low-support bins;
- keep direct high-support residual estimates unchanged;
- write audit fields under `fit_summary.surface_multiscale_prior`;
- forward all config and summary fields into the W&B runner.

Static validation:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

CLI help for both adapter and runner exposes the new v69 flags.

## Experiment

Common output root:

```text
/dev/shm/peilincai_spcarnet_v69_multiscale_prior_probe_20260624
```

Persistent artifact root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v69_multiscale_prior_probe_20260624
```

Shared probe command pattern:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <counter|kitchen> \
  --gpu <2|3> \
  --output_root /dev/shm/peilincai_spcarnet_v69_multiscale_prior_probe_20260624 \
  --tag v69_countpyramid_b246_min8_tau32_blend1_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode count_pyramid \
  --surface_multiscale_prior_block_sizes 2,4,6 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32 \
  --surface_multiscale_prior_blend 1.0 \
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
  --wandb_group v69_multiscale_prior_probe \
  --wandb_run_name v69_multiscale_<scene>_20260624 \
  --wandb_mode online \
  --force
```

W&B:

| scene | run id | result |
|---|---|---|
| counter | `zez8vl50` | completed |
| kitchen | `6jdm5tc1` | completed |

## Results

| scene | reference | v68 | v69 | delta vs reference | verdict |
|---|---:|---:|---:|---:|---|
| counter | `26.756130 / 0.862126 / 0.251691` | `26.753967 / 0.862119 / 0.251854` | `26.751703 / 0.862084 / 0.251951` | `-0.004427 / -0.0000419 / +0.000260` | reject |
| kitchen | `27.822626 / 0.876538 / 0.198849` | `27.819143 / 0.876533 / 0.199032` | `27.819000 / 0.876532 / 0.199036` | `-0.003626 / -0.00000634 / +0.000187` | reject |

Audit:

| scene | accepted | alpha | changed fraction | low-support bins | blended bins | blended-bin fraction | mean blend |
|---|---:|---:|---:|---:|---:|---:|---:|
| counter | true | 0.0625 | 0.065630 | 1390496 | 893908 | 0.615843 | 0.860981 |
| kitchen | true | 0.1250 | 0.039585 | 1041866 | 665982 | 0.599008 | 0.846014 |

Policy-val gate stayed positive on both scenes, but held-out test was worse than the selected references. This gap is the key diagnosis: the count-pyramid prior changes many bins and looks safe under policy-val, but the current blending strength smooths or biases residuals enough to lose held-out detail.

## Decision

Do not promote v69.

What we learned:

- The new interface is useful and correctly wired into train/eval/W&B/audits.
- The method attacks the right bottleneck: low-support residual coverage.
- The first prior is too aggressive: around 60% of atlas bins are blended with mean blend weight around 0.85.
- The next version should make blending confidence sharper and more local, likely with lower blend, smaller prior radius, or a gate that only activates when the coarse prior has policy-val support rather than count support alone.

Current reportable endpoint remains Phase-J. Current best fixed representation-level policy remains v64.

