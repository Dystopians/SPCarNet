# v70 Policy-Val Blend-Ladder Multi-Scale Prior Probe Log

Date: 2026-06-24

Status: `NOT_PROMOTED_SAFE_AUTO_FALLBACK`

## Motivation

v69 showed that count-pyramid multi-scale surface residual priors attack a real bottleneck, but a fixed strong blend is too aggressive: around 60% of atlas bins were blended, and held-out metrics dropped versus the v64/v68 references.

v70 converts that fixed prior into a train-only policy decision:

```text
candidate blend ladder: 0, 0.125, 0.25, 0.5
  -> fit each candidate from train evidence
  -> score each with policy-val gates
  -> accept nonzero prior only if it is non-regressive versus zero-blend anchor
  -> otherwise safely fall back to blend=0
```

This is a real train/eval pipeline change. It removes the manual prior-strength choice and makes the prior a guarded option inside the same fixed policy.

## Implementation

Primary files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New CLI:

```text
--surface_multiscale_prior_blend_candidates 0,0.125,0.25,0.5
```

Adapter behavior:

- parses a candidate blend ladder in `[0,1]`;
- keeps legacy fixed behavior when the candidate list is omitted;
- disables the ladder when `--surface_multiscale_prior_mode none`;
- extends policy candidates with `surface_multiscale_prior_blend`;
- uses the zero-blend accepted candidate as an anchor when available;
- requires nonzero blend candidates to be non-regressive versus the anchor across train-only relative gain, SSIM gain, image-L1 gain, min-view, and CVaR gates;
- records requested, candidate, and selected blend values in the audit and report.

Runner behavior:

- forwards the candidate ladder;
- logs selected blend and candidate count to W&B;
- keeps the same train/test separation as previous fixed-policy probes.

Static validation:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

Both adapter and runner help expose `--surface_multiscale_prior_blend_candidates`.

## Experiment

Common output root:

```text
/dev/shm/peilincai_spcarnet_v70_blendladder_probe_20260624
```

Persistent artifact root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v70_blendladder_probe_20260624
```

Shared command pattern:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <counter|kitchen> \
  --gpu <2|3> \
  --output_root /dev/shm/peilincai_spcarnet_v70_blendladder_probe_20260624 \
  --tag v70_countpyramid_blendladder_0_0125_025_05_support4096_tex16_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --texture_size_candidates 16 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode count_pyramid \
  --surface_multiscale_prior_block_sizes 2,4,6 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32 \
  --surface_multiscale_prior_blend 0.5 \
  --surface_multiscale_prior_blend_candidates 0,0.125,0.25,0.5 \
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
  --wandb_group v70_blendladder_probe \
  --wandb_run_name v70_blendladder_<scene>_20260624 \
  --wandb_mode online \
  --force
```

W&B:

| scene | run id | result |
|---|---|---|
| counter | `y3r060we` | completed |
| kitchen | `9jbpbvcj` | completed |

Runtime note:

- actual policy candidate count was `2 support sets x 4 blend values = 8` per scene;
- each candidate refits the atlas, so this probe took about one hour;
- this cost is acceptable as a diagnostic, but future versions should cache blend-independent atlas statistics.

## Results

| scene | reference | v68 | v69 | v70 | v70 delta vs reference | v70 delta vs v68 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| counter | `26.756130 / 0.862126 / 0.251691` | `26.753967 / 0.862119 / 0.251854` | `26.751703 / 0.862084 / 0.251951` | `26.753996 / 0.862119 / 0.251853` | `-0.002134 / -0.00000674 / +0.000162` | `+0.000029 / -0.00000012 / -0.000001` | safe fallback, not promoted |
| kitchen | `27.822626 / 0.876538 / 0.198849` | `27.819143 / 0.876533 / 0.199032` | `27.819000 / 0.876532 / 0.199036` | `27.819157 / 0.876533 / 0.199031` | `-0.003469 / -0.00000479 / +0.000182` | `+0.000013 / +0.00000006 / -0.000001` | safe fallback, not promoted |

Selection audit:

| scene | accepted | selected alpha | selected blend | blend candidates | selected support | selected fill | accepted candidates | changed fraction |
|---|---:|---:|---:|---|---|---|---:|---:|
| counter | true | `0.125` | `0.0` | `0,0.125,0.25,0.5` | `fit_residual_topk` | `nearest_observed` | `4` | `0.065630` |
| kitchen | true | `0.125` | `0.0` | `0,0.125,0.25,0.5` | `fit_residual_topk` | `nearest_observed` | `4` | `0.039585` |

Train policy-val gates for the selected zero-blend rows:

| scene | positive view fraction | SSIM gain | SSIM positive view fraction | image-L1 gain | image-L1 positive view fraction | image-L1 min-view gain | image-L1 CVaR20 gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| counter | `1.000000` | `0.000196626` | `1.000000` | `0.000017083` | `0.916667` | `-0.000000026` | `0.000002506` |
| kitchen | `1.000000` | `0.000178893` | `1.000000` | `0.000019656` | `1.000000` | `0.000007428` | `0.000010659` |

## Decision

Do not promote v70 as the new best endpoint.

What v70 proves:

- fixed strong multi-scale priors are unsafe, matching the v69 negative result;
- a train-only blend ladder can automatically reject unsafe prior strength;
- the selected safe fallback recovers v68-level performance and avoids the v69 regression;
- the method is cleaner than manual prior-strength tuning because `blend=0` is selected by the policy, not by held-out test inspection.

What v70 does not solve:

- it does not beat v64 on the tested scenes;
- it still has a high search cost because every support/blend candidate refits a full atlas;
- current count-pyramid priors improve coverage but do not yet recover missing high-frequency surface detail.

Current reportable endpoint remains Phase-J. Current best fixed representation-level policy remains v64. v70 should be presented as a safety/policy upgrade and as evidence that the next real improvement needs a stronger prior quality estimator, not another fixed blend value.

