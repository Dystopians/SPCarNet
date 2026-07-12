# v57a Face-Alpha Reliability Shrink Probe

Date: 2026-06-24

Status: `REAL_PIPELINE_CHANGE_PROBED_NOT_PROMOTED`.

## Motivation

v55d proved that policy-val per-face alpha can find a real positive signal on `counter`, but it also exposed two failure modes:

- `kitchen` improves PSNR/LPIPS but regresses SSIM and needs a high selected global alpha;
- `bonsai` regresses all three held-out metrics;
- boundary scenes such as `garden/room` can be internally accepted while still lacking worst-view SSIM margin.

v57a tests whether the per-face alpha profile should be reliability-shrunk by train policy-val evidence before target application.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New adapter/runner CLI:

```text
--face_alpha_calibration_shrink_count_tau
--face_alpha_calibration_shrink_denominator_tau
--face_alpha_calibration_shrink_prior {fallback,zero}
```

Default values preserve legacy behavior:

```text
count_tau = 0.0
denominator_tau = 0.0
prior = fallback
```

When enabled, each fitted face alpha is shrunk as:

```text
reliability =
  count / (count + count_tau)
  * denominator / (denominator + denominator_tau)

alpha_final = prior_alpha + reliability * (alpha_ls_clipped - prior_alpha)
```

This is a train-policy-val reliability prior. It does not read held-out test GT for alpha fitting or branch selection.

## Counter Probe

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 1 \
  --output_root /dev/shm/peilincai_spcarnet_v57a_face_alpha_shrink_counter_20260624 \
  --tag v57a_face_alpha_shrink_count512_den01_zero_support4096_tex32_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --face_alpha_calibration_shrink_count_tau 512 \
  --face_alpha_calibration_shrink_denominator_tau 0.1 \
  --face_alpha_calibration_shrink_prior zero \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v57a_face_alpha_reliability_shrink_probe \
  --wandb_run_name v57a_shrink_counter_20260624 \
  --wandb_mode online \
  --force
```

W&B:

```text
counter parent/per-scene run: fptifheb
```

Output:

```text
/dev/shm/peilincai_spcarnet_v57a_face_alpha_shrink_counter_20260624/counter_v57a_face_alpha_shrink_count512_den01_zero_support4096_tex32_nearest_region_texture_adapter
```

## Result

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v52 counter | `26.7534599304` | `0.8621146679` | `0.2518683374` | baseline fallback |
| raw v55d counter | `26.7561302185` | `0.8621262312` | `0.2516913712` | current v56 selected row |
| v57a shrink counter | `26.7550086975` | `0.8621243238` | `0.2517508566` | positive vs v52, weaker than raw v55d |

Delta:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v57a vs v52 | `+0.0015487671` | `+0.0000096559` | `-0.0001174808` |
| v57a vs raw v55d | `-0.0011215210` | `-0.0000019074` | `+0.0000594854` |

Audit:

```text
accepted: True
effective policy: accepted_atlas
selected alpha: 0.5
changed fraction: 6.536248%
face_alpha_count: 394
fallback_face_count: 5234
policy-val relative gain: 0.034402
policy-val SSIM min-view gain: 0.000051498
policy-val image-L1 positive view fraction: 0.916667
```

## Kitchen Risk Probe

W&B:

```text
kitchen parent/per-scene run: 4zevmx9g
```

Output:

```text
/dev/shm/peilincai_spcarnet_v57a_face_alpha_shrink_kitchen_20260624/kitchen_v57a_face_alpha_shrink_count512_den01_zero_support4096_tex32_nearest_region_texture_adapter
```

Result:

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v52 kitchen | `27.8189353943` | `0.8765353560` | `0.1990194172` | baseline fallback |
| raw v55d kitchen | `27.8234424591` | `0.8764375448` | `0.1987804621` | PSNR/LPIPS up, SSIM down |
| v57a shrink kitchen | `27.8229331970` | `0.8764361739` | `0.1988590807` | PSNR/LPIPS up vs v52, still SSIM down |

Delta:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v57a vs v52 | `+0.0039978027` | `-0.0000991821` | `-0.0001603365` |
| v57a vs raw v55d | `-0.0005092621` | `-0.0000013709` | `+0.0000786185` |

Audit:

```text
accepted: True
effective policy: accepted_atlas
selected alpha: 1.25
changed fraction: 3.936124%
face_alpha_count: 240
fallback_face_count: 4093
policy-val SSIM min-view gain: 0.000202358
policy-val image-L1 positive view fraction: 1.000000
policy-val image-L1 CVaR20 view gain: 0.000028343
```

Interpretation: the shrink setting does not solve the `kitchen` SSIM regression. It keeps the PSNR/LPIPS improvement relative to v52, but the held-out SSIM remains below v52 and the row still requires a high selected global alpha (`1.25`). This means v57a does not close the face-alpha reliability problem.

## Interpretation

The implementation is valid and the probe remains positive versus v52, but this shrink setting is too conservative for `counter`: it suppresses the strongest raw v55d signal and does not improve the current v56 selected row.

Do not promote v57a shrink as-is.

The useful lesson is narrower:

- reliability shrink is a real train/eval pipeline mechanism and now has a clean CLI;
- `counter` wants little or no shrink under the current atlas;
- `kitchen` confirms that simple reliability shrink does not repair the SSIM risk;
- the next representation-level method should shift to view-conditioned residual basis rather than more face-alpha shrink variants.
