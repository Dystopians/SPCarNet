# v58a View-Conditioned Surface Residual Basis Probe

Date: 2026-06-24

Status: `REAL_PIPELINE_CHANGE_ACTIVE_PROBE`.

## Motivation

v57a showed that simply shrinking per-face alpha is not a strong enough answer:

- `counter` stayed positive versus v52, but became weaker than raw v55d;
- `kitchen` kept PSNR/LPIPS gains but still regressed SSIM versus v52;
- the failure mode is not just residual amplitude, but view-dependent residual structure.

v58a therefore changes the representation itself. Instead of storing only a mean residual RGB per surface face/UV bin, it can store a small view-conditioned residual basis fitted from train fit views:

```text
residual(face, uv, view) =
  beta0(face, uv)
  + beta1(face, uv) * cx
  + beta2(face, uv) * cy
  + beta3(face, uv) * cz
```

where `[cx, cy, cz]` is the normalized `camera_center` vector in the evidence npz. Unsupported bins fall back to the legacy mean atlas, so the old method remains the default and the new mechanism is explicitly opt-in.

## Implementation

Updated files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New CLI:

```text
--view_conditioned_basis_mode {none,camera_center_linear}
--view_conditioned_basis_min_bin_samples
--view_conditioned_basis_ridge
```

Default behavior is unchanged:

```text
--view_conditioned_basis_mode none
```

When enabled, each face/UV bin with enough train-fit support solves a ridge least-squares system against `[1, normalized_camera_center]`. At target/test time the same target-view `camera_center` predicts the residual. No held-out test GT is used by fitting or branch selection.

## Running Probes

### Kitchen

Purpose: test whether view-conditioned residual structure can fix the `kitchen` SSIM risk that v57a did not solve.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene kitchen \
  --gpu 2 \
  --output_root /dev/shm/peilincai_spcarnet_v58a_viewbasis_kitchen_20260624 \
  --tag v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode camera_center_linear \
  --view_conditioned_basis_min_bin_samples 64 \
  --view_conditioned_basis_ridge 0.01 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v58_view_conditioned_surface_residual_basis_probe \
  --wandb_run_name v58a_viewbasis_kitchen_20260624 \
  --wandb_mode online \
  --force
```

Output:

```text
/dev/shm/peilincai_spcarnet_v58a_viewbasis_kitchen_20260624/kitchen_v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter
```

### Counter

Purpose: check whether the new basis preserves the positive `counter` residual signal.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 3 \
  --output_root /dev/shm/peilincai_spcarnet_v58a_viewbasis_counter_20260624 \
  --tag v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter \
  --v48_roots outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware,/dev/shm/peilincai_spcarnet_v48_full9_20260623 \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --view_conditioned_basis_mode camera_center_linear \
  --view_conditioned_basis_min_bin_samples 64 \
  --view_conditioned_basis_ridge 0.01 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.0 \
  --wandb_project SPCarNet \
  --wandb_group v58_view_conditioned_surface_residual_basis_probe \
  --wandb_run_name v58a_viewbasis_counter_20260624 \
  --wandb_mode online \
  --force
```

Output:

```text
/dev/shm/peilincai_spcarnet_v58a_viewbasis_counter_20260624/counter_v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter
```

## Decision Rule

Do not promote v58a unless it gives a real held-out improvement over v52/v56 on at least one diagnostic scene without creating a new SSIM or LPIPS regression. A useful first milestone would be:

- `kitchen`: PSNR/LPIPS improve versus v52 and SSIM no longer regresses, or the policy safely rejects to no-op;
- `counter`: remains non-regressive versus v56/v55d, or rejects safely.

If v58a fails both scenes, the lesson is still useful: camera-center linear basis is not enough, and the next representation-level step must use a stronger view feature or a more local support certification.

## v58a Min64 Result

The first probe completed but did not activate the new basis because the support threshold was too conservative.

W&B:

```text
counter: 8ng4dnih
kitchen: ezsrdzbx
```

| scene | PSNR | SSIM | LPIPS | selected alpha | changed fraction | supported bins | supported-bin fraction | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | `26.7534599304` | `0.8621146679` | `0.2518683374` | `0.125` | `6.536248%` | `0` | `0.000000` | degenerates to v52; not promoted |
| kitchen | `27.8189353943` | `0.8765353560` | `0.1990194172` | `0.125` | `3.936124%` | `0` | `0.000000` | degenerates to v52; not promoted |

Interpretation:

- the implementation path is executable and the audit records the new basis fields;
- `min_bin_samples=64` is unusable for `texture_size=32` on these support sets;
- the unchanged metrics are expected because no face/UV bin passed the basis support threshold.

## v58b Min4 Follow-Up

To test the actual mechanism, a follow-up probe lowers the basis support threshold:

```text
--view_conditioned_basis_min_bin_samples 4
--view_conditioned_basis_ridge 0.01
```

Running outputs:

```text
/dev/shm/peilincai_spcarnet_v58b_viewbasis_counter_20260624/counter_v58b_viewbasis_camcenter_min4_ridge001_support4096_tex32_nearest_region_texture_adapter
/dev/shm/peilincai_spcarnet_v58b_viewbasis_kitchen_20260624/kitchen_v58b_viewbasis_camcenter_min4_ridge001_support4096_tex32_nearest_region_texture_adapter
```

W&B run names:

```text
v58b_viewbasis_min4_counter_20260624
v58b_viewbasis_min4_kitchen_20260624
```

Completed W&B:

```text
counter: 1wvclw9g
kitchen: 7puzt1qa
```

| scene | PSNR | SSIM | LPIPS | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | supported bins | supported-bin fraction | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| counter | `26.7529773712` | `0.8620327711` | `0.2517679930` | `-0.000483` | `-0.00008190` | `-0.00010034` | `116171` | `0.020009` | LPIPS improves but PSNR/SSIM regress; not promoted |
| kitchen | `27.8192138672` | `0.8765175939` | `0.1989833564` | `+0.000278` | `-0.00001776` | `-0.00003606` | `114273` | `0.025695` | PSNR/LPIPS improve but SSIM still below v52; not promoted |

Interpretation:

- lowering the threshold activates the basis and gives measurable held-out changes;
- coverage is still only about 2-2.6% of atlas bins at `texture_size=32`;
- train policy-val predicts positive SSIM/L1, but held-out SSIM still regresses on both diagnostic scenes;
- camera-center linear basis at this capacity is not reliable enough to promote.

## v58c Texture16 Follow-Up

Because v58b's active support is sparse at `texture_size=32`, v58c tests the same basis at `texture_size=16` to increase per-bin support density.

Running outputs:

```text
/dev/shm/peilincai_spcarnet_v58c_viewbasis_tex16_counter_20260624/counter_v58c_viewbasis_camcenter_min4_ridge001_support4096_tex16_nearest_region_texture_adapter
/dev/shm/peilincai_spcarnet_v58c_viewbasis_tex16_kitchen_20260624/kitchen_v58c_viewbasis_camcenter_min4_ridge001_support4096_tex16_nearest_region_texture_adapter
```

W&B:

```text
counter: e76hlmtb
kitchen: ig7k3vtp
```

Completed results:

| scene | PSNR | SSIM | LPIPS | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | supported bins | supported-bin fraction | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| counter | `26.7527809143` | `0.8619601130` | `0.2517301738` | `-0.000679` | `-0.00015455` | `-0.00013816` | `167172` | `0.115170` | coverage improves, but PSNR/SSIM regress; not promoted |
| kitchen | `27.8194313049` | `0.8764430285` | `0.1990009248` | `+0.000496` | `-0.00009233` | `-0.00001849` | `170037` | `0.152937` | PSNR/LPIPS slightly improve vs v52, but SSIM still regresses; not promoted |

Additional comparison:

| scene | comparison | dPSNR | dSSIM | dLPIPS |
|---|---|---:|---:|---:|
| counter | v58c vs v56 | `-0.003349` | `-0.00016612` | `+0.00003880` |
| counter | v58c vs v57a | `-0.002228` | `-0.00016421` | `-0.00002068` |
| kitchen | v58c vs raw v55d | `-0.004011` | `+0.00000548` | `+0.00022046` |
| kitchen | v58c vs v57a | `-0.003502` | `+0.00000685` | `+0.00014184` |

Interpretation:

- reducing `texture_size` from `32` to `16` did increase the active basis support from about `2.0%/2.6%` to `11.5%/15.3%`;
- higher basis coverage did not translate into reliable held-out metric gains;
- `counter` becomes worse than v52/v56 on PSNR and SSIM;
- `kitchen` keeps a tiny PSNR/LPIPS gain over v52, but does not fix the SSIM regression and is weaker than raw v55d/v57a on PSNR/LPIPS;
- therefore the camera-center linear residual basis is a useful implemented diagnostic, but it is not a promotable method in its current form.

Final v58 decision:

```text
DO_NOT_PROMOTE_V58_CAMERA_CENTER_LINEAR_BASIS_AS_CURRENT_ENDPOINT
```

The method change is real and fully wired into the train/eval pipeline, but the evidence says the bottleneck is not just per-bin support density. A paper-level next step should use stronger surface/view features and a better uncertainty guard, e.g. surface normal/view-angle features, per-region uncertainty calibration, or a no-regression support-fraction guard that falls back before held-out SSIM damage.
