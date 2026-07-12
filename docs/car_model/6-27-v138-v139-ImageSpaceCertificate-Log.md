# 2026-06-27 v138/v139 Image-Space Certificate Log

This log records the first train-only image-space certificate upgrade after the
v137 view-cluster/adaptive-activity line. It is a real pipeline change, but it
has not yet produced an accepted target method on `flowers`.

## Motivation

v134b-v137 showed that residual-MSE bin shrink was too sparse and repeatedly
collapsed to no-op. The next hypothesis was that the shrink policy was
optimizing the wrong local proxy: a bin can be useful only if its predicted
delta improves the actual policy-val image, not merely if residual-MSE
statistics look favorable.

The v138/v139 change therefore adds an image-space certificate computed only
from policy-val views:

- build the candidate residual texture from fit/train evidence;
- render policy-val views with the candidate delta;
- compare image L1 before/after against policy-val GT;
- aggregate the before/after L1 per `(cluster, face, uv-bin)`;
- allow a bin to shrink only when image-space evidence is positive;
- keep target/test GT invisible to selection and target application.

## Implementation

Changed files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New adapter controls:

- `--enable_policy_val_image_l1_bin_certificate`
- `--image_l1_bin_certificate_mode {and,or,replace}`
- `--image_l1_bin_certificate_min_relative_gain`
- `--image_l1_bin_certificate_min_positive_view_fraction`
- `--image_l1_bin_certificate_gain_tau`
- `--image_l1_bin_certificate_pool_radius`

The certificate audit records:

- `uses_policy_val_gt=true`
- `uses_target_or_test_gt=false`
- number of policy-val views used/missing image L1;
- candidate/selected image-space evidence counts;
- selected relative gain and positive-view fraction;
- top candidate bins for diagnosis.

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Both checks passed after the v138/v139 edits.

## v138 Exact-Bin Image-L1 Certificate

Run root:

`/dev/shm/peilincai_spcarnet_v138_image_l1_bin_cert_flowers_20260627_025853/flowers`

W&B offline run:

`/dev/shm/peilincai_wandb_v138_image_l1_bin_cert_flowers_20260627_025853/wandb/offline-run-20260627_032117-ajl4uwh1`

Protocol:

- strict no target/test GT for selection and target apply;
- policy-val image L1 certificate enabled;
- exact `(cluster, face, bin)` evidence, no patch pooling;
- train-only adaptive residual activity threshold selected `min_l1=0.01174163818359375`.

Result:

- `accepted=false`
- `effective_policy=fallback_noop`
- `changed_pixels=0 / 37100800`
- `candidate_bin_count=6626`
- `candidate_evidence_ok_count=0`
- `bin_uncertainty_shrink_count=0`
- `policy_val_views_with_image_l1=12`
- `policy_val_views_missing_image_l1=0`
- test metrics remained fallback/no-op:
  - PSNR `20.452775955`
  - SSIM `0.549059212`
  - LPIPS `0.355544209`

Reject reason:

`positive_view_fraction`, SSIM, image-L1, and LPIPS positive-view fractions
were all `0.0`; effective relative gain was `0.0 < 0.001`.

Interpretation:

Exact bin-level image-space evidence is too sparse on this outdoor scene. The
certificate is safe, but it cannot find any sufficiently supported bin.

## v139 Patch-Pooled Image-L1 Certificate

Run root:

`/dev/shm/peilincai_spcarnet_v139_patch_l1_bin_cert_flowers_20260627_032344/flowers`

W&B offline run:

`/dev/shm/peilincai_wandb_v139_patch_l1_bin_cert_flowers_20260627_032344/wandb/offline-run-20260627_035121-lj03absd`

Protocol:

- strict no target/test GT for selection and target apply;
- policy-val image L1 certificate enabled in `replace` mode;
- patch-pooled certificate radius `1`;
- view-cluster local shrink enabled;
- train-only adaptive residual activity threshold selected `min_l1=0.01174163818359375`.

Result:

- `accepted=false`
- `effective_policy=fallback_noop`
- `changed_pixels=0 / 37100800`
- `candidate_bin_count=6626`
- `candidate_evidence_ok_count=21`
- `selected_evidence_ok_count=20`
- `bin_uncertainty_shrink_count=20`
- `selected_face_count=3`
- `view_cluster_selected_cluster_count=1`
- selected-bin mean image-L1 relative gain `0.038319989`
- selected-bin mean image-L1 positive-view fraction `1.0`
- policy-val best alpha `0.5`
- policy-val global relative gain `0.0000165096`
- policy-val positive-view fraction `0.1666667`
- policy-val image-L1 gain `3.1044e-10`
- policy-val image-L1 positive-view fraction `0.0833333`
- policy-val LPIPS gain `-8.8165e-08`
- test metrics remained fallback/no-op:
  - PSNR `20.452775955`
  - SSIM `0.549059212`
  - LPIPS `0.355544209`

Reject reason:

- `cvar20_view_relative_gain=-0.0000011927 < 0`
- `min_view_relative_gain=-0.000003578 < -0.000001`
- `lpips_gain=-0.000000088 < 0`
- `effective_relative_gain=0.000016510 < 0.001`

Interpretation:

Patch pooling is a meaningful diagnostic step: it converts the exact-bin
certificate from zero evidence into 20 selected shrink bins with positive local
image-L1 evidence. However, coverage is still far too small. The full-image
gain is two orders of magnitude below the acceptance threshold, and the
candidate slightly hurts LPIPS. The safe gate correctly rejects it and falls
back to no-op.

## Current Conclusion

The new prompt direction has produced a better diagnostic mechanism, not yet a
successful method. It proves that policy-val image-space evidence can identify
small locally helpful regions, but it also proves that the current residual
texture/shrink representation does not cover enough pixels or views to improve
the full target render.

The next method should not continue as manual threshold tuning. The evidence
points to a representation/coverage problem:

- exact bin support is too sparse;
- patch-pooled support is still localized to only 20 bins and 3 faces;
- global PSNR/SSIM/LPIPS gates remain non-positive or far below threshold;
- target application remains fallback/no-op.

## Next Required Upgrade

The next version should promote from sparse bin shrink to a train-only adaptive
region support policy:

1. select larger connected face/bin regions from policy-val image-L1 evidence,
   not isolated bins;
2. choose the pooling/region scale from train/policy-val support statistics,
   not scene-specific manual parameters;
3. cap deltas with the same final `max_abs_delta_rgb` used at application;
4. require non-regression across PSNR/SSIM/L1/LPIPS before target write;
5. audit every chosen region with support count, view count, positive fraction,
   and image-space gain.

Until that upgrade produces accepted non-noop target renders on multiple
scenes, this line remains incomplete for a paper-level claim.
