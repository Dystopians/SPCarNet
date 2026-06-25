# v103 Hard-Triad MinCount1 Validation Log

Date: 2026-06-25

Status: hard-triad surface-field evidence. This is stronger than counter-only evidence, but it is still not full9 validation, not unseen-camera generalization, and not final paper completion.

## 0. Verdict

v103 `affine_barycentric` with `min_count=1` now passes the clean-baseline gate on the hard triad:

```text
counter, kitchen, bonsai:
  v103 beats clean on PSNR, SSIM, and LPIPS in every scene.
```

Hard-triad mean:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting reference | `27.821853` | `0.878303` | `0.236894` |
| v103 affine min_count=1 | `28.384418` | `0.879855` | `0.226611` |
| v101/v102a endpoint ceiling | `30.167395` | `0.913355` | `0.163709` |

Mean delta:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v103 minus clean | `+0.562565` | `+0.001552` | `-0.010283` |
| v103 minus v101/v102a | `-1.782977` | `-0.033500` | `+0.062902` |

Interpretation:

```text
v103 is a real surface-field improvement over clean MeshSplatting on the hard triad.
v103 is still not strong enough to replace v101/v102a because it lacks view-conditioned and evidence-gated behavior.
```

## 1. Method Under Test

Method:

```text
v103 face-local affine barycentric residual field
```

Implementation:

```text
render.py
scripts/car_model/build_v103_surface_affine_residual_field.py
```

Field basis:

```text
basis_type = affine_barycentric
basis_order = [1, barycentric_0, barycentric_1]
coefficient_layout = triangle,basis,rgb
```

Fixed configuration:

| item | value |
|---|---|
| endpoint method | `ours_26000_v100_checkpoint_attached_ela_endpoint` |
| renderer scaling | `4` |
| residual dtype | `float16` |
| ridge | `0.0001` |
| min_count | `1` |
| render flag | `--checkpoint_endpoint_require_surface_field` |
| intermediate outputs | disabled |

Important boundary: the fields are distilled from v102 preprojected target-camera deltas. They store no target GT, but the current hard-triad result is same-target-camera field validation, not unseen-camera generalization.

## 2. Scene Results

| scene | clean PSNR | clean SSIM | clean LPIPS | v103 PSNR | v103 SSIM | v103 LPIPS | v101/v102a PSNR | v101/v102a SSIM | v101/v102a LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `26.751774` | `0.862055` | `0.252003` | `27.208200` | `0.863405` | `0.243176` | `28.442907` | `0.893696` | `0.186557` |
| kitchen | `27.818552` | `0.876452` | `0.199186` | `28.310152` | `0.877554` | `0.194518` | `30.197395` | `0.916093` | `0.132004` |
| bonsai | `28.895233` | `0.896400` | `0.259493` | `29.634901` | `0.898607` | `0.242140` | `31.861883` | `0.930276` | `0.172566` |

Per-scene delta versus clean:

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| counter | `+0.456427` | `+0.001350` | `-0.008827` |
| kitchen | `+0.491600` | `+0.001101` | `-0.004668` |
| bonsai | `+0.739668` | `+0.002207` | `-0.017353` |

Per-scene delta versus v101/v102a:

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| counter | `-1.234707` | `-0.030291` | `+0.056619` |
| kitchen | `-1.887243` | `-0.038540` | `+0.062514` |
| bonsai | `-2.226982` | `-0.031669` | `+0.069574` |

## 3. Field And Render Evidence

| scene | valid triangles | triangle count | valid fraction | visible valid fraction | render sec | solve failures |
|---|---:|---:|---:|---:|---:|---:|
| counter | `2,716,465` | `9,644,247` | `0.281667` | `0.999039` | `47.2076` | `0` |
| kitchen | `3,076,126` | `9,512,393` | `0.323381` | `0.999501` | `49.6192` | `0` |
| bonsai | `3,405,864` | `9,555,533` | `0.356428` | `0.999666` | `56.9827` | `0` |

Key manifests:

```text
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_min1_field.manifest.json
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/kitchen/v103_surface_affine_min1_field.manifest.json
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/bonsai/v103_surface_affine_min1_field.manifest.json
```

Key metric files:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/kitchen/detached_model/results.json
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/bonsai/detached_model/results.json
```

Kitchen had one intentionally failed/recovered render attempt caused by passing mutually exclusive surface-field and preprojected-bank modes together. The successful rerun without the preprojected bank completed. This verifies the fail-closed mutual-exclusion guard is active.

## 4. Claim Boundary

Safe claims:

- v103 is a real train/eval pipeline method change: render-time surface fields now support face-local affine barycentric residuals.
- On hard-triad `counter/kitchen/bonsai`, v103 `min_count=1` beats clean MeshSplatting in PSNR, SSIM, and LPIPS for every scene.
- v103 improves over v102b constant surface residuals on counter.
- The hard-triad v103 fields are fail-closed and use `--checkpoint_endpoint_require_surface_field`.

Unsafe claims:

- Do not claim v103 beats v101/v102a.
- Do not claim full9 validation.
- Do not claim unseen-camera generalization.
- Do not claim this is a vanilla MeshSplatting checkpoint.
- Do not claim paper-level closure; the remaining gap to v101/v102a is still large.

## 5. Next Required Step

v103 establishes the right representation direction, but the remaining gap says the field still needs conditional behavior:

```text
face id
  + barycentric coordinate
  + view direction / camera bin
  + evidence confidence or support agreement
  -> gated residual coefficients
  -> safe fallback where evidence is weak
```

The next milestone should be a v104 view-conditioned and evidence-gated field, validated first on counter, then hard triad, then full9 only if the same fixed policy passes.
