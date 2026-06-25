# v103 Counter Summary

Date: 2026-06-25

Status: counter-only method evidence. This is not full9 validation, not unseen-camera generalization, and not final paper completion.

## Result Table

| method | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---|
| clean counter reference | 26.751774 | 0.862055 | 0.252003 | base MeshSplatting reference |
| v102b constant surface field | 27.058163 | 0.860653 | 0.249832 | one RGB residual per triangle |
| v103 affine surface field, min_count=3 | 27.167145 | 0.862644 | 0.244518 | face-local affine residual |
| v103 affine surface field, min_count=1 | 27.208200 | 0.863405 | 0.243176 | best current counter surface field |
| v101/v102a reference | 28.442907 | 0.893696 | 0.186557 | endpoint/preprojected ceiling |

## Main Takeaway

v103 is a real method upgrade over v102b: replacing a constant per-triangle residual with an affine barycentric face-local residual improves PSNR, SSIM, and LPIPS over both clean and v102b on counter.

The best current v103 counter variant is `min_count=1`, with:

```text
27.208200 PSNR / 0.863405 SSIM / 0.243176 LPIPS
```

Its gains are:

```text
vs clean: +0.456427 PSNR / +0.001350 SSIM / -0.008827 LPIPS
vs v102b: +0.150038 PSNR / +0.002752 SSIM / -0.006656 LPIPS
```

The gap to v101/v102a remains large:

```text
-1.234707 PSNR / -0.030291 SSIM / +0.056619 LPIPS
```

This means face-local structure is necessary, but the paper method still needs view conditioning and evidence-gated fallback.

## Key Paths

```text
Field:
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_min1_field.pt

Manifest:
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_min1_field.manifest.json

Render report:
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/test/ours_26000_v103_affine_min1_surface_field_counter/render_py_endpoint_report.json

Metrics:
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json
```

## Next Required Step

Do not expand the claim yet. The next method should add view-conditioned and evidence-gated coefficients on top of the v103 face-local affine basis, then validate counter, hard triad, and only then full9.
