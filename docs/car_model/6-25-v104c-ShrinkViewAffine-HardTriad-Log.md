# v104c Shrink View-Affine Hard-Triad Log

Date: 2026-06-25

## Motivation

v104a proved that adding view direction to the face-local affine residual field is useful, but the fit is under-supported for many triangles: a 6-feature RGB model is often fitted from very few pixels. A hard rank/min-view gate was a natural stability attempt, but it throws away too much of the view-conditioned signal.

v104c keeps the same render-time interface as v104a:

```text
basis_type = affine_barycentric_viewdir
stored coefficients = [1, u, v, viewdir_x, viewdir_y, viewdir_z]
```

The difference is in the builder. It fits a centered/scaled view-direction basis, folds the coefficients back into the raw render-time basis, and then shrinks the view-affine coefficients toward the v103 affine fallback using a fixed algebraic confidence score from rank, view support, and condition diagnostics.

## Implemented Files

```text
scripts/car_model/build_v104b_centered_view_affine_residual_field.py
```

The script supports both:

- `--fallback_mode hard`: strict v103 fallback for unsupported or ill-conditioned triangles.
- `--fallback_mode shrink`: soft interpolation from v103 fallback to centered view-affine coefficients.

No `render.py` change is needed because both modes store the existing `affine_barycentric_viewdir` payload layout.

## Counter Ablation

| method | PSNR | SSIM | LPIPS | interpretation |
|---|---:|---:|---:|---|
| v103 affine min_count=1 | 27.208200 | 0.863405 | 0.243176 | lower-capacity field |
| v104a raw view-affine | 27.492378 | 0.867344 | 0.239003 | useful raw view-direction basis |
| v104b hard fallback | 27.442575 | 0.865448 | 0.241198 | over-conservative; loses v104a gain |
| v104c shrink fallback | 27.498068 | 0.867420 | 0.238986 | fixed-policy weak positive over v104a |

v104b hard mode diagnosed the failure mode: `2,103,953` counter triangles fell back, including `1,199,825` for insufficient views and `904,128` for rank/condition. This was safer but weaker.

v104c shrink mode used all `2,716,449` observed counter triangles with `shrink_alpha_mean=0.566197`, which preserved the view signal while damping unstable coefficients.

## Hard-Triad Result

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 |
| v103 affine min_count=1 | 28.384418 | 0.879855 | 0.226611 |
| v104a raw view-affine | 28.823045 | 0.884927 | 0.219492 |
| v104c shrink view-affine | 28.859798 | 0.885459 | 0.219064 |
| v101/v102a endpoint ceiling | 30.167397 | 0.913355 | 0.163709 |

v104c improves over v104a by `+0.036753` PSNR, `+0.000532` SSIM, and `-0.000427` LPIPS on mean hard-triad metrics. The gain is modest but consistent across all three scenes and all three metrics.

It still trails v101/v102a by `-1.307599` PSNR, `-0.027896` SSIM, and `+0.055355` LPIPS. This remains the main paper gap.

## Claim Boundary

Safe claim:

> v104c is a fixed-policy, representation-level improvement over v104a: centered view-affine fitting plus algebraic shrinkage improves every hard-triad scene and metric without changing render-time access to target GT.

Unsafe claim:

> v104c fully replaces v101/v102a or closes the endpoint-to-field gap.

The honest story is that v104c improves the surface-field line, but the endpoint/delta-bank ceiling still carries significantly stronger image quality.

## Next Step

The highest-value next experiment is not more manual parameter scanning. The next method step should introduce either:

- a compact per-triangle residual mixture with two learned/shrinkage components, or
- a policy-val calibrated blend between v104c field output and v102a delta-bank output, then distill that blend back into a field.

Both target the remaining structural loss: v104c still compresses a per-pixel guarded residual into one low-order triangle-local function.
