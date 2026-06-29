# SPCarNet v191-v199 Residual Projection Audit Summary

Date: 2026-06-29

This audit follows the v195-v199 negative result. The goal is to separate two
failure modes:

1. the trained checkpoint cannot project the Phase-J teacher residual even on
   held-out policy-val views;
2. the checkpoint can project the source residual, but target apply/certification
   destroys it.

The new tool is:

```text
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

It loads a trained checkpoint, applies it to the original teacher surface
evidence, and compares predicted residuals against `teacher_residual_rgb` on
policy-val views. It also compares final target renders against target GT
residuals after no-GT apply. The tool is read-only for checkpoints/evidence.

## Compared Runs

| Run | Role | Clean claim? | Official flowers PSNR / SSIM / LPIPS |
| --- | --- | ---: | --- |
| v191 | image-space U-Net calibration, GT-assisted | no | `20.606058 / 0.578882 / 0.323687` |
| v195 | surface texture MLP, teacher-only | yes | `19.878033 / 0.509020 / 0.402998` |
| v196 | surface texture MLP, GT-assisted diagnostic | no | `20.084991 / 0.523929 / 0.385202` |
| v199 | support-aware low-rank + target-visible capacity, teacher-only | yes | `19.835337 / 0.505801 / 0.404194` |

The Phase-J flowers gate remains:

```text
20.304358 / 0.557770 / 0.329222
```

## Residual Projection Results

| Run | Policy energy retention | Policy residual cosine | Policy changed frac | Target energy retention | Target residual cosine | Target changed frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v191 | 9.916031 | 0.279888 | 0.989272 | 0.253365 | 0.393485 | 0.982280 |
| v195 | 0.068206 | 0.112638 | 0.476928 | 0.002863 | 0.133734 | 0.566890 |
| v196 | 1.427611 | 0.138419 | 0.869394 | 0.029127 | 0.199612 | 0.839057 |
| v199 | 0.015229 | 0.039391 | 0.049312 | 0.000847 | 0.028702 | 0.071136 |

## Interpretation

The surface-based v195-v199 family fails before full target evaluation:

- v199 is safe but almost no-ops. It retains only `1.52%` of policy-val teacher
  residual energy and only `0.085%` of target GT residual energy.
- v195 writes more than v199, but still retains only `6.82%` of policy-val teacher
  residual energy. It is not a strong teacher-residual projection.
- v196 proves stronger supervision can increase write magnitude, but residual
  direction remains weak: policy cosine is only `0.138`, and target cosine only
  `0.200`.
- v191 has much stronger target residual alignment, but it is an image-space
  GT-assisted calibration run rather than a clean surface-representation method.

Therefore, the current bottleneck is representation/objective mismatch in the
surface carrier. The next method should not be another low-rank width or face
budget sweep. It needs a view-conditioned residual carrier whose held-out
source-view residual energy and cosine are explicitly certified before target
apply.

## Next Engineering Gate

Before any new full9 or paper-ready claim, require a candidate to satisfy a
source-view projection gate:

```text
policy residual cosine >= 0.25
target-free policy residual energy retention in [0.25, 4.0]
policy PSNR/SSIM vs teacher does not degrade materially
```

This gate is not sufficient for paper readiness, but v195-v199 show it is a
necessary filter. Without it, the exact target run mostly measures a weak or
misaligned carrier.

