# v103 Hard-Triad Summary

Date: 2026-06-25

Status: hard-triad evidence only. Not full9, not unseen-camera generalization, not final closure.

## Mean Metrics

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 |
| v103 affine min_count=1 | 28.384418 | 0.879855 | 0.226611 |
| v101/v102a endpoint ceiling | 30.167395 | 0.913355 | 0.163709 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v103 minus clean | +0.562565 | +0.001552 | -0.010283 |
| v103 minus v101/v102a | -1.782977 | -0.033500 | +0.062902 |

## Per Scene

| scene | v103 PSNR | v103 SSIM | v103 LPIPS | vs clean |
|---|---:|---:|---:|---|
| counter | 27.208200 | 0.863405 | 0.243176 | all three better |
| kitchen | 28.310152 | 0.877554 | 0.194518 | all three better |
| bonsai | 29.634901 | 0.898607 | 0.242140 | all three better |

## Verdict

v103 `affine_barycentric` `min_count=1` is the first surface-field method that beats clean MeshSplatting on PSNR, SSIM, and LPIPS across the hard triad.

It still falls well below the v101/v102a endpoint ceiling, so the next step is not full9 promotion. The next step is v104 view-conditioned and evidence-gated residual coefficients.
