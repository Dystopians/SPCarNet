# v104a Counter Summary

Date: 2026-06-25

Status: counter smoke passed. Not hard-triad, not full9, not final closure.

## Metrics

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean counter | 26.751774 | 0.862055 | 0.252003 |
| v103 affine min_count=1 | 27.208200 | 0.863405 | 0.243176 |
| v104a view-affine min_count=1 | 27.492378 | 0.867344 | 0.239003 |
| v101/v102a ceiling | 28.442907 | 0.893696 | 0.186557 |

## Delta

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104a minus clean | +0.740604 | +0.005288 | -0.013000 |
| v104a minus v103 | +0.284178 | +0.003939 | -0.004173 |
| v104a minus v101/v102a | -0.950529 | -0.026352 | +0.052446 |

## Verdict

Adding a linear view-direction basis is useful on counter. It closes part of the gap between v103 and the v101/v102a endpoint ceiling.

The current v104a is still a minimal, uncentered view-direction smoke. It must pass `kitchen` and `bonsai` before any hard-triad claim, and v104b should add centered view features plus fallback if hard-triad is unstable.
