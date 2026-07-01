# Target-Neighbor Candidate Rerank Probe

Target GT is used only after target-blind candidate selection for analysis.

## Macro

| metric | current | fixed | learned | pure_tnc | oracle | pure_tnc-current | oracle-current |
|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | 0.107097397630 | 0.090757456239 | 0.120312893360 | 0.112655552646 | 0.133574686638 | +0.005558155016 | +0.026477289008 |
| ssim_gain | 0.001694096459 | 0.001593212287 | 0.001588390933 | 0.001650902960 | 0.001776662138 | -0.000043193499 | +0.000082565678 |

## Per Scene

| scene | current PSNR | pure_tnc PSNR | oracle PSNR | pure_tnc-current | oracle-current | TNC/GT match | pure_tnc best counts |
|---|---:|---:|---:|---:|---:|---:|---|
| treehill | 0.107097397630 | 0.112655552646 | 0.133574686638 | +0.005558155016 | +0.026477289008 | 12/18 | {'learned': 8, 'fixed': 9, 'mix0250': 1} |

## Verdict

pure_tnc is useful for measuring candidate-selection headroom. Promote it only if full9 macro improves over current without unacceptable scene regressions.
