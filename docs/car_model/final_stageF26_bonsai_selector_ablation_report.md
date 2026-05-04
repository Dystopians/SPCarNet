# Final Stage F26 - Bonsai Selector Ablation Report

Decision: `FINAL_F26_BONSAI_SELECTOR_ABLATION_PASS_AREA_PARETO_RANDOM_FAIL`.

## Goal

Close the remaining bonsai selector-control gap with fair long-horizon rows. Both controls start from the same clean-long 22k checkpoint, remove exactly 50 percent of triangles, recover with strict topology freeze from `22000->26000`, log online W&B, then run independent render metrics and sparse COLMAP geometry evaluation.

## Runs

| row | selector | triangles | vertices | W&B |
| --- | --- | ---: | ---: | --- |
| area50 | `area_smallest` | 44,230 | 87,854 | `a29ayt8w` |
| random50 | `random_same_count` | 44,230 | 99,588 | `noqp4nhp` |

Both compact checkpoints had `0` invalid indices and `0` degenerate faces.

## Independent Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| CSEF50 26k | 44,230 | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 |
| Open3D QEM50 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |
| area50 26k | 44,230 | 11.072339 | 0.242361 | 0.570040 | 0.179402 | 1.755109 | 42.834537 |
| random50 26k | 44,230 | 10.725461 | 0.197036 | 0.603335 | 0.210644 | 1.736676 | 43.797014 |

## Interpretation

Random same-count pruning is rejected: it is worse than clean-long on PSNR, SSIM, LPIPS, and AbsRel, and much worse than structured selectors at the identical triangle count.

Area50 is a strong structured Pareto row. It is slightly behind QEM50 on PSNR (`-0.010066`) and SSIM (`-0.000888`), but improves LPIPS (`-0.000137`), AbsRel (`-0.003564`), Depth MAE (`-0.038743`), and Normal (`-0.054802`) relative to QEM50. QEM50 remains the bonsai render-headline row; area50 becomes the bonsai geometry/perceptual Pareto control.

