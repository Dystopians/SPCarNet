# Parking Best Clean Long vs Method Long Report

Date: 2026-05-03

## Purpose

Correct the clean-baseline comparison for `parking_phone_tiny`. The fair comparison is not clean 7000 iterations versus a method checkpoint at 22000 or 30000 iterations. The fair comparison is the best available clean long-horizon baseline versus the best available long-horizon method checkpoint.

## Runs

| run | iter | W&B | output | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth MAE ↓ | AbsRel ↓ | Normal deg ↓ | triangles | vertices |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean 7k | 7000 | historical | `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model` | `17.2047` | `0.5350` | `0.4507` | `1.7522` | `0.0761` | `45.5620` | `833775` | `1071408` |
| clean 22k | 22000 | `uus7fi39` | `outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model` | `18.4800` | `0.6346` | `0.3469` | `1.8684` | `0.0822` | `45.1084` | `8548242` | `2286499` |
| clean 30k | 30000 | `2q807xuf` | `outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_22000to30000/model` | `18.4088` | `0.6315` | `0.3510` | `1.8658` | `0.0816` | `44.8389` | `8548242` | `2286499` |
| ours R44 22k | 22000 | `c1rxa6q6` | `outputs/carnet/meshsplatopt/stageR44_01_parking_decay_sparse_frac0p50_lam0p001_16000to22000/recovery_model` | `17.1695` | `0.5487` | `0.4419` | `2.9194` | `0.1871` | `42.2183` | `782982` | `820107` |
| ours R43 30k | 30000 | `mhz6t8ps` | `outputs/carnet/meshsplatopt/stageR43_01b_parking_sparse_frac0p50_lam0p001_16000to30000/recovery_model` | `16.2492` | `0.5110` | `0.4774` | `3.0181` | `0.1937` | `43.7145` | `782982` | `820107` |

## Correct Conclusion

The strongest clean long baseline by render is clean 22k, not the earlier clean 7k reference. Compared with ours R44 22k, clean 22k is better on PSNR (`18.4800` vs `17.1695`), SSIM (`0.6346` vs `0.5487`), LPIPS (`0.3469` vs `0.4419`), Depth MAE (`1.8684` vs `2.9194`), and AbsRel (`0.0822` vs `0.1871`).

Ours R44 22k only wins on the sparse normal proxy (`42.2183` vs `45.1084` degrees) and topology size (`782982` vs `8548242` triangles). The parking result must therefore be framed as a low-topology/normal Pareto point, not as a render-quality win over the strongest clean long baseline.

## Artefacts

- Quantitative summary: `outputs/carnet/meshsplatopt/best_clean_long_vs_method_long/best_clean_long_vs_method_long_summary.md`
- Render montage: `outputs/carnet/meshsplatopt/best_clean_long_vs_method_long/best_clean_long_vs_method_long_render_montage.png`
