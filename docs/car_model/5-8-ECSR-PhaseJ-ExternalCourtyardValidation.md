# Phase-J External Courtyard Validation

This report checks whether the Phase-J train-only ELA policy transfers outside the selected Mip-NeRF360 full9 protocol. It contains both a positive clean-checkpoint validation and a degraded-checkpoint diagnostic.

## Results

| protocol | method | base | dPSNR | dSSIM | dLPIPS | alpha | edge q | strict RGB | W&B |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_ela` | `ours_9000` | 0.244770 | 0.013113 | -0.015389 | 0.500 | 0.700 | yes | `vne962ci` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_ela` | `ours_9000_ela7_pareto_portfolio` | 0.041258 | 0.001397 | 0.003258 | 0.500 | 0.700 | no | `vne962ci` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_lpips_ela` | `ours_9000` | 0.264227 | 0.009396 | -0.022513 | 0.500 | 0.300 | yes | `k6i8bg64` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_lpips_ela` | `ours_9000_ela7_pareto_portfolio` | 0.060715 | -0.002319 | -0.003865 | 0.500 | 0.300 | no | `k6i8bg64` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_autoedge_lpips_ela` | `ours_9000` | 0.263348 | 0.009438 | -0.022823 | 0.500 | 0.300 | yes | `yvskkcod` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_autoedge_lpips_ela` | `ours_9000_ela7_pareto_portfolio` | 0.059835 | -0.002277 | -0.004176 | 0.500 | 0.300 | no | `yvskkcod` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_fixededge_ela` | `ours_26000` | 0.000000 | -0.000000 | -0.000000 | 0.000 | 0.500 | no | `e7aqkn3j` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_micro_autoedge_ela` | `ours_26000` | 0.000000 | -0.000000 | -0.000000 | 0.000 | 0.300 | no | `0plpo822` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_autoedge_ela` | `ours_26000` | 0.005244 | 0.000717 | -0.000651 | 0.125 | 0.900 | yes | `m651uff6` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_fast_autoedge_ela` | `ours_26000` | 0.005758 | 0.000741 | -0.000664 | 0.125 | 0.900 | yes | `d7gckkmu` |

## Reading

- On the fair clean9000 courtyard checkpoint, all completed Phase-J variants improve all three RGB metrics over the clean baseline. The best LPIPS-aware row reaches `+0.263348` PSNR, `+0.009438` SSIM, `-0.022823` LPIPS vs clean9000.
- Against the older ELA7 courtyard portfolio, the LPIPS-aware rows improve PSNR and LPIPS but not SSIM, while the no-LPIPS micro row improves PSNR and SSIM but not LPIPS. This is a useful external validation, not a strict ELA7 replacement.
- On the F82 degraded checkpoint, full auto-edge produces only a very small strict RGB improvement. Fixed and micro policies correctly no-op. This is useful negative evidence: the policy is conservative, but severe checkpoint degradation is not solved by render-time residual transfer alone.
