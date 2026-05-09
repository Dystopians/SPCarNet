# Phase-J External Courtyard Validation

This report checks whether the Phase-J train-only ELA policy transfers outside the selected Mip-NeRF360 full9 protocol. It contains both a positive clean-checkpoint validation and a degraded-checkpoint diagnostic.

## Results

| protocol | method | base | dPSNR | dSSIM | dLPIPS | alpha | edge q | strict RGB | W&B |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_ela` | `ours_9000` | 0.244770 | 0.013113 | -0.015389 | 0.500 | 0.700 | yes | `vne962ci` |
| ETH3D courtyard clean9000 | `ours_9000_phasej_external_clean9000_micro_autoedge_ela` | `ours_9000_ela7_pareto_portfolio` | 0.041258 | 0.001397 | 0.003258 | 0.500 | 0.700 | no | `vne962ci` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_fixededge_ela` | `ours_26000` | 0.000000 | -0.000000 | -0.000000 | 0.000 | 0.500 | no | `e7aqkn3j` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_micro_autoedge_ela` | `ours_26000` | 0.000000 | -0.000000 | -0.000000 | 0.000 | 0.300 | no | `0plpo822` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_autoedge_ela` | `ours_26000` | 0.005244 | 0.000717 | -0.000651 | 0.125 | 0.900 | yes | `m651uff6` |
| ETH3D courtyard F82 degraded checkpoint | `ours_26000_phasej_external_fast_autoedge_ela` | `ours_26000` | 0.005758 | 0.000741 | -0.000664 | 0.125 | 0.900 | yes | `d7gckkmu` |

## Reading

- On the fair clean9000 courtyard checkpoint, Phase-J micro auto-edge improves all three RGB metrics over the clean baseline: `+0.244770` PSNR, `+0.013113` SSIM, `-0.015389` LPIPS.
- Against the older ELA7 courtyard portfolio, the same method improves PSNR and SSIM but not LPIPS, so it should be reported as a mixed replacement rather than a strict dominance result.
- On the F82 degraded checkpoint, full auto-edge produces only a very small strict RGB improvement. Fixed and micro policies correctly no-op. This is useful negative evidence: the policy is conservative, but severe checkpoint degradation is not solved by render-time residual transfer alone.
