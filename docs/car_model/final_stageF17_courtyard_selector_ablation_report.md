# Final Stage F17 - Courtyard Selector Ablation

Date: 2026-05-04

Decision: `FINAL_F17_COURTYARD_SELECTOR_ABLATION_PASS_STRUCTURED_SELECTION`.

## Goal

Replicate the counter selector-control logic on a larger public scene. The comparison
uses the same clean-long source, same 50 percent target triangle count, same strict
topology-frozen 22k->26k recovery budget, online W&B logging, independent rendering,
independent image metrics, and COLMAP sparse geometry evaluation.

## Setup

- scene: `courtyard`
- dataset: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`
- clean source: `outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000`
- clean iteration: `22000`
- target prune fraction: `0.50`
- CSEF50 W&B: `jz93wrbc`
- area50 W&B: `hctwxtbe`
- random50 W&B: `faz0c00o`

## Results

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| area50 26k | 838,742 | 12.552895 | 0.338469 | 0.544993 | 0.324157 | 3.630241 | 40.907990 |
| random50 26k | 838,742 | 11.383848 | 0.264778 | 0.587667 | 0.371186 | 4.015910 | 41.158282 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSEF50 - clean | +0.452301 | +0.041625 | -0.024231 | -0.032415 | -0.220612 | +0.008508 |
| area50 - clean | +0.449387 | +0.041821 | -0.024315 | -0.030491 | -0.198803 | +0.086341 |
| random50 - clean | -0.719660 | -0.031870 | +0.018359 | +0.016538 | +0.186866 | +0.336633 |
| CSEF50 - area50 | +0.002914 | -0.000196 | -0.000084 | -0.001924 | -0.021809 | -0.077833 |
| CSEF50 - random50 | +1.171961 | +0.073495 | -0.042590 | -0.048953 | -0.407478 | -0.328125 |

## Interpretation

Courtyard confirms that arbitrary same-count pruning is not enough: random50 fails
clean-long and is much worse than both structured selectors. CSEF50 and area50 are
near-tied on render metrics, with area50 slightly ahead on SSIM/LPIPS and CSEF50
slightly ahead on PSNR and sparse geometry. The correct claim is therefore
`structured selection + strict recovery`, not a universal CSEF-over-area result.

For the main courtyard row, keep CSEF50 as the geometry-balanced choice because it has
better PSNR, AbsRel, Depth MAE, and Normal than area50 while matching the same triangle
budget.

