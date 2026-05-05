# Final Stage F53 - Parking Delayed Robust Calibrated Gate Ratio0.02 7000-Iteration Repair

Decision: `F53_COMPLETE`.

F53 delays candidate selection until geometry has become reliable and lowers the prune ratio from 0.04 to 0.02. This directly tests the F51/F52 diagnosis: early ratio0.04 commits can improve appearance but leave sparse geometry mixed, while delayed ratio0.04 is too aggressive to pass a reliable gate.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000/summary/final_stageF53_parking_delayed_robust_calibrated_gate_ratio002_7000.json`

## Runs

| row | W&B | ready | candidate rounds | commits | rollbacks | selected | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 1 | 0 | 1 | 2579 | 0 | 822904 |
| calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/k2rr83jh | `True` | 1 | 0 | 1 | 2579 | 0 | 829157 |
| early_robust_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/02qseh0s | `True` | 1 | 1 | 0 | 2579 | 2579 | 828362 |
| delayed_robust_gate_ratio002_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gstmhuzq | `True` | 1 | 1 | 0 | 3961 | 3961 | 848570 |
| no_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/o05nx4za | `True` | 1 | 1 | 0 | 2579 | 2579 | 829354 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 17.254513 | 0.535237 | 0.453228 | 0.077416 | 1.775428 | 45.816557 |
| calibrated_gate_ratio004_7000 | 17.166624 | 0.533479 | 0.453611 | 0.076883 | 1.756994 | 45.744147 |
| early_robust_gate_ratio004_7000 | 17.167955 | 0.533058 | 0.453975 | 0.079928 | 1.794996 | 45.658641 |
| delayed_robust_gate_ratio002_7000 | 17.255213 | 0.538878 | 0.448564 | 0.076453 | 1.730419 | 45.241471 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F53 - strict | 0.000700 | 0.003641 | -0.004664 | -0.000963 | -0.045009 | -0.575087 |
| F53 - calibrated | 0.088589 | 0.005399 | -0.005048 | -0.000430 | -0.026575 | -0.502676 |
| F53 - early_robust | 0.087257 | 0.005820 | -0.005411 | -0.003475 | -0.064576 | -0.417170 |
| F53 - no_gate | 0.110083 | 0.006528 | -0.005470 | 0.000280 | 0.006784 | -0.399226 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| calibrated_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| early_robust_gate_ratio004_7000 | 141 | `True` | 1 | 0 | 2579 | 64497 | 61918 |
| delayed_robust_gate_ratio002_7000 | 1501 | `True` | 1 | 0 | 3961 | 198057 | 194096 |
| no_gate_ratio004_7000 | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

F53 is a strict all-metric win over the strict gate reference. It strictly repairs F50 calibrated gate. It strictly repairs F51 early robust gate. It is mixed versus no-gate, so further work is required before claiming universal dominance.
