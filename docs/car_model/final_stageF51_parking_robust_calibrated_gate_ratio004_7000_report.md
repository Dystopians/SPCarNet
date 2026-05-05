# Final Stage F51 - Parking Robust Calibrated Gate Ratio0.04 7000-Iteration Repair

Decision: `F51_COMPLETE`.

F51 tests the AbsRel-reliability repair: when baseline AbsRel is above the reliability threshold, the counterfactual gate does not let that unstable AbsRel delta alone reject an otherwise visually tiny edit. F51 is compared against F42 strict/no-gate and F50 calibrated-gate references.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF51_parking_robust_calibrated_gate_ratio004_7000/summary/final_stageF51_parking_robust_calibrated_gate_ratio004_7000.json`

## Runs

| row | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 1 | 0 | 1 | 2579 | 0 | 822904 |
| calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/k2rr83jh | `True` | 1 | 0 | 1 | 2579 | 0 | 829157 |
| robust_calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/02qseh0s | `True` | 1 | 1 | 0 | 2579 | 2579 | 828362 |
| no_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/o05nx4za | `True` | 1 | 1 | 0 | 2579 | 2579 | 829354 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 17.254513 | 0.535237 | 0.453228 | 0.077416 | 1.775428 | 45.816557 |
| calibrated_gate_ratio004_7000 | 17.166624 | 0.533479 | 0.453611 | 0.076883 | 1.756994 | 45.744147 |
| robust_calibrated_gate_ratio004_7000 | 17.167955 | 0.533058 | 0.453975 | 0.079928 | 1.794996 | 45.658641 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| robust - strict_gate | -0.086557 | -0.002179 | 0.000747 | 0.002512 | 0.019568 | -0.157917 |
| robust - calibrated_gate | 0.001331 | -0.000421 | 0.000363 | 0.003045 | 0.038001 | -0.085506 |
| robust - no_gate | 0.022825 | 0.000708 | -0.000059 | 0.003755 | 0.071360 | 0.017944 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| calibrated_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| robust_calibrated_gate_ratio004_7000 | 141 | `True` | 1 | 0 | 2579 | 64497 | 61918 |
| no_gate_ratio004_7000 | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

F51 fixes the F50 mechanism weakness: robust calibrated gating commits the early ratio0.04 candidate that F50/strict gate rolled back. It remains render-positive versus no-gate, but sparse geometry is not fully dominant.
