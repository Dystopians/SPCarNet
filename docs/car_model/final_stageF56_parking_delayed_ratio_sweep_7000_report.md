# Final Stage F56 - Parking Delayed Ratio Sweep 7000

Decision: `F56_IN_PROGRESS`.

F56 tests ratio0.0125 after F53 ratio0.02 fixed normal/render but left tiny no-gate depth gaps, F54 ratio0.01 fixed depth but lost normal, and F55 ratio0.015 was rejected by the reliable delayed gate.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF56_parking_delayed_robust_calibrated_gate_ratio00125_7000/summary/final_stageF56_parking_delayed_ratio_sweep_7000.json`

## Runs

| row | W&B | ready | commits | rollbacks | selected | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 0 | 1 | 2579 | 0 | 822904 |
| calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/k2rr83jh | `True` | 0 | 1 | 2579 | 0 | 829157 |
| delayed_ratio002_F53 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gstmhuzq | `True` | 1 | 0 | 3961 | 3961 | 848570 |
| delayed_ratio001_F54 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/089vvbqq | `True` | 1 | 0 | 1980 | 1980 | 842970 |
| delayed_ratio0015_F55 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bhyrvl0v | `False` | 0 | 1 | 2970 | 0 | NA |
| delayed_ratio00125_F56 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j92cfr5z | `True` | 1 | 0 | 2475 | 2475 | 847638 |
| no_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/o05nx4za | `True` | 1 | 0 | 2579 | 2579 | 829354 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 17.254513 | 0.535237 | 0.453228 | 0.077416 | 1.775428 | 45.816557 |
| calibrated_gate_ratio004_7000 | 17.166624 | 0.533479 | 0.453611 | 0.076883 | 1.756994 | 45.744147 |
| delayed_ratio002_F53 | 17.255213 | 0.538878 | 0.448564 | 0.076453 | 1.730419 | 45.241471 |
| delayed_ratio001_F54 | 17.236671 | 0.536608 | 0.449873 | 0.075799 | 1.660006 | 45.771574 |
| delayed_ratio0015_F55 | NA | NA | NA | NA | NA | NA |
| delayed_ratio00125_F56 | 17.220566 | 0.535924 | 0.451069 | 0.075541 | 1.621127 | 45.559417 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F56 - no_gate | 0.075436 | 0.003574 | -0.002965 | -0.000632 | -0.102508 | -0.081280 |
| F56 - strict | -0.033947 | 0.000687 | -0.002159 | -0.001876 | -0.154301 | -0.257141 |
| F56 - calibrated | 0.053942 | 0.002445 | -0.002543 | -0.001342 | -0.135867 | -0.184730 |

## Interpretation

F56 is still in progress.
