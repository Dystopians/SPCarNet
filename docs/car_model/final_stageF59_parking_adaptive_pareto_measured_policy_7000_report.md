# Final Stage F59 - Parking Adaptive Measured CSEF Policy 7000

Decision: `F59_COMPLETE`.

F59 replaces the fixed ratio sweep with an Adaptive Pareto Measured CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF59_parking_adaptive_pareto_measured_policy_7000/summary/final_stageF59_parking_adaptive_pareto_measured_policy_7000.json`

## Runs

| row | W&B | ready | commits | rollbacks | selected | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 0 | 1 | 2579 | 0 | 822904 |
| calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/k2rr83jh | `True` | 0 | 1 | 2579 | 0 | 829157 |
| delayed_ratio002_F53 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gstmhuzq | `True` | 1 | 0 | 3961 | 3961 | 848570 |
| delayed_ratio001_F54 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/089vvbqq | `True` | 1 | 0 | 1980 | 1980 | 842970 |
| delayed_ratio0015_F55 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bhyrvl0v | `False` | 0 | 1 | 2970 | 0 | NA |
| delayed_ratio00125_F56 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j92cfr5z | `True` | 1 | 0 | 2475 | 2475 | 847638 |
| adaptive_pareto_policy_F58 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/8nwdc8gi | `True` | 1 | 0 | 2471 | 2471 | 848518 |
| adaptive_pareto_measured_policy_F59 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/2jq5cd1v | `True` | 1 | 0 | 2048 | 2048 | 847919 |
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
| adaptive_pareto_policy_F58 | 17.255550 | 0.538375 | 0.448470 | 0.074927 | 1.714717 | 45.391741 |
| adaptive_pareto_measured_policy_F59 | 17.332283 | 0.538152 | 0.449448 | 0.075229 | 1.703694 | 45.355506 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F59 - no_gate | 0.187153 | 0.005801 | -0.004585 | -0.000943 | -0.019941 | -0.285190 |
| F59 - strict | 0.077770 | 0.002915 | -0.003779 | -0.002187 | -0.071734 | -0.461051 |
| F59 - calibrated | 0.165659 | 0.004673 | -0.004163 | -0.001654 | -0.053300 | -0.388641 |
| F59 - F58 | 0.076733 | -0.000223 | 0.000979 | 0.000303 | -0.011023 | -0.036234 |
| F59 - F56 | 0.111717 | 0.002228 | -0.001620 | -0.000311 | 0.082567 | -0.203910 |

## Interpretation

F59 adaptive CSEF policy strictly beats no-gate on all six tracked metrics but remains mixed versus F56.
