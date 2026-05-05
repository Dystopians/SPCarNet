# Final Stage F61 - Parking Adaptive Geometry Pareto Cover CSEF Policy 7000

Decision: `F61_COMPLETE`.

F61 replaces the fixed ratio sweep with an Adaptive Geometry Pareto Cover CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF61_parking_adaptive_geom_pareto_cover_policy_7000/summary/final_stageF61_parking_adaptive_geom_pareto_cover_policy_7000.json`

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
| adaptive_geom_pareto_cover_policy_F61 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/6mz4a2i6 | `True` | 1 | 0 | 2476 | 2476 | 846223 |
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
| adaptive_geom_pareto_cover_policy_F61 | 17.211843 | 0.537384 | 0.447826 | 0.077971 | 1.707759 | 45.560175 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F61 - no_gate | 0.066713 | 0.005033 | -0.006207 | 0.001798 | -0.015877 | -0.080522 |
| F61 - strict | -0.042669 | 0.002146 | -0.005402 | 0.000555 | -0.067669 | -0.256382 |
| F61 - calibrated | 0.045219 | 0.003904 | -0.005785 | 0.001088 | -0.049235 | -0.183972 |
| F61 - F58 | -0.043707 | -0.000991 | -0.000644 | 0.003045 | -0.006959 | 0.168434 |
| F61 - F56 | -0.008722 | 0.001459 | -0.003242 | 0.002431 | 0.086631 | 0.000758 |

## Interpretation

F61 is complete but still mixed versus no-gate; inspect deltas before promoting the adaptive policy.
