# Final Stage F64 - Parking Adaptive Arbitrated CSEF Policy 7000

Decision: `F64_COMPLETE`.

F64 replaces the fixed ratio sweep with an Adaptive Order-Preserving Bulk-Fallback CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF64_parking_adaptive_bulk_fallback_policy_7000/summary/final_stageF64_parking_adaptive_bulk_fallback_policy_7000.json`

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
| adaptive_bulk_fallback_policy_F64 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jr8w32cj | `True` | 1 | 0 | 2474 | 2474 | 844615 |
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
| adaptive_bulk_fallback_policy_F64 | 17.260597 | 0.537938 | 0.449156 | 0.078015 | 1.773286 | 45.480246 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F64 - no_gate | 0.115467 | 0.005587 | -0.004877 | 0.001842 | 0.049650 | -0.160450 |
| F64 - strict | 0.006084 | 0.002701 | -0.004071 | 0.000599 | -0.002142 | -0.336311 |
| F64 - calibrated | 0.093973 | 0.004459 | -0.004455 | 0.001132 | 0.016292 | -0.263901 |
| F64 - F58 | 0.005047 | -0.000437 | 0.000687 | 0.003088 | 0.058569 | 0.088506 |
| F64 - F56 | 0.040031 | 0.002014 | -0.001912 | 0.002474 | 0.152158 | -0.079170 |

## Interpretation

F64 is complete but still mixed versus no-gate; inspect deltas before promoting the adaptive policy.
