# Final Stage F63 - Parking Adaptive Arbitrated CSEF Policy 7000

Decision: `F63_COMPLETE`.

F63 replaces the fixed ratio sweep with an Adaptive Order-Preserving Arbitrated CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF63_parking_adaptive_order_preserving_arbitrated_policy_7000/summary/final_stageF63_parking_adaptive_order_preserving_arbitrated_policy_7000.json`

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
| adaptive_order_preserving_arbitrated_policy_F63 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2sz0k6w | `True` | 1 | 0 | 2475 | 2475 | 850186 |
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
| adaptive_order_preserving_arbitrated_policy_F63 | 17.258318 | 0.538725 | 0.447951 | 0.076046 | 1.708921 | 45.638923 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F63 - no_gate | 0.113188 | 0.006374 | -0.006083 | -0.000126 | -0.014715 | -0.001774 |
| F63 - strict | 0.003805 | 0.003488 | -0.005277 | -0.001370 | -0.066507 | -0.177634 |
| F63 - calibrated | 0.091694 | 0.005246 | -0.005661 | -0.000837 | -0.048073 | -0.105224 |
| F63 - F58 | 0.002768 | 0.000350 | -0.000519 | 0.001120 | -0.005796 | 0.247182 |
| F63 - F56 | 0.037752 | 0.002801 | -0.003118 | 0.000506 | 0.087793 | 0.079506 |

## Interpretation

F63 adaptive order-preserving CSEF policy strictly beats no-gate on all six tracked metrics but remains mixed versus F56.
