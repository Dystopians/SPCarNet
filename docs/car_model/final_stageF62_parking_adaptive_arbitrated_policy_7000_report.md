# Final Stage F62 - Parking Adaptive Arbitrated CSEF Policy 7000

Decision: `F62_COMPLETE`.

F62 replaces the fixed ratio sweep with an Adaptive Arbitrated CSEF Edit Policy that chooses candidate ratio, candidate cap, risk-sensitive ranking weights, and gate scaling from scene evidence.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF62_parking_adaptive_arbitrated_policy_7000/summary/final_stageF62_parking_adaptive_arbitrated_policy_7000.json`

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
| adaptive_arbitrated_policy_F62 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/wui1nbfy | `True` | 1 | 0 | 2472 | 2472 | 841771 |
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
| adaptive_arbitrated_policy_F62 | 17.183020 | 0.537146 | 0.449773 | 0.075983 | 1.726114 | 45.463908 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| F62 - no_gate | 0.037889 | 0.004795 | -0.004260 | -0.000190 | 0.002479 | -0.176788 |
| F62 - strict | -0.071493 | 0.001908 | -0.003455 | -0.001434 | -0.049314 | -0.352649 |
| F62 - calibrated | 0.016396 | 0.003666 | -0.003838 | -0.000901 | -0.030880 | -0.280239 |
| F62 - F58 | -0.072531 | -0.001229 | 0.001303 | 0.001056 | 0.011397 | 0.072168 |
| F62 - F56 | -0.037546 | 0.001221 | -0.001295 | 0.000442 | 0.104987 | -0.095508 |

## Interpretation

F62 is complete but still mixed versus no-gate; inspect deltas before promoting the adaptive policy.
