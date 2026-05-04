# Final Stage F43 - Bonsai Gate-Removed 7000-Iteration Ablation

Decision: `F43_COMPLETE`.

F43 extends the real gate-removed ablation beyond parking. Both rows use the same bonsai ratio0.02 PRISM schedule, 7000 training iterations, online W&B, independent test-set rendering metrics, and sparse COLMAP geometry evaluation.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF43_bonsai_gate_removed_7000/summary/final_stageF43_bonsai_gate_removed_7000.json`

## Runs

| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_bonsai_ratio002_7000 | `True` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xymcrg63 | `True` | 6 | 0 | 6 | 76110 | 0 | 123115 |
| no_gate_bonsai_ratio002_7000 | `False` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1vnfreq6 | `True` | 6 | 6 | 0 | 25743 | 25743 | 21231 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_bonsai_ratio002_7000 | 23.025888 | 0.732148 | 0.300588 | 0.106861 | 1.211684 | 41.492277 |
| no_gate_bonsai_ratio002_7000 | 24.719440 | 0.837326 | 0.184327 | 0.017143 | 0.143226 | 23.980772 |
| no_gate - gated | 1.693552 | 0.105178 | -0.116261 | -0.089718 | -1.068458 | -17.511505 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| gated_bonsai_ratio002_7000 | 1501 | `False` | 0 | 1 | 12685 | 634299 | 634299 |
| no_gate_bonsai_ratio002_7000 | 1501 | `True` | 0 | 0 | 12685 | 634299 | 621614 |

## Interpretation

Mechanism divergence is real: the same bonsai ratio0.02 7000-step schedule produces different commit/rollback behavior when the counterfactual gate is enabled versus removed. This is a negative gate-generalization result for the current bonsai schedule: no-gate wins all three render metrics and all three sparse geometry proxies, while also ending with a much smaller mesh. The safe paper conclusion is not broad multi-scene final-budget gate superiority. F43 should be used as a weakness-finding ablation: the current gate/rollback policy can be too conservative or can steer the recovery trajectory poorly on bonsai, so any final claim must emphasize the validated compact-recovery main table and the parking/synthetic unsafe-edit evidence, while listing adaptive scene-aware gate calibration as required future/fix work before claiming universal gate dominance.
