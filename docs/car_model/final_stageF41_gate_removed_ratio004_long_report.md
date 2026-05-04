# Final Stage F41 - Real Gate-Removed Ratio0.04 Long Ablation

Decision: `F41_COMPLETE`.

F41 extends F39's real parking gate ablation from the short 500-step aggressive ratio0.04 case to a 2000-iteration same-schedule pair with online W&B. This is intended to close the reviewer concern that the real gate-removed evidence was too short-budget.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF41_real_gate_removed_ratio004_long/summary/final_stageF41_gate_removed_ratio004_long.json`

## Runs

| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_long | `True` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/eaz8fh2o | `True` | 1 | 0 | 1 | 2579 | 0 | 783009 |
| no_gate_ratio004_long | `False` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vyi2uf4h | `True` | 1 | 1 | 0 | 2579 | 2579 | 751960 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_long | 11.637346 | 0.265729 | 0.635825 | 0.422537 | 4.285561 | 52.706472 |
| no_gate_ratio004_long | 11.667192 | 0.270661 | 0.635146 | 0.418002 | 4.323293 | 52.965324 |
| no_gate - gated | 0.029846 | 0.004932 | -0.000679 | -0.004536 | 0.037732 | 0.258852 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_long | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| no_gate_ratio004_long | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

F41 is the long-budget real-scene counterpart to F38/F39 for the aggressive ratio0.04 edit schedule. The mechanism evidence is strong: the gated run rolls back the same no-counterfactual-accept candidate set that the gate-removed run commits. The final metrics are mixed rather than a clean gate-on win: no-gate is slightly better on PSNR, SSIM, LPIPS, and AbsRel, while gated is better on Depth MAE and normal and preserves more topology. This should be reported as long-budget gate/rollback necessity evidence for unsafe edit rejection, not as proof that the gate monotonically improves every final metric.
