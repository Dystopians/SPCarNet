# Final Stage F39 - Real Gate-Removed Ablation

Decision: `F39_COMPLETE`.

F39 runs a same-schedule parking PRISM ablation with the counterfactual gate enabled and removed. Both runs use online W&B and the same 500-iteration integrated topology-control configuration.

## Runs

| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| gated_control | `True` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1aggpvnr | `True` | 1 | 0 | 2579 | 0 | 64497 |
| no_gate | `False` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rx6tp8oi | `True` | 1 | 1 | 2579 | 2579 | 61918 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_control | 11.739765 | 0.324842 | 0.631196 | 0.314017 | 3.620568 | 51.551999 |
| no_gate | 11.733771 | 0.324798 | 0.631697 | 0.315812 | 3.632382 | 51.242019 |
| no_gate - gated | -0.005994 | -0.000044 | 0.000501 | 0.001795 | 0.011814 | -0.309979 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| gated_control | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| no_gate | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

If complete, this is the real-scene counterpart to F38: the gate-removed run can commit a candidate set that has no counterfactual acceptance, while the gated control exposes whether the same schedule rejects or rolls back the edit. The row remains a medium-budget ablation, not a replacement for final long-budget compact-recovery results.

## Supplemental 2000-Iteration Matched Control

A less aggressive ratio `0.02` 2000-iteration matched control was also run with online W&B:

| row | W&B | first round | final triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gated ratio0.02 2000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/je1f5z6x | accepted 1289 candidates | 767524 | 11.581725 | 0.271937 | 0.636721 | 0.434640 | 4.380206 | 53.124413 |
| no-gate ratio0.02 2000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d5bxekdo | committed same 1289 candidates without counterfactual acceptance | 767464 | 11.645683 | 0.266934 | 0.636923 | 0.420860 | 4.337693 | 52.692784 |

This supplemental control is mixed: the gate accepts the first candidate set at ratio `0.02`, so it is not a gate-necessity negative case. It is still useful because it shows the gate path does not block every topology edit; the load-bearing evidence is the ratio `0.04` row above, where the same selected candidate set is rolled back by the gate and committed by the gate-removed run.
