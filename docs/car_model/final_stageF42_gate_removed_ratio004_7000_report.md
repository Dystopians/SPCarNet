# Final Stage F42 - Real Gate-Removed Ratio0.04 7000-Iteration Ablation

Decision: `F42_COMPLETE`.

F42 extends F41 from 2000 to 7000 iterations on the same parking ratio0.04 gate-on/gate-off schedule with online W&B. It is a closer-to-long-budget mechanism check, not a replacement for the final 22k/26k compact-recovery table.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF42_real_gate_removed_ratio004_7000/summary/final_stageF42_gate_removed_ratio004_7000.json`

## Runs

| row | gate enabled | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_7000 | `True` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 1 | 0 | 1 | 2579 | 0 | 822904 |
| no_gate_ratio004_7000 | `False` | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/o05nx4za | `True` | 1 | 1 | 0 | 2579 | 2579 | 829354 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_7000 | 17.254513 | 0.535237 | 0.453228 | 0.077416 | 1.775428 | 45.816557 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| no_gate - gated | -0.109383 | -0.002887 | 0.000806 | -0.001243 | -0.051792 | -0.175861 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| gated_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| no_gate_ratio004_7000 | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

Mechanism pass: both rows select the same 2579-candidate ratio0.04 edit at iter 141; the gated row records counterfactual_accept=0 and rolls back, while the gate-removed row commits it. 7000-step render evidence favors the gate: gated improves PSNR by 0.109383, improves SSIM by 0.002887, and reduces LPIPS by 0.000806 versus no-gate. Sparse geometry proxies favor no-gate at this budget: AbsRel, Depth MAE, and normal angle are lower by 0.001243, 0.051792, and 0.175861. The safe paper claim is therefore stronger than F41 for visual/held-out-render gate necessity, but still not universal metric dominance: use F42 as long-budget parking evidence that rollback prevents an unsafe no-accept topology edit and improves render quality, while geometry proxies remain mixed.
