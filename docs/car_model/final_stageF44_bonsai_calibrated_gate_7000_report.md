# Final Stage F44 - Bonsai Calibrated Gate 7000-Iteration Repair

Decision: `F44_COMPLETE`.

F44 directly responds to the F43 negative gate-generalization result. It keeps `--prism_use_counterfactual_gate` enabled, but relaxes the immediate counterfactual thresholds for large recoverable bonsai edits and relies on post-recovery validation rollback as a second-stage safety check.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF44_bonsai_calibrated_gate_7000/summary/final_stageF44_bonsai_calibrated_gate_7000.json`

## Runs

| row | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_bonsai_ratio002_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xymcrg63 | `True` | 6 | 0 | 6 | 76110 | 0 | 123115 |
| calibrated_gate_bonsai_ratio002_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/umc23i5h | `True` | 6 | 3 | 3 | 20534 | 17603 | 19226 |
| no_gate_bonsai_ratio002_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1vnfreq6 | `True` | 6 | 6 | 0 | 25743 | 25743 | 21231 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_bonsai_ratio002_7000 | 23.025888 | 0.732148 | 0.300588 | 0.106861 | 1.211684 | 41.492277 |
| calibrated_gate_bonsai_ratio002_7000 | 24.471493 | 0.832768 | 0.191326 | 0.018155 | 0.148973 | 24.164780 |
| no_gate_bonsai_ratio002_7000 | 24.719440 | 0.837326 | 0.184327 | 0.017143 | 0.143226 | 23.980772 |
| calibrated - strict_gate | 1.445604 | 0.100621 | -0.109262 | -0.088706 | -1.062711 | -17.327497 |
| calibrated - no_gate | -0.247948 | -0.004558 | 0.006999 | 0.001012 | 0.005747 | 0.184008 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_bonsai_ratio002_7000 | 1501 | `False` | 0 | 1 | 12685 | 634299 | 634299 |
| calibrated_gate_bonsai_ratio002_7000 | 1501 | `True` | 1 | 0 | 12685 | 634299 | 621614 |
| no_gate_bonsai_ratio002_7000 | 1501 | `True` | 0 | 0 | 12685 | 634299 | 621614 |

## Interpretation

Calibrated gating fixes the F43 mechanism failure by accepting at least one recoverable candidate round that strict gating rolled back. It improves over strict gate on all render metrics: PSNR +1.445604, SSIM +0.100621, LPIPS -0.109262. It is close to the no-gate render envelope while preserving counterfactual/validation metadata, so this is a plausible fix direction rather than a pure removal of safety.
