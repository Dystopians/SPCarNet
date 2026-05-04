# Final Stage F50 - Parking Calibrated Gate Ratio0.04 7000-Iteration Replication

Decision: `F50_COMPLETE`.

F50 tests whether the F44 calibrated counterfactual-gate thresholds replicate on the parking F42 ratio0.04 7000-step schedule. The strict-gate and no-gate rows are the completed F42 references; the calibrated row is a new online-W&B long run with the same candidate schedule and relaxed immediate gate thresholds.

- summary JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshprior/stageF50_parking_calibrated_gate_ratio004_7000/summary/final_stageF50_parking_calibrated_gate_ratio004_7000.json`

## Runs

| row | W&B | results ready | candidate rounds | committed rounds | rollback rounds | selected candidates | committed selected | final triangles |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/era2si2w | `True` | 1 | 0 | 1 | 2579 | 0 | 822904 |
| calibrated_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/k2rr83jh | `True` | 1 | 0 | 1 | 2579 | 0 | 829157 |
| no_gate_ratio004_7000 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/o05nx4za | `True` | 1 | 1 | 0 | 2579 | 2579 | 829354 |

## Metrics

| row | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 17.254513 | 0.535237 | 0.453228 | 0.077416 | 1.775428 | 45.816557 |
| calibrated_gate_ratio004_7000 | 17.166624 | 0.533479 | 0.453611 | 0.076883 | 1.756994 | 45.744147 |
| no_gate_ratio004_7000 | 17.145130 | 0.532351 | 0.454033 | 0.076173 | 1.723636 | 45.640697 |
| calibrated - strict_gate | -0.087889 | -0.001758 | 0.000384 | -0.000533 | -0.018434 | -0.072410 |
| calibrated - no_gate | 0.021494 | 0.001129 | -0.000422 | 0.000710 | 0.033359 | 0.103450 |

## First Candidate Round

| row | iteration | committed | counterfactual accept | rollback | selected | pre triangles | post triangles |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| strict_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| calibrated_gate_ratio004_7000 | 141 | `False` | 0 | 1 | 2579 | 64497 | 64497 |
| no_gate_ratio004_7000 | 141 | `True` | 0 | 0 | 2579 | 64497 | 61918 |

## Interpretation

The calibrated gate follows the same mechanism as the strict F42 gate on parking: it rejects and rolls back the same no-accept ratio0.04 candidate round, so F50 does not replicate the F44 bonsai behavior of accepting recoverable edits. Relative to strict gate, calibrated gate gives back render quality (PSNR -0.087889, SSIM -0.001758, LPIPS +0.000384). It does improve sparse geometry proxies versus strict gate (AbsRel -0.000533, Depth MAE -0.018434, Normal -0.072410). Against no-gate, calibrated gate still wins all render metrics while preserving rollback metadata (PSNR +0.021494, SSIM +0.001129, LPIPS -0.000422). No-gate remains better on sparse geometry proxies, so the paper claim should stay narrow: F50 supports render-quality rollback safety on parking, but not broad calibrated-gate superiority.
