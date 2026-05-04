# Final Stage F16 - Counter Random Same-Count Control

Date: 2026-05-04

Decision: `FINAL_F16_COUNTER_RANDOM_SAME_COUNT_CONTROL_PASS_FOR_CSEF`.

## Goal

Address the reviewer risk that counter CSEF40 may simply be equivalent to deleting any
40 percent of triangles followed by recovery. This control uses the same scene, same
clean-long source checkpoint, same target triangle count, same 22k->26k strict
topology-frozen recovery budget, and the same independent evaluation protocol.

## Setup

- scene: `counter`
- clean source: `outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000`
- source iteration: `22000`
- selector: `random_same_count`
- target prune fraction: `0.40`
- seed: `20260504`
- compact model: `outputs/carnet/meshsplatopt/final_stageF16_counter_random_same_count_control/prune40/compact_model`
- recovery model: `outputs/carnet/meshsplatopt/final_stageF16_counter_random_same_count_control/prune40/recovery_model`
- W&B run: `0hlz8q0u`
- W&B URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/0hlz8q0u`

## Topology

| method | triangles | vertices | reduction |
| --- | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 155,104 | - |
| CSEF40 26k | 50,300 | 119,096 | 40.0% |
| random40 26k | 50,300 | 111,605 | 40.0% |

## Independent Metrics

| method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF40 26k | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |
| random40 26k | 13.875822 | 0.482349 | 0.485052 | 0.099779 | 0.444684 | 43.941494 |

## Deltas

| comparison | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSEF40 - clean | +0.075851 | +0.005599 | -0.001568 | +0.008546 | +0.036400 | -0.810063 |
| random40 - clean | -0.260360 | -0.030453 | +0.033003 | +0.022783 | +0.074711 | -0.345541 |
| CSEF40 - random40 | +0.336211 | +0.036052 | -0.034571 | -0.014237 | -0.038311 | -0.464522 |

## Interpretation

The random same-count control fails the clean-long gate and is substantially worse than
CSEF40 at the same triangle count. This directly supports the claim that the counter
Pareto win is not explained by arbitrary topology reduction plus recovery. It does not
complete the full selector-ablation matrix; the same random control should still be
replicated on at least courtyard and one additional public scene before a strict
NeurIPS ablation claim.

