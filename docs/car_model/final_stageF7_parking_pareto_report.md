# Final Stage F7 Parking Pareto Report

Date: 2026-05-04

## Decision

`FINAL_F7_PARKING_PARETO_PASS`.

The parking compact Pareto interface is implemented, and the first non-area CSEF boundary-protected 70 percent compaction run is independently validated at 26k. It keeps the same topology as the R53 area70 compact baseline, beats clean22k on render and sparse geometry, and slightly improves over R53 on PSNR, LPIPS, AbsRel, Depth MAE, and normal angle.

## Implemented

```text
scripts/car_model/final_run_parking_compact_pareto.py
scripts/car_model/final_collect_parking_compact_pareto.py
scripts/car_model/meshsplatopt_eval_render_metrics_single_iteration.py
outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/
```

The manifest covers 28 planned jobs: 4 selectors (`area_smallest`, `csef_low_evidence_boundary_protected`, `pareto_area_csef`, `random_same_count`) times 7 pruning fractions (50/60/65/70/75/80/90 percent). The metrics command now evaluates exactly the requested iteration, avoiding stale `ours_*` render directories left by copied checkpoints.

## Validated CSEF70 Run

```text
selector: csef_low_evidence_boundary_protected
prune fraction: 70%
load iteration: 22000
final iteration: 26000
triangles: 2,564,473
vertices: 1,661,616
topology_unchanged: true
output: outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/csef_low_evidence_boundary_protected/prune70/recovery_model
W&B run: oqpkykcw
W&B URL: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/oqpkykcw
```

## Results

| row | selector | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean22k | baseline | 8,548,242 | 18.479990 | 0.634623 | 0.346913 | 0.082177 | 1.868398 | 45.108437 |
| R53.01 | area_smallest 70% | 2,564,473 | 18.705738 | 0.647807 | 0.338492 | 0.079555 | 1.853751 | 44.261391 |
| F7.csef70 | CSEF protected 70% | 2,564,473 | 18.706079 | 0.647764 | 0.338282 | 0.079404 | 1.852816 | 44.204497 |

Against clean22k, F7.csef70 improves PSNR by `+0.226089`, SSIM by `+0.013141`, LPIPS by `-0.008631`, AbsRel by `-0.002773`, Depth MAE by `-0.015582`, and normal angle by `-0.903940`, while reducing triangles by `70.0%`.

Against R53 at identical topology, F7.csef70 changes PSNR by `+0.000341`, SSIM by `-0.000043`, LPIPS by `-0.000210`, AbsRel by `-0.000151`, Depth MAE by `-0.000935`, and normal angle by `-0.056894`. This is a small but real Pareto refinement: the CSEF selector preserves render quality and improves LPIPS plus all sparse geometry metrics at the same triangle count.

## Gate

PASS condition: include at least one compact row that beats clean22k on render and geometry while reducing triangles by at least 50 percent.

Validated compact rows satisfying the gate: `R53.01`, `R55.01`, and `F7.csef70`.

## Next Use

Use CSEF70 as the first final-method parking checkpoint when testing cross-scene transfer. Keep R53 area70 as the strongest area-only control, and run `pareto_area_csef` plus `random_same_count` only where the scene has a matched clean-long baseline.
