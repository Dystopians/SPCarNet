# Final Stage F10 - Fourth-Scene Counter Evidence

Date: 2026-05-04

## Goal

Add a fourth public-scene validation point for MeshSplatOpt/SPCarNet using the `counter`
scene, with fair clean-long comparison, W&B-logged recovery, independent rendering,
independent image metrics, and COLMAP-aligned geometry evaluation.

## Scene

- scene: `counter`
- dataset: `/data/peilincai/mesh_datasets/mipnerf360/counter`
- images: `images_4`
- resolution: `4`
- clean baseline: `outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000`
- clean iteration: `22000`

## W&B Runs

- clean-long 9k->22k: `jl5vtp4m`
- CSEF50 22k->26k: `58od8x2f`
- CSEF50 extended 26k->30k: `erjis9bc`
- CSEF40 initial failed launch, missing copied compact checkpoint: `ag6wtjwh`
- CSEF40 retry 22k->26k: `glzzth4b`

## Independent Results

| method | iteration | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long | 22000 | 83,834 | - | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF50 | 26000 | 41,917 | 50.0% | 14.077559 | 0.498974 | 0.468391 | 0.094731 | 0.438932 | 43.823390 |
| CSEF50 extended | 30000 | 41,917 | 50.0% | 14.099902 | 0.485554 | 0.479640 | 0.092779 | 0.431583 | 44.029069 |
| CSEF40 | 26000 | 50,300 | 40.0% | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |

## Deltas vs Clean-Long

| method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSEF50 26k | -0.058622 | -0.013827 | +0.016342 | +0.017735 | +0.068959 | -0.463645 |
| CSEF50 30k | -0.036280 | -0.027248 | +0.027591 | +0.015783 | +0.061610 | -0.257966 |
| CSEF40 26k | +0.075851 | +0.005599 | -0.001568 | +0.008546 | +0.036400 | -0.810063 |

## Decision

`FINAL_F10_FOURTH_SCENE_COUNTER_PARETO_PASS`.

The counter scene exposes the compression limit more clearly than bonsai, courtyard,
and room. The 50 percent point remains useful but is not the final recommended
setting for this scene because SSIM misses the strict `-0.01` gate by `0.003827`,
although PSNR, LPIPS, AbsRel, Depth MAE, and Normal remain within their gates. The
30k extension did not fix that issue and is rejected because SSIM and LPIPS both
worsened.

The 40 percent CSEF point is the recommended counter operating point. It removes
33,534 triangles while improving PSNR, SSIM, LPIPS, and Normal against the fair
clean-long baseline; AbsRel and Depth MAE regress mildly but stay inside the same
geometry tolerance used by the cross-scene gate.

## Evidence Paths

- CSEF50 recovery: `outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune50/recovery_model`
- CSEF40 recovery: `outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune40/recovery_model`
- CSEF40 renders: `outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune40/recovery_model/test/ours_26000`
- CSEF40 geometry: `outputs/carnet/meshsplatopt/final_stageF10_fourth_scene_counter/csef_low_evidence_boundary_protected/prune40/recovery_model/geometry_eval_colmap/iter_26000_max500.json`

