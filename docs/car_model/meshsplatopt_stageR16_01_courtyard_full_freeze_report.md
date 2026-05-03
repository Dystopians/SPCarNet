# MeshSplatOpt Stage R16.01 Courtyard Full Freeze Report

Date: 2026-05-03

## Decision

`FULL_SCHEDULE_PASS_SINGLE_PUBLIC_SCENE`.

R16.01 runs the freeze-densify/skip-Delaunay schedule from `courtyard` iteration 2000 to iteration 7000 with online W&B logging. This is the first full-budget validation of the R15 schedule on a public scene.

## W&B

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z2i5ndyu
```

## Results

| row | iter | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `2000` | `410254` | `444301` | `14.946162223815918` | `0.4387754499912262` | `0.5924432873725891` | `0.3547996069696563` | `3.647069967658135` | `35.32471188743233` |
| R15.01 freeze medium | `4000` | `410254` | `444301` | `17.819637298583984` | `0.5783027410507202` | `0.46039170026779175` | `0.24305365085457115` | `2.6916776369705433` | `37.96788445741664` |
| R16.01 freeze full | `7000` | `410254` | `444301` | `18.321130752563477` | `0.5942807793617249` | `0.4400215744972229` | `0.17145306790424905` | `2.0675095533889585` | `37.57569562265334` |

R16.01 full minus baseline 2000:

| triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0` | `3.3749685287475586` | `0.15550532937049866` | `-0.1524217128753662` | `-0.18334653906540725` | `-1.5795604142691765` | `2.250983735221011` |

R16.01 full minus R15.01 medium:

| triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0` | `0.5014934539794922` | `0.01597803831100464` | `-0.020370125770568848` | `-0.0716005829503221` | `-0.6241680835815848` | `-0.3921888347633004` |

## Interpretation

The full-budget row preserves checkpoint topology exactly and continues improving render and depth proxy metrics beyond the medium row. This is important because it shows that freeze-densify recovery is not merely a short-budget artifact.

The remaining weakness is the sparse normal proxy: R16.01 is better than R15.01 medium but still worse than the 2000-iteration baseline. This should be treated as a failure mode or a missing geometry regularizer, not ignored.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/real_tiny_recovery_report.json`
- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR16_01_courtyard_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model/geometry_eval_colmap/iter_7000_max500.json`

## Next Gate

Run a full-budget row on `bonsai` or `parking_phone_tiny`, then add a geometry/normal-aware proposal or recovery term. The schedule is now strong enough for a paper methods section, but the current selector and normal behavior are not yet strong enough for a final top-conference claim.
