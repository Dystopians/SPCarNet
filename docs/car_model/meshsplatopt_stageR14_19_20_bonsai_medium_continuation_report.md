# MeshSplatOpt Stage R14.19-R14.20 Bonsai Medium Continuation Report

Date: 2026-05-02

## Decision

`MEDIUM_CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN`.

This stage runs a W&B-logged medium continuation on `bonsai` from iteration 2000 to 4000 for both:

- R14.19: accepted non-delete `SNAP_VERTICES` checkpoint.
- R14.20: unedited baseline continuation.

The medium continuation improves both rows versus the 2000iter baseline, but the unedited baseline continuation is stronger on render and sparse-depth metrics. The snap row is only better on the sparse normal proxy.

## W&B

```text
snap:     https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fjzy6lun
baseline: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gxeskhta
```

## Results

| row | edit | iteration | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R14.19 snap continuation | `SNAP_VERTICES` | `4000` | `5090526` | `4270548` | `15.81759262084961` | `0.33459141850471497` | `0.5731096863746643` | `0.40904864176963485` | `4.261201179402033` | `47.83674765098326` |
| R14.20 baseline continuation | none | `4000` | `5090601` | `4270293` | `15.834700584411621` | `0.33469849824905396` | `0.5714929699897766` | `0.40514114339865287` | `4.241773913061498` | `48.11943889631045` |

Snap minus baseline continuation:

| triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `-75` | `255` | `-0.01710796356201172` | `-0.00010707974433898926` | `0.0016167163848876953` | `0.00390749837098198` | `0.019427266340535047` | `-0.2826912453271897` |

## Interpretation

The medium control confirms the R14.18 short-control conclusion: the current checkpoint-statistics snap selector is safe and trainable, but it does not produce an equal-budget quality improvement on `bonsai`.

The topology growth is also a blocker. Both medium rows grow from `2487474` triangles at 2000iter to about `5.09M` triangles at 4000iter, so a full 7000iter continuation would be a poor use of GPU time unless paired with a topology-retention schedule or a stronger edit selector.

## Full-Run Decision

Do not launch full-budget R15 from this snap selector. The next R15 candidate should first add one of:

- a topology-retention schedule during recovery;
- a render-residual selector rather than pure area outlier statistics;
- a gate that requires expected equal-budget benefit, not only non-regression safety.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_19_bonsai_snap_medium_continuation_2000step/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_19_bonsai_snap_medium_continuation_2000step/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_19_bonsai_snap_medium_continuation_2000step/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
- `outputs/carnet/meshsplatopt/stageR14_20_bonsai_baseline_medium_continuation_2000step/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_20_bonsai_baseline_medium_continuation_2000step/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_20_bonsai_baseline_medium_continuation_2000step/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
