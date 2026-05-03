# MeshSplatOpt Stage R14.21-R14.22 Freeze-Densify Recovery Control Report

Date: 2026-05-02

## Decision

`TOPOLOGY_RETENTION_PASS`.

R14.19-R14.20 showed that naive 2000->4000 continuation inflates `bonsai` from `2.49M` triangles to about `5.09M` triangles and does not support a snap-gain claim. R14.21-R14.22 adds an opt-in recovery schedule that freezes densification at the loaded checkpoint and skips the delayed restricted-Delaunay refresh. Under the same medium W&B budget, this changes the result from a negative control into a strong topology-retention row.

## Implementation

Added recovery-time train overrides:

- `--train_densify_until_iter`
- `--train_densify_from_iter`
- `--train_densification_interval`
- `--train_skip_restricted_delaunay`

Added train option:

- `--skip_restricted_delaunay`

This matters because setting `densify_until_iter=2000` alone still schedules restricted Delaunay at `densify_until_iter + 1000`, which caused the first R14.21 diagnostic run to stall around iteration 3000. R14.21b and R14.22 use both `--densify_until_iter 2000` and `--skip_restricted_delaunay`.

## W&B

```text
R14.21 diagnostic aborted: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gpqeybmc
R14.21b baseline freeze:  https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qdwbbpob
R14.22 snap freeze:      https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/srdr58z6
```

## Results

| row | edit | train schedule | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R14.20 baseline continuation | none | unfrozen | `5090601` | `4270293` | `15.834700584411621` | `0.33469849824905396` | `0.5714929699897766` | `0.40514114339865287` | `4.241773913061498` | `48.11943889631045` |
| R14.21b baseline freeze | none | freeze densify, skip Delaunay | `2487474` | `2478890` | `17.429750442504883` | `0.43235236406326294` | `0.5064895749092102` | `0.27106212926722306` | `2.897163412813164` | `43.347689336379396` |
| R14.22 snap freeze | `SNAP_VERTICES` | freeze densify, skip Delaunay | `2487474` | `2478890` | `17.437725067138672` | `0.4337323307991028` | `0.5067973732948303` | `0.2728521602266819` | `2.8930862576166856` | `43.570728874963045` |

R14.21b baseline freeze minus R14.20 unfrozen baseline:

| triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `-2603127` | `-1791403` | `1.5950498580932617` | `0.09765386581420898` | `-0.0650033950805664` | `-0.1340790141314298` | `-1.3446105002483337` | `-4.771749559931052` |

R14.22 snap freeze minus R14.21b baseline freeze:

| triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0` | `0.007974624633789062` | `0.0013799667358398438` | `0.0003077983856201172` | `0.0017900309594588437` | `-0.004077155196478444` | `0.22303953858364878` |

## Interpretation

The main R14.21-R14.22 finding is not that the current snap selector is a large standalone win. The current snap row is a mixed, very small delta over the freeze baseline: PSNR and SSIM improve, LPIPS and normal proxy worsen slightly, and depth MAE improves slightly.

The important result is the topology-retention schedule. Compared with the unfrozen medium baseline, freezing densification at the checkpoint and skipping the delayed Delaunay refresh cuts triangles by `51.13594642361481%` while improving all independent render metrics and both sparse geometry proxies. This makes topology retention a mandatory recovery schedule for any full-budget MeshSplatOpt claim.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_21b_bonsai_baseline_freeze_densify_skip_delaunay_2000to4000/real_tiny_recovery_report.json`
- `outputs/carnet/meshsplatopt/stageR14_21b_bonsai_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_21b_bonsai_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`
- `outputs/carnet/meshsplatopt/stageR14_22_bonsai_snap_freeze_densify_skip_delaunay_2000to4000/real_tiny_recovery_report.json`
- `outputs/carnet/meshsplatopt/stageR14_22_bonsai_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_22_bonsai_snap_freeze_densify_skip_delaunay_2000to4000/recovery_model/geometry_eval_colmap/iter_4000_max500.json`

## Next Gate

Use the freeze-densify/skip-Delaunay schedule as the default for R15 full or multi-scene recovery. A full-budget run is now justified for the schedule, but the current snap selector still needs a stronger proposal signal or multi-scene evidence before being framed as the main method improvement.
