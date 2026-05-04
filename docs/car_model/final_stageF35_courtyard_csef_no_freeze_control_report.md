# Final Stage F35 - Courtyard CSEF No-Freeze Control

Date: 2026-05-04

Decision: `FINAL_F35_COURTYARD_CSEF_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`.

## Goal

Replicate the strict-topology-freeze control on a fourth final-package scene. The run
starts from the accepted `courtyard` CSEF50 compact checkpoint and uses the same
`22000->26000` recovery budget as the frozen CSEF50 main row, but deliberately omits
`--freeze_topology_updates` while keeping online W&B logging.

## Run

| field | value |
| --- | --- |
| source compact checkpoint | `outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot/courtyard/csef_low_evidence_boundary_protected/prune50/compact_model` |
| recovery checkpoint | `outputs/carnet/meshsplatopt/final_stageF35_courtyard_csef_no_freeze_control/csef50/recovery_model` |
| schedule | `22000->26000` |
| W&B | `3bk0z0vs` |
| deliberate control | omitted `--freeze_topology_updates`; kept `--skip_restricted_delaunay` |
| start topology | `838,742` triangles / `1,643,471` vertices |
| final topology | `1,317,435` triangles / `1,689,410` vertices |

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 frozen 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| CSEF50 no-freeze 26k | 1,317,435 | 8.646640 | 0.111272 | 0.675011 | 0.513762 | 5.539791 | 42.710115 |

## Deltas

| comparison | dTriangles | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-freeze - clean | -360,049 | -3.456868 | -0.185376 | +0.105703 | +0.159114 | +1.710747 | +1.888466 |
| no-freeze - frozen CSEF50 | +478,693 | -3.909169 | -0.227001 | +0.129934 | +0.191529 | +1.931359 | +1.879958 |

## Finding

`--skip_restricted_delaunay` alone does not preserve the compact-recovery contract. On
`courtyard`, omitting strict topology freeze allows the model to drift from `838,742`
to `1,317,435` triangles, while still failing badly on every independent render and
sparse-depth metric. This is worse than the frozen CSEF50 row by `3.909169` PSNR,
`0.227001` SSIM, `0.129934` LPIPS, `0.191529` AbsRel, `1.931359` Depth MAE, and
`1.879958` normal degrees.

Together with bonsai, room, and counter no-freeze controls, F35 establishes strict
topology freezing as a replicated load-bearing mechanism across four final-package
scenes and both CSEF/QEM compact operators.
