# Stage ELA10 Room Strict Multi-Axis Repair Report

Date: 2026-05-06

## Decision

`ROOM_STRICT_MULTIAXIS_SOLVED_GLOBAL_SELECTED_SCENES_NOT_YET`

Stage ELA10 is the first branch that strictly beats the clean Mesh Splatting room baseline on RGB quality, sparse geometry proxies, and triangle count at the same time. It is not yet a complete selected-scene claim because the fixed policy still needs to be replicated on bonsai, courtyard, and counter.

## Baseline

Room clean Mesh Splatting baseline:

- model: `outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000`
- method: `ours_9000`
- PSNR: `26.217100`
- SSIM: `0.889372`
- LPIPS: `0.135088`
- AbsRel: `0.033025`
- Depth MAE: `0.384271`
- normal mean angle: `49.732398`
- triangles: `196057`
- vertices: `274368`

## Fixed Policy

The promoted ELA10 room policy is no longer a scene-specific parameter scan:

1. start from the strongest clean9000 Mesh Splatting checkpoint;
2. apply Open3D QEM decimation with `target_fraction=0.5`, meaning 50% of triangles remain and about 50% are deleted;
3. build a train-only sparse depth sentinel cache by comparing the compact checkpoint with the clean parent;
4. run topology-frozen recovery from 9000 to 12000 with sparse COLMAP depth, sparse parent rollback, checkpoint geometry anchor, and parent render rollback;
5. apply the train-only ELA safe appearance adapter on the recovered 12000 checkpoint.

This policy uses no test GT for parameter selection. The ELA stage logs to W&B and only changes render outputs, while the recovery checkpoint supplies the geometry/topology win.

## Key Runs

- QEM50 sparse teacher rollback recovery: W&B `wdvuvkd7`.
- CSEF adaptive teacher rollback recovery: W&B `l105dps8`.
- QEM30 sparse teacher rollback recovery: W&B `yhquce04`.
- QEM20 sparse teacher rollback recovery: W&B `47hlyeb4`.
- QEM50 compact ELA safe: W&B `40omjhr6`.
- QEM30 compact ELA safe: W&B `8nbk6hhg`.
- QEM50 sparse parent rollback recovery: W&B `7cmz8vhv`.
- QEM50 sparse parent rollback + ELA safe: W&B `9t01dwd8`.

## Strict Results vs Clean Mesh Splatting Room

| method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | full |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| QEM50 sparse teacher rollback | 0.153080 | 0.003044 | -0.002497 | 0.000052 | 0.001910 | -0.213866 | 50.00% | False |
| QEM50 compact + ELA safe | 2.820644 | 0.043782 | -0.053010 | 0.000033 | -0.001857 | -0.088153 | 50.00% | False |
| QEM30 compact + ELA safe | 2.773758 | 0.044134 | -0.052992 | 0.000074 | -0.000198 | -0.044555 | 30.00% | False |
| QEM50 sparse parent rollback | 0.692919 | 0.013745 | -0.015990 | -0.002331 | -0.019509 | -1.824378 | 50.00% | True |
| QEM50 sparse parent rollback + ELA safe | 3.304691 | 0.050085 | -0.062170 | -0.002331 | -0.019509 | -1.824378 | 50.00% | True |

## Lessons

- Compact-only QEM is not enough. It can reduce triangles and sometimes improve PSNR/depth MAE/normal, but sparse AbsRel can remain slightly worse.
- ELA alone is not enough for the strict claim. It can repair RGB dramatically, but it inherits the compact checkpoint geometry.
- The sparse parent rollback loss is the decisive ELA10 change. It targets train-only sentinel points where compact geometry regresses against the clean parent, while the checkpoint anchor and parent render rollback keep the recovery from drifting.
- QEM30 and QEM20 did not beat QEM50. The issue was not simply excessive pruning; the recovery objective needed parent-aware geometry rollback.
- The current result is strong but still local to room. The same fixed policy must be run on the remaining selected scenes before any paper claim says "selected-scene full win."

## Audit Artifacts

- strict audit report: `docs/car_model/stageELA9_strict_multiaxis_audit_report.md`
- JSON: `outputs/carnet/meshsplatopt/stageELA9_strict_multiaxis_audit/strict_multiaxis_audit.json`
- selected-scene CSV: `outputs/carnet/meshsplatopt/stageELA9_strict_multiaxis_audit/selected_scene_strict_rows.csv`
