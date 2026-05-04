# Final Stage F9 Third-Scene Room and Qualitative Evidence

Date: 2026-05-04

## Decision

`FINAL_F9_THIRD_SCENE_ROOM_PASS`.

F9 adds a third non-parking scene (`room`) using the same fair protocol as F8: clean-long 9k->22k, CSEF boundary-protected 50 percent compaction, strict topology-frozen recovery 22k->26k, then independent render, image metrics, and sparse COLMAP geometry evaluation.

## W&B Runs

| scene | role | W&B run | URL |
| --- | --- | --- | --- |
| room | clean-long 9k->22k | `kqyusaoe` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kqyusaoe` |
| room | CSEF50 22k->26k | `pb1tg4p2` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/pb1tg4p2` |

Both runs kept online W&B scalar logging enabled and disabled inline image logging. Independent render and metric evaluation were run after checkpoint save.

## Room Results

| method | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | - | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| CSEF50 26k | 42,253 | 50.0% | 14.387163 | 0.414954 | 0.568281 | 0.225027 | 1.603030 | 54.642793 |

## Room Deltas

| method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSEF50 | +0.128784 | +0.014090 | -0.010638 | +0.018745 | +0.122800 | -0.799860 |

The room row passes the same conservative gate used in F8:
- triangle reduction is exactly 50 percent;
- PSNR and SSIM improve;
- LPIPS improves;
- AbsRel regression is below the allowed +0.02 margin;
- Depth MAE regression is below the allowed +0.20 margin;
- normal angle improves.

## Qualitative Evidence

Generated qualitative output:

```text
outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_montage.png
outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_manifest.json
outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_report.md
```

The montage currently covers:
- parking clean-long 22k vs CSEF70 26k;
- bonsai clean-long 22k vs CSEF50 26k;
- courtyard clean-long 22k vs CSEF50 26k.

Room is intentionally not yet included in the montage because the montage was generated before room compact recovery finished. The next montage refresh should add room and use selected representative views rather than the first test view only.

## Aggregate State

MeshSplatOpt now has:
- one strong parking anchor with 70 percent triangle reduction and all-metric gains;
- two F8 non-parking PASS scenes (`bonsai`, `courtyard`) with fair clean-long comparisons;
- one F9 third-scene PASS (`room`) with the same CSEF50 protocol.

This is a materially stronger evidence state than the previous parking-only or two-scene transfer state. The remaining paper-readiness gap is breadth and presentation: add `counter`, refresh qualitative montages with room/counter, and consolidate all independent metrics into one paper table.
