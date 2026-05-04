# Final Stage F24 - Room QEM No-Freeze Control

Decision: `FINAL_F24_ROOM_QEM_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`.

## Goal

Replicate the F18 counter no-freeze control on a second scene and a stronger QEM compact operator. The control starts from the accepted `room` Open3D QEM50 compact checkpoint and uses the same `22000 -> 26000` recovery budget as the frozen QEM50 row, but deliberately omits `--freeze_topology_updates`.

## Implementation

- source compact: `outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/compact_model`
- recovery: `outputs/carnet/meshsplatopt/final_stageF24_room_qem_no_freeze_control/prune50/recovery_model`
- W&B: `byjyx9zx`
- deliberate control: omitted `--freeze_topology_updates`
- start topology: `42,253` triangles, `84,806` vertices
- final topology: `20,742` triangles, `46,865` vertices

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| QEM50 frozen 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |
| QEM50 no-freeze 26k | 20,742 | 13.789439 | 0.399147 | 0.567857 | 0.212804 | 1.497902 | 55.443601 |

## Finding

Removing strict topology freeze causes the room QEM50 compact checkpoint to drift from `42,253` to `20,742` triangles. This is not a benign continuation: it loses badly to the frozen QEM50 row on every independent render and sparse-geometry metric.

Relative to frozen QEM50, no-freeze changes PSNR by `-1.271751`, SSIM by `-0.081935`, LPIPS by `+0.051052`, AbsRel by `+0.031675`, Depth MAE by `+0.152681`, and normal by `+0.542822` degrees. It also falls below clean-long on PSNR, SSIM, AbsRel, Depth MAE, and normal.

## Gate

FAIL control, supports strict topology freeze. Together with F18 counter no-freeze, this shows that `--skip_restricted_delaunay` alone is not enough and that the final method's strict topology-frozen recovery contract is load-bearing.
