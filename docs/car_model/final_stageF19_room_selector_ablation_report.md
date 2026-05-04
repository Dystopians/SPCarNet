# Final Stage F19 - Room Selector Ablation

Decision: `FINAL_F19_ROOM_SELECTOR_ABLATION_PASS_AREA50_BEST_RANDOM_FAIL`.

## Goal

Add a third public-scene selector ablation after counter and courtyard. The comparison uses the same `room` clean-long checkpoint and the same `50%` triangle target as the accepted CSEF50 room row.

## Runs

- scene: `mipnerf360/room`
- clean source: `outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000`
- area50 compact: `outputs/carnet/meshsplatopt/final_stageF19_room_selector_ablation/area_smallest/prune50/compact_model`
- area50 recovery: `outputs/carnet/meshsplatopt/final_stageF19_room_selector_ablation/area_smallest/prune50/recovery_model`
- random50 compact: `outputs/carnet/meshsplatopt/final_stageF19_room_selector_ablation/random_same_count/prune50/compact_model`
- random50 recovery: `outputs/carnet/meshsplatopt/final_stageF19_room_selector_ablation/random_same_count/prune50/recovery_model`
- schedule: `22000 -> 26000`
- topology flags: `--skip_restricted_delaunay --freeze_topology_updates`
- W&B area50: `eagvu7em`
- W&B random50: `p0vxzf01`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| CSEF50 26k | 42,253 | 14.387163 | 0.414954 | 0.568281 | 0.225027 | 1.603030 | 54.642793 |
| area50 26k | 42,253 | 14.844683 | 0.461875 | 0.530461 | 0.185703 | 1.353216 | 54.615295 |
| random50 26k | 42,253 | 13.428182 | 0.345278 | 0.609467 | 0.272092 | 1.873476 | 54.469912 |

## Finding

Area50 is the new best room compact-recovery row. Relative to clean-long, it improves PSNR by `+0.586304`, SSIM by `+0.061011`, LPIPS by `-0.048458`, AbsRel by `-0.020579`, Depth MAE by `-0.127014`, and Normal by `-0.827358` degrees while halving triangles.

At the same `42,253` triangle count, random50 fails badly: relative to area50 it loses `1.416501` PSNR, `0.116597` SSIM, worsens LPIPS by `0.079006`, worsens AbsRel by `0.086389`, and worsens Depth MAE by `0.520260`. This makes the third selector-control scene consistent with counter and courtyard: arbitrary same-count pruning is not sufficient.

## Gate

PASS. The main package should promote room from CSEF50 to area50. The selector story remains honest: area-based compacting is strongest on counter and room, CSEF is slightly more geometry-balanced on courtyard, and random same-count pruning fails across all three controlled scenes.
