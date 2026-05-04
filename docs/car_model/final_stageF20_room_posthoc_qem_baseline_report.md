# Final Stage F20 - Room Posthoc QEM Baseline

Decision: `FINAL_F20_ROOM_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_AREA50_ON_RENDER_DEPTH`.

## Goal

Run a real posthoc simplification baseline instead of leaving QEM as a missing reviewer-risk row. The baseline applies Open3D quadric decimation to the clean-long `room` checkpoint, transfers checkpoint attributes by nearest neighbors, then uses the same strict topology-frozen `22000 -> 26000` recovery budget as area50, CSEF50, and random50.

## Implementation

- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`
- simplifier: Open3D `simplify_quadric_decimation`
- source: `outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000`
- compact: `outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/compact_model`
- recovery: `outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/recovery_model`
- W&B: `9wri3owt`
- topology: `84,506 -> 42,253` triangles, `149,450 -> 84,806` vertices
- validation: `degenerate_face_count=0`, `invalid_index_count=0`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| CSEF50 26k | 42,253 | 14.387163 | 0.414954 | 0.568281 | 0.225027 | 1.603030 | 54.642793 |
| area50 26k | 42,253 | 14.844683 | 0.461875 | 0.530461 | 0.185703 | 1.353216 | 54.615295 |
| random50 26k | 42,253 | 13.428182 | 0.345278 | 0.609467 | 0.272092 | 1.873476 | 54.469912 |
| Open3D QEM50 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |

## Finding

Open3D QEM50 plus strict topology-frozen recovery is the strongest `room` row on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE. Relative to clean-long, it improves PSNR by `+0.802811`, SSIM by `+0.080218`, LPIPS by `-0.062114`, AbsRel by `-0.025153`, and Depth MAE by `-0.135009` while halving triangles. It also improves normal versus clean by `-0.541874` degrees, although area50 remains slightly better on normal by `0.285484` degrees.

This is a strong baseline, not a weak strawman. The paper should not claim that MeshSplatOpt simply beats classical simplification on this scene. The better framing is that the strict fixed-topology recovery framework can absorb multiple compact operators, and that QEM is a strong collapse-style operator for `room`.

## Gate

PASS. The posthoc QEM row removes a major missing-baseline risk for one scene and upgrades the room main-table row. It also creates a new requirement: replicate QEM beyond `room` before making broad claims against classical simplification.
