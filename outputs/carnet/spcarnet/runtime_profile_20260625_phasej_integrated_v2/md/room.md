# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `room_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `39`
- repeats: `2`
- mean ms/view: `1012.581573`
- mean FPS: `0.989955`
- mean render ms/view: `37.773754`
- mean adapter ms/view: `974.120425`
- adapter/render ratio: `25.772492`
- peak allocated MiB max: `17703.596`
- peak reserved MiB max: `24498.000`
- triangles: `10938652`
- vertices: `2777389`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.426953 | 1062.229563 | 0.941416 | 38.594884 | 1022.731312 | 17052.478 |
| 2 | 37.554410 | 962.933583 | 1.038493 | 36.952624 | 925.509538 | 17703.596 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
