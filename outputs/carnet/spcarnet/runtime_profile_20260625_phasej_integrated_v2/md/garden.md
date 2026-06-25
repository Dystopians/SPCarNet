# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `garden_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `24`
- repeats: `2`
- mean ms/view: `849.438973`
- mean FPS: `1.181303`
- mean render ms/view: `35.903718`
- mean adapter ms/view: `813.106209`
- adapter/render ratio: `22.633026`
- peak allocated MiB max: `11966.776`
- peak reserved MiB max: `17046.000`
- triangles: `11166587`
- vertices: `3315236`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.581042 | 899.210101 | 1.112087 | 36.363013 | 862.308452 | 11177.905 |
| 2 | 19.192028 | 799.667846 | 1.250519 | 35.444424 | 763.903967 | 11966.776 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
