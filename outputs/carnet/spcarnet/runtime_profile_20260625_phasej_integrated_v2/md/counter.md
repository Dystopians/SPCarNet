# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `counter_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `30`
- repeats: `2`
- mean ms/view: `1134.619511`
- mean FPS: `0.883180`
- mean render ms/view: `45.699373`
- mean adapter ms/view: `1088.201614`
- adapter/render ratio: `23.808257`
- peak allocated MiB max: `17327.615`
- peak reserved MiB max: `21058.000`
- triangles: `9644247`
- vertices: `2478825`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.586799 | 1186.226620 | 0.843009 | 45.872490 | 1139.632174 | 17118.756 |
| 2 | 32.490372 | 1083.012401 | 0.923350 | 45.526255 | 1036.771053 | 17327.615 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
