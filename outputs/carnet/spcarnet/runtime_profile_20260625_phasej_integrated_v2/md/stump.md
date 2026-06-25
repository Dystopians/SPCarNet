# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `stump_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/stump/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `16`
- repeats: `2`
- mean ms/view: `872.583973`
- mean FPS: `1.153822`
- mean render ms/view: `31.794933`
- mean adapter ms/view: `840.348024`
- adapter/render ratio: `26.384285`
- peak allocated MiB max: `9935.544`
- peak reserved MiB max: `10736.000`
- triangles: `8180134`
- vertices: `3383973`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.813416 | 800.838482 | 1.248691 | 30.322280 | 769.939916 | 8815.480 |
| 2 | 15.109271 | 944.329464 | 1.058952 | 33.267587 | 910.756132 | 9935.544 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
