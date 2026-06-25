# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `bonsai_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `37`
- repeats: `2`
- mean ms/view: `791.108688`
- mean FPS: `1.265503`
- mean render ms/view: `36.910879`
- mean adapter ms/view: `753.878722`
- adapter/render ratio: `20.421496`
- peak allocated MiB max: `17053.017`
- peak reserved MiB max: `22402.000`
- triangles: `9555533`
- vertices: `3295557`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.263404 | 817.929845 | 1.222599 | 37.074816 | 780.496840 | 17021.013 |
| 2 | 28.278639 | 764.287532 | 1.308408 | 36.746942 | 727.260604 | 17053.017 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
