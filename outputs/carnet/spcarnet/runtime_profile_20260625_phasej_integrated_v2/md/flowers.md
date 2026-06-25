# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `flowers_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `22`
- repeats: `2`
- mean ms/view: `832.619785`
- mean FPS: `1.201150`
- mean render ms/view: `32.364652`
- mean adapter ms/view: `799.868747`
- adapter/render ratio: `24.750449`
- peak allocated MiB max: `10881.523`
- peak reserved MiB max: `11932.000`
- triangles: `8509358`
- vertices: `3414899`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.501759 | 840.989026 | 1.189076 | 33.749385 | 806.774292 | 9910.001 |
| 2 | 18.133512 | 824.250544 | 1.213223 | 30.979919 | 792.963202 | 10881.523 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
