# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `kitchen_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `35`
- repeats: `2`
- mean ms/view: `1010.527740`
- mean FPS: `0.993914`
- mean render ms/view: `42.970477`
- mean adapter ms/view: `967.213790`
- adapter/render ratio: `22.502615`
- peak allocated MiB max: `17167.559`
- peak reserved MiB max: `22996.000`
- triangles: `9512393`
- vertices: `2391146`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.703616 | 1077.246179 | 0.928293 | 43.153476 | 1033.716445 | 17143.882 |
| 2 | 33.033326 | 943.809301 | 1.059536 | 42.787478 | 900.711136 | 17167.559 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
