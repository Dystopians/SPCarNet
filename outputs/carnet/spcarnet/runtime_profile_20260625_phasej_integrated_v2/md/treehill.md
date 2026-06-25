# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `treehill_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `18`
- repeats: `2`
- mean ms/view: `849.546097`
- mean FPS: `1.194664`
- mean render ms/view: `30.359684`
- mean adapter ms/view: `818.822917`
- adapter/render ratio: `26.898832`
- peak allocated MiB max: `10301.594`
- peak reserved MiB max: `11932.000`
- triangles: `8402362`
- vertices: `3419320`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.146024 | 952.556890 | 1.049806 | 31.190557 | 920.933711 | 9217.882 |
| 2 | 13.437635 | 746.535305 | 1.339521 | 29.528812 | 716.712123 | 10301.594 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
