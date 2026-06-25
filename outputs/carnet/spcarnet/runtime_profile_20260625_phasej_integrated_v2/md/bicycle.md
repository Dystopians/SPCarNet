# Integrated Phase-J Runtime Profile

This profile measures renderer forward + Phase-J ELA `adapt_frame` in one process.
It excludes PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- label: `bicycle_phasej_integrated_v2`
- model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`
- split: `test`
- iteration: `26000`
- views: `25`
- repeats: `2`
- mean ms/view: `1116.839100`
- mean FPS: `0.899391`
- mean render ms/view: `31.260607`
- mean adapter ms/view: `1085.184300`
- adapter/render ratio: `34.705190`
- peak allocated MiB max: `10818.907`
- peak reserved MiB max: `11714.000`
- triangles: `8309749`
- vertices: `3318902`

## Repeats

| repeat | elapsed sec | ms/view | FPS | render ms/view | adapter ms/view | peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.784710 | 1191.388398 | 0.839357 | 31.385178 | 1159.523576 | 10762.170 |
| 2 | 26.057245 | 1042.289802 | 0.959426 | 31.136036 | 1010.845024 | 10818.907 |

## Scope Note

- Uses fresh renderer RGB/depth tensors for each target view.
- Uses existing train/support evidence artifacts for support residuals.
- Does not write adapted PNGs, so wall-clock image export remains a separate deployment concern.
