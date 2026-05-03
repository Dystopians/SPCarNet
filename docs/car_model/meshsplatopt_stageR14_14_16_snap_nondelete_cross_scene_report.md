# MeshSplatOpt Stage R14.14-R14.16 Snap Non-Delete Cross-Scene Report

Date: 2026-05-02

## Decision

`PASS_DIAGNOSTIC_CROSS_SCENE`.

The real-checkpoint non-delete `SNAP_VERTICES` proposal now passes render-backed gates on three scenes: `parking_phone_tiny`, `bonsai`, and `courtyard`. This removes the earlier R14 blocker that all real checkpoint edits were effectively delete/fill path validation.

## Method

The selector reads a saved Mesh Splatting checkpoint, computes triangle areas from checkpoint tensors, selects the strongest high-percentile area outlier, and writes a `SNAP_VERTICES` edit that moves the selected triangle vertices toward their centroid by `shrink_factor = 0.25`.

The edit is accepted only after model materialization, independent rendering, image metrics, and sparse COLMAP geometry proxy evaluation.

## Selection Results

| stage | scene | selected face | area before | area after | area reduction | max displacement | candidates |
|---|---|---:|---:|---:|---:|---:|---:|
| R14.14 | `parking_phone_tiny` | `727102` | `247.02622985839844` | `15.439142227172852` | `231.5870876312256` | `12.383612632751465` | `784` |
| R14.15 | `bonsai` | `2462659` | `164.05824279785156` | `10.253642082214355` | `153.8046007156372` | `10.094804763793945` | `2488` |
| R14.16 | `courtyard` | `404443` | `873.2474365234375` | `54.57794952392578` | `818.6694869995117` | `23.436288833618164` | `411` |

## Render-Backed Gate Results

| stage | scene | status | triangles delta | vertices delta | PSNR delta | SSIM delta | LPIPS delta | AbsRel delta | Depth MAE delta | normal deg delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R14.14 | `parking_phone_tiny` | `PASS` | `0` | `0` | `0.00000286102294921875` | `-0.0000012516975402832031` | `-0.000002086162567138672` | `0.0` | `0.0` | `0.00000000036786929058507667` |
| R14.15 | `bonsai` | `PASS` | `0` | `0` | `-0.00019073486328125` | `-0.000013679265975952148` | `-0.00005561113357543945` | `0.0` | `0.0` | `-0.0000019103464410363813` |
| R14.16 | `courtyard` | `PASS` | `0` | `0` | `-0.005673408508300781` | `0.000041097402572631836` | `0.0000642538070678711` | `0.0` | `0.0` | `-0.00000014456123409445354` |

All three gates remain comfortably inside the acceptance thresholds:

- max PSNR drop: `0.02`
- max SSIM drop: `0.002`
- max LPIPS increase: `0.005`
- max AbsRel increase: `0.02`
- max Depth MAE increase: `0.1`
- max normal-angle increase: `1.0`

## Artefacts

Selection reports:

- `outputs/carnet/meshsplatopt/stageR14_14_parking_snap_outlier_nondelete/area_outlier_snap_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR14_15_bonsai_snap_outlier_nondelete/area_outlier_snap_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR14_16_courtyard_snap_outlier_nondelete/area_outlier_snap_selection_report.json`

Gate reports:

- `outputs/carnet/meshsplatopt/stageR14_14_parking_snap_outlier_nondelete/gate/render_backed_checkpoint_gate_report.json`
- `outputs/carnet/meshsplatopt/stageR14_15_bonsai_snap_outlier_nondelete/gate/render_backed_checkpoint_gate_report.json`
- `outputs/carnet/meshsplatopt/stageR14_16_courtyard_snap_outlier_nondelete/gate/render_backed_checkpoint_gate_report.json`

Code:

- `scripts/car_model/meshsplatopt_select_checkpoint_area_outlier_snap_edit.py`

## Interpretation

This is a diagnostic pass, not a final benchmark row. It proves that MeshSplatOpt can propose, materialize, and validate a real non-delete geometry edit directly from checkpoint evidence across three scenes. It does not yet prove equal-budget training gains from non-delete edits.

The next high-value step is to combine this selector with W&B-logged recovery training on a public scene, then compare equal-budget results against the current branch and Stage35-style baselines.
