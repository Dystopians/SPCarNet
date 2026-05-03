# MeshSplatOpt Stage R14.12 Courtyard Area-Outlier Diagnostic Report

Date: 2026-05-02

## Gate

`PASS_DIAGNOSTIC`.

R14.12 applies the checkpoint area-outlier selector and render-backed gate to ETH3D `courtyard`, a third scene after `parking_phone_tiny` and `bonsai`.

This remains a posthoc edit-gate diagnostic, not a W&B medium recovery run.

## Selector

| field | value |
|---|---:|
| triangles | `410254` |
| vertices | `444301` |
| median area | `0.007861965335905552` |
| max area | `873.2474365234375` |
| 99.9 percentile threshold | `271.4895324707031` |
| selected face | `404443` |
| selected area | `873.2474365234375` |

## Render-Backed Gate

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| triangles | `410254` | `410253` | `-1` |
| vertices | `444301` | `444301` | `0` |
| PSNR | `14.946162223815918` | `14.94556713104248` | `-0.0005950927734375` |
| SSIM | `0.4387754499912262` | `0.4387872815132141` | `0.000011831521987915039` |
| LPIPS | `0.5924432873725891` | `0.5925152897834778` | `0.00007200241088867188` |
| AbsRel | `0.3547996069696563` | `0.3547996069696563` | `0.0` |
| Depth MAE | `3.647069967658135` | `3.647069967658135` | `0.0` |
| normal mean deg | `35.32471188743233` | `35.32471188743233` | `0.0` |

## Decision

`PASS_DIAGNOSTIC`.

The third-scene checkpoint-statistics edit is accepted by the render-backed gate. Across `parking_phone_tiny`, `bonsai`, and `courtyard`, conservative area-outlier deletion is consistently render/geometry-safe. The effect is intentionally tiny, so this supports infrastructure and safety rather than the final repair-quality claim.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_12_courtyard_area_outlier_diagnostic/selected_edit.json`
- `outputs/carnet/meshsplatopt/stageR14_12_courtyard_area_outlier_diagnostic/area_outlier_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR14_12_courtyard_area_outlier_diagnostic/gate/render_backed_checkpoint_gate_report.json`
