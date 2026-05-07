# Stage ELA11 Final Selected-Scene Package

This package freezes the current best adaptive-policy row per selected scene and audits average metrics, sparse geometry, topology, per-view RGB deltas, and qualitative examples against the clean Mesh Splatting baseline.

## Promoted Average Rows

| scene | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | strict full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai | SOR10 + ELA safe | 2.838371 | 0.163376 | -0.099541 | -0.105169 | -1.032433 | -2.410058 | 10.25% | `True` |
| courtyard | SOR10 + ELA safe | 0.969368 | 0.028828 | -0.056569 | -0.104763 | -1.288431 | -2.711335 | 10.34% | `True` |
| room | QEM50 parent-rollback + ELA safe | 3.304691 | 0.050085 | -0.062170 | -0.002331 | -0.019509 | -1.824378 | 50.00% | `True` |
| counter | QEM50 parent-rollback + ELA safe | 3.157017 | 0.069925 | -0.070661 | -0.000686 | -0.008253 | -2.080537 | 50.00% | `True` |

## Per-View RGB Stress Test

Per-view rows are not the headline claim, but they expose whether gains are broad or dominated by a few views.

| scene | views | RGB full-pass views | min dPSNR | mean dPSNR | worst dLPIPS |
|---|---:|---:|---:|---:|---:|
| bonsai | 37 | 37 | 0.837978 | 2.838371 | -0.056966 |
| courtyard | 5 | 5 | 0.210857 | 0.969366 | -0.031749 |
| room | 39 | 39 | 0.453001 | 3.304691 | -0.013167 |
| counter | 30 | 30 | 1.043489 | 3.157015 | -0.034378 |

## Artifacts

- summary JSON: `outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package/final_selected_scene_summary.json`
- average CSV: `outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package/promoted_average_rows.csv`
- per-view CSV: `outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package/per_view_rgb_deltas.csv`
- qualitative gallery: `outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package/qualitative_gallery/gallery.html`
- qualitative manifest: `outputs/carnet/meshsplatopt/stageELA11_final_selected_scene_package/qualitative_gallery/gallery_manifest.md`
