# MeshSplatOpt Stage R14.11 Bonsai Area-Outlier Diagnostic Report

Date: 2026-05-02

## Gate

`PASS_DIAGNOSTIC`.

R14.11 applies the automatic checkpoint area-outlier selector to a second public scene, Mip-NeRF 360 `bonsai`, and validates the selected edit with the render-backed checkpoint gate.

This is a second-scene edit-gate diagnostic, not a second W&B medium recovery run.

## Selector

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_select_checkpoint_area_outlier_edit.py \
  --checkpoint_path outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt \
  --output_dir outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic \
  --top_k 1 \
  --min_area_ratio_to_median 1000 \
  --min_percentile 99.9
```

Selection:

| field | value |
|---|---:|
| triangles | `2487474` |
| vertices | `2478890` |
| median area | `0.0002083771105390042` |
| max area | `164.05824279785156` |
| 99.9 percentile threshold | `0.5959478616714478` |
| selected face | `2462659` |
| selected area | `164.05824279785156` |

Implementation note: the selector now computes checkpoint triangle areas with torch chunking so large public-scene checkpoints do not require a huge `vertices[faces]` numpy expansion.

## Render-Backed Gate

Command:

```bash
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py \
  --baseline_model outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/model \
  --checkpoint_path outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt \
  --edit_json outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/selected_edit.json \
  --output_root outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/gate \
  --iteration 2000 \
  --gpu 4 \
  --max_points_per_view 500
```

Result:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| triangles | `2487474` | `2487473` | `-1` |
| vertices | `2478890` | `2478890` | `0` |
| PSNR | `12.201611518859863` | `12.20124340057373` | `-0.0003681182861328125` |
| SSIM | `0.20731531083583832` | `0.20730286836624146` | `-0.000012442469596862793` |
| LPIPS | `0.6242585182189941` | `0.6242548227310181` | `-0.0000036954879760742188` |
| AbsRel | `0.49587362441894434` | `0.49587362441894434` | `0.0` |
| Depth MAE | `4.907808996255763` | `4.907808996255763` | `0.0` |
| normal mean deg | `50.118300749023625` | `50.11830075792723` | `0.000000008903604964416445` |

## Decision

`PASS_DIAGNOSTIC`.

The second-scene checkpoint-statistics edit is accepted by the render-backed gate. It validates that the conservative area-outlier selector and checkpoint gate are stable beyond `parking_phone_tiny`.

This does not upgrade R14 to a full `PASS`, because the second scene was not followed by a W&B-logged recovery/medium training run and the effect size is deliberately tiny.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/selected_edit.json`
- `outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/area_outlier_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/gate/render_backed_checkpoint_gate_report.json`
