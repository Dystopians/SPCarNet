# MeshSplatOpt Stage R14.9 Area-Outlier Real Edit Selection Report

Date: 2026-05-02

## Gate

`PASS`.

R14.9 closes the first automatic real-checkpoint edit-selection loop:

```text
checkpoint evidence -> edit JSON -> checkpoint copy -> render-backed gate -> accepted/rejected report
```

This is a conservative delete edit selected from checkpoint statistics, not a final full-repair result.

## Selection Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_select_checkpoint_area_outlier_edit.py \
  --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt \
  --output_dir outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection \
  --top_k 1 \
  --min_area_ratio_to_median 1000 \
  --min_percentile 99.9
```

Selection result:

| field | value |
|---|---:|
| median triangle area | `0.005547030811843575` |
| max triangle area | `15501.270805580434` |
| 99.9 percentile threshold | `121.87766683618489` |
| selected face | `55379` |
| selected area | `15501.270805580434` |

## Gate Command

```bash
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py \
  --baseline_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model \
  --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt \
  --edit_json outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/selected_edit.json \
  --output_root outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/gate \
  --iteration 200 \
  --gpu 4 \
  --max_points_per_view 500
```

## Gate Result

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| triangles | `64497` | `64496` | `-1` |
| vertices | `193491` | `193491` | `0` |
| PSNR | `10.949986457824707` | `10.949986457824707` | `0.0` |
| SSIM | `0.2898596525192261` | `0.2898596525192261` | `0.0` |
| LPIPS | `0.6441746354103088` | `0.6441746354103088` | `0.0` |
| AbsRel | `0.32417137460470213` | `0.32417137460470213` | `0.0` |
| Depth MAE | `3.6485552222775537` | `3.6485552222775537` | `0.0` |
| normal mean deg | `51.68797353552561` | `51.68797353552561` | `0.0` |

## Decision

`PASS`.

The first automatic real checkpoint edit is accepted by independent render and geometry gates. This validates the end-to-end edit-selection infrastructure. It is still only a tiny conservative deletion, so it does not satisfy the R14 medium-scene repair claim by itself.

## Artefacts

- `scripts/car_model/meshsplatopt_select_checkpoint_area_outlier_edit.py`
- `outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/selected_edit.json`
- `outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/area_outlier_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/gate/render_backed_checkpoint_gate_report.json`
