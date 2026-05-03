# MeshSplatOpt Stage R17.02 Checkpoint Local Snap Gate Report

Date: 2026-05-03

## Decision

`REAL_CHECKPOINT_LOCAL_SNAP_GATE_PASS`.

This stage connects the CSEF-aware local snap selector to a real Mesh Splatting checkpoint and validates the selected non-delete edit with the render-backed checkpoint gate on `parking_phone_tiny`.

## Implementation

Added:

- `scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py`

The selector:

- seeds candidate regions from large checkpoint triangle areas;
- extracts candidate vertices from the top-ranked faces;
- evaluates only those candidate vertices with the local CSEF snap selector;
- selects the proposal with the largest local residual reduction;
- writes `selected_local_snap_edit.json` for the existing checkpoint edit and gate path.

The shared snap proposal code was also optimized so explicit candidate lists do not trigger full-scene local-plane fitting.

## Selection

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt --output_dir outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection --top_k_faces 64 --min_area_ratio_to_median 100 --min_percentile 99.5 --max_displacement_fraction 0.02 --residual_threshold_fraction 0.002
```

Selection summary:

- checkpoint triangles: `782982`
- checkpoint vertices: `820107`
- candidate faces above threshold: `3915`
- top faces used: `64`
- candidate vertices: `45`
- generated proposals: `135`
- valid proposals: `113`
- selected vertex: `704480`
- selected local residual: `0.042196625106825536 -> 0.021098312553412768`
- topology cost delta: `0`

## Counterfactual Gate

Command:

```text
CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py --baseline_model outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt --edit_json outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/selected_local_snap_edit.json --output_root outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/gate --iteration 2000 --gpu 1 --python /home/peilincai/micromamba/envs/mesh_splatting/bin/python --max_points_per_view 500
```

Gate status: `PASS`.

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| triangles | `782982` | `782982` | `0` |
| vertices | `820107` | `820107` | `0` |
| PSNR | `11.599437713623047` | `11.59943675994873` | `-9.5367431640625e-07` |
| SSIM | `0.2702677547931671` | `0.2702677547931671` | `0.0` |
| LPIPS | `0.6347319483757019` | `0.6347321271896362` | `1.7881393432617188e-07` |
| AbsRel | `0.42787965657189714` | `0.4278796565966673` | `2.4770185902411868e-11` |
| Depth MAE | `4.414160625200222` | `4.414160629291564` | `4.0913414878218646e-09` |
| normal mean deg | `52.565184963415106` | `52.5651853881184` | `4.2470329475463586e-07` |

## Interpretation

This is a real-scene safety and integration pass. The local CSEF snap selector can now produce a checkpoint edit that is materialized, rendered, geometrically evaluated, and accepted by the existing counterfactual gate without topology growth.

The effect size is intentionally tiny because this diagnostic uses a single conservative vertex snap. It should not be reported as a quality improvement. Its value is that the snap path is no longer synthetic-only: it now has a real checkpoint proposal/gate audit trail and can be scaled to stronger proposal portfolios.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/local_snap_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/selected_local_snap_edit.json`
- `outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/gate/render_backed_checkpoint_gate_report.json`
- `outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/gate/candidate_model/results.json`
- `outputs/carnet/meshsplatopt/stageR17_02_parking_checkpoint_local_snap_selection/gate/candidate_model/geometry_eval_colmap/iter_2000_max500.json`

## Next Gate

The next useful step is portfolio scale-up: select multiple local snap candidates with free-space and render-risk prefilters, then compare W&B-logged equal-budget recovery against the freeze baseline only after the counterfactual gate accepts the edits.
