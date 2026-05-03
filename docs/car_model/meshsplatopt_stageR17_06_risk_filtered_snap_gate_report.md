# MeshSplatOpt Stage R17.06 Risk-Filtered Local Snap Gate Report

Date: 2026-05-03

## Decision

`RISK_FILTERED_LOCAL_SNAP_GATE_PASS`.

After R17.03-R17.05 showed that a large-area-seeded snap portfolio is gate-safe but not quality-improving, this stage adds stricter selector controls: maximum proposal uncertainty and optional boundary-vertex exclusion. The filtered selector is again validated on a real `parking_phone_tiny` checkpoint.

## Implementation

Updated:

- `scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py`

New CLI controls:

- `--max_proposal_uncertainty`
- `--exclude_boundary_vertices`

These controls remove high-risk boundary snaps from the portfolio before selection.

## Selection

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt --output_dir outputs/carnet/meshsplatopt/stageR17_06_parking_risk_filtered_local_snap_selection --top_k_faces 1024 --min_area_ratio_to_median 25 --min_percentile 98.5 --max_displacement_fraction 0.01 --residual_threshold_fraction 0.001 --max_selected_vertices 16 --min_selected_vertex_distance 0.25 --max_proposal_uncertainty 0.35 --exclude_boundary_vertices
```

Selection summary:

- candidate faces above threshold: `11746`
- selected vertices: `16`
- all selected proposals are non-boundary vertices
- max selected uncertainty: `0.35`
- total expected local residual reduction: `0.8844110663521292`
- topology cost delta: `0`

## Counterfactual Gate

Gate status: `PASS`.

| metric | baseline 2000 | risk-filtered candidate 2000 | delta |
|---|---:|---:|---:|
| triangles | `782982` | `782982` | `0` |
| vertices | `820107` | `820107` | `0` |
| PSNR | `11.599437713623047` | `11.59943675994873` | `-9.5367431640625e-07` |
| SSIM | `0.2702677547931671` | `0.2702675759792328` | `-1.7881393432617188e-07` |
| LPIPS | `0.6347319483757019` | `0.6347324848175049` | `+5.364418029785156e-07` |
| AbsRel | `0.42787965657189714` | `0.42787965657189714` | `0.0` |
| Depth MAE | `4.414160625200222` | `4.414160625200222` | `0.0` |
| normal mean deg | `52.565184963415106` | `52.56518371534546` | `-1.2480696440775318e-06` |

## Interpretation

The risk-filtered selector is safer than R17.03 because it avoids boundary vertices and caps uncertainty. It is still not a quality claim: the gate deltas remain numerical-noise level, and the previous W&B equal-budget recovery already showed that area-seeded snap portfolios do not beat continuation.

The next selector should not be another area-only variant. It should rank candidates using actual explanation debt: render residual, sparse-depth residual, normal disagreement, or CSEF defect regions.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR17_06_parking_risk_filtered_local_snap_selection/local_snap_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR17_06_parking_risk_filtered_local_snap_selection/selected_local_snap_edit.json`
- `outputs/carnet/meshsplatopt/stageR17_06_parking_risk_filtered_local_snap_selection/gate/render_backed_checkpoint_gate_report.json`
