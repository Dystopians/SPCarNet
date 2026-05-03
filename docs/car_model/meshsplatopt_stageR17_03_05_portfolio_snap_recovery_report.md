# MeshSplatOpt Stage R17.03-R17.05 Portfolio Snap Recovery Report

Date: 2026-05-03

## Decision

`PORTFOLIO_SNAP_GATE_PASS_RECOVERY_QUALITY_FAIL`.

The multi-candidate local snap portfolio is safe under the render-backed checkpoint gate, but it does not beat equal-budget baseline continuation after 200 W&B-logged recovery steps. This is a negative method result and should not be used as a quality-improvement claim.

## Portfolio Selection

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt --output_dir outputs/carnet/meshsplatopt/stageR17_03_parking_checkpoint_local_snap_portfolio_selection --top_k_faces 512 --min_area_ratio_to_median 50 --min_percentile 99.0 --max_displacement_fraction 0.02 --residual_threshold_fraction 0.002 --max_selected_vertices 16 --min_selected_vertex_distance 0.25
```

Selection summary:

- checkpoint triangles: `782982`
- checkpoint vertices: `820107`
- candidate faces above threshold: `7831`
- candidate vertices: `443`
- generated proposals: `1446`
- valid proposals: `1291`
- selected vertices: `16`
- total expected local residual reduction: `2.5543751879508467`
- topology cost delta: `0`

## Counterfactual Gate

Gate command used GPU4 because GPU1 was heavily occupied.

Gate status: `PASS`.

| metric | baseline 2000 | portfolio candidate 2000 | delta |
|---|---:|---:|---:|
| triangles | `782982` | `782982` | `0` |
| vertices | `820107` | `820107` | `0` |
| PSNR | `11.599437713623047` | `11.599437713623047` | `0.0` |
| SSIM | `0.2702677547931671` | `0.2702678143978119` | `+5.960464477539063e-08` |
| LPIPS | `0.6347319483757019` | `0.6347317099571228` | `-2.384185791015625e-07` |
| AbsRel | `0.42787965657189714` | `0.42787965657189714` | `0.0` |
| Depth MAE | `4.414160625200222` | `4.414160625200222` | `0.0` |
| normal mean deg | `52.565184963415106` | `52.56518587521641` | `+9.118013011288895e-07` |

## W&B Recovery

Both rows used:

- `--train_densify_until_iter 2000`
- `--train_skip_restricted_delaunay`
- `--iterations 200`
- `--load_iteration 2000`
- online W&B logging

W&B:

```text
baseline continuation: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/2puomo88
portfolio snap:        https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d6dc9qja
```

## Equal-Budget Results

| row | iter | triangles | vertices | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline freeze continuation | `2200` | `782982` | `820107` | `12.331464767456055` | `0.29822155833244324` | `0.6223230957984924` | `0.409263270533092` | `4.300272951592725` | `52.59563908919155` |
| portfolio snap freeze continuation | `2200` | `782982` | `820107` | `12.326042175292969` | `0.29780852794647217` | `0.6217538118362427` | `0.41021545914223945` | `4.307690971509187` | `52.82749430760039` |

Portfolio minus baseline:

| metric | delta |
|---|---:|
| triangles | `0` |
| vertices | `0` |
| PSNR | `-0.0054225921630859375` |
| SSIM | `-0.00041303038597106934` |
| LPIPS | `-0.0005692839622497559` |
| AbsRel | `+0.0009521886091474433` |
| Depth MAE | `+0.007418019916462255` |
| normal mean deg | `+0.23185521840883936` |

## Interpretation

The selector and portfolio plumbing are now real-checkpoint capable:

- multi-candidate local snap proposal selection works;
- the combined edit preserves topology exactly;
- the render-backed gate accepts the edit;
- W&B recovery runs successfully under equal budget.

The method signal is still not strong enough. The portfolio edit is too weak or poorly targeted: it is safe, but equal-budget recovery is worse on PSNR, SSIM, depth, and sparse normal, with only LPIPS slightly better. The next useful selector needs stronger evidence than large-area seeding, such as render residuals, sparse-depth residuals, normal disagreement, or defect regions from CSEF/defect mining.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR17_03_parking_checkpoint_local_snap_portfolio_selection/local_snap_selection_report.json`
- `outputs/carnet/meshsplatopt/stageR17_03_parking_checkpoint_local_snap_portfolio_selection/selected_local_snap_edit.json`
- `outputs/carnet/meshsplatopt/stageR17_03_parking_checkpoint_local_snap_portfolio_selection/gate/render_backed_checkpoint_gate_report.json`
- `outputs/carnet/meshsplatopt/stageR17_04_parking_baseline_freeze_skip_delaunay_2000to2200/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR17_04_parking_baseline_freeze_skip_delaunay_2000to2200/recovery_model/geometry_eval_colmap/iter_2200_max500.json`
- `outputs/carnet/meshsplatopt/stageR17_05_parking_portfolio_snap_freeze_skip_delaunay_2000to2200/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR17_05_parking_portfolio_snap_freeze_skip_delaunay_2000to2200/recovery_model/geometry_eval_colmap/iter_2200_max500.json`
