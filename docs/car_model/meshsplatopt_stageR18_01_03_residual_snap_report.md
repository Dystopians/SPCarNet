# MeshSplatOpt R18.01-R18.03 Residual-Aware Local Snap Report

Date: 2026-05-03

## Objective

R17 showed that area-seeded local snap portfolios were safe under the render-backed gate but did not beat an equal-budget continuation after recovery. R18 addresses that shortcoming by selecting local snap vertices from high-error input-view residuals, then validating the resulting checkpoint edit on held-out test views.

## Implementation

Added `scripts/car_model/meshsplatopt_select_checkpoint_residual_snap_edit.py`.

The selector:

- loads rendered RGB and GT RGB pairs from `train/ours_<iter>` or `test/ours_<iter>`;
- ranks residual-heavy views by mean absolute RGB residual;
- projects large-area candidate vertices into selected residual maps;
- scores candidates by sampled multi-view residual and local plane residual reduction;
- filters proposals with CSEF risk controls: max uncertainty, optional boundary exclusion, minimum expected reduction;
- emits a `SNAP_VERTICES` edit JSON plus a compact audit report.

Important protocol note: `render_set=test` is diagnostic only. The paper-valid selector run uses `render_set=train` with `--camera_index_offset 54`, because this model's `cameras.json` stores the 54 test cameras before the 371 train cameras.

## Diagnostic Test-Residual Run

Output root:

`outputs/carnet/meshsplatopt/stageR18_01_parking_test_residual_snap_selection`

Selection summary:

- status: `PASS`
- render set: `test`
- candidate faces: `19575`
- candidate vertices: `4469`
- scored vertices: `2948`
- proposals: `3006`
- valid proposals: `525`
- selected vertices: `16`

Held-out gate is not claimed for method validity here because test residuals were used for selection. It was only used to debug the residual selector.

## Train-Residual Selector

Output root:

`outputs/carnet/meshsplatopt/stageR18_02_parking_train_residual_snap_selection`

Selection summary:

- status: `PASS`
- render set: `train`
- camera index offset: `54`
- candidate faces: `19575`
- candidate vertices: `4469`
- scored vertices: `3918`
- proposals: `3000`
- valid proposals: `438`
- selected vertices: `16`

Top selected vertices:

| vertex | residual score | local residual before | local residual after | uncertainty |
| --- | ---: | ---: | ---: | ---: |
| `730295` | `0.826144` | `0.169120` | `0.084560` | `0.35` |
| `500770` | `0.654902` | `0.120799` | `0.060400` | `0.35` |
| `676458` | `0.488889` | `0.105257` | `0.052629` | `0.35` |

Edit JSON:

`outputs/carnet/meshsplatopt/stageR18_02_parking_train_residual_snap_selection/selected_residual_snap_edit.json`

## Render-Backed Gate

Gate output:

`outputs/carnet/meshsplatopt/stageR18_02_parking_train_residual_snap_selection/gate/render_backed_checkpoint_gate_report.json`

Result: `PASS`

Topology:

- baseline: `782982` triangles, `820107` vertices
- candidate: `782982` triangles, `820107` vertices

Held-out iteration-2000 deltas:

| metric | delta |
| --- | ---: |
| PSNR | `0.0` |
| SSIM | `-1.4901161193847656e-07` |
| LPIPS | `+1.7881393432617188e-07` |
| AbsRel | `0.0` |
| Depth MAE | `0.0` |
| Normal mean deg | `-2.47278670428841e-07` |

## W&B Recovery

Recovery run:

`outputs/carnet/meshsplatopt/stageR18_03_parking_train_residual_snap_freeze_skip_delaunay_2000to2200`

W&B:

- project: `spcarnet_meshprior`
- run id: `1oqymqmp`
- URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1oqymqmp`

Training protocol:

- scene: parking tiny real checkpoint
- load iteration: `2000`
- train until: `2200`
- densify until: `2000`
- restricted Delaunay: skipped
- GPU: `4`

## Equal-Budget Result

Comparison at iteration 2200:

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R17.04 baseline continuation | `12.331465` | `0.298222` | `0.622323` | `0.409263` | `4.300273` | `52.595639` |
| R17.05 area portfolio snap | `12.326042` | `0.297809` | `0.621754` | `0.410215` | `4.307691` | `52.827494` |
| R18.03 train-residual snap | `12.342549` | `0.298893` | `0.622299` | `0.408892` | `4.302941` | `52.354489` |

R18.03 minus R17.04 baseline:

- PSNR: `+0.011085`
- SSIM: `+0.000672`
- LPIPS: `-0.000024`
- AbsRel: `-0.000371`
- Depth MAE: `+0.002668`
- Normal mean deg: `-0.241150`

## Decision

`TRAIN_RESIDUAL_SNAP_GATE_PASS_RECOVERY_MOSTLY_POSITIVE`

This is the first local snap portfolio variant that improves PSNR, SSIM, AbsRel, and normal angle over the equal-budget continuation baseline while staying gate-safe. The effect size is still small and depth MAE is slightly worse, so this is not yet a headline result. It is, however, a meaningful repair of the R17 shortcoming: proposal selection is now tied to observed residual evidence rather than triangle area alone.

## Next Work

- replace RGB residual scoring with a joint RGB, sparse-depth, and normal-disagreement score;
- run the same train-residual selector across at least two more scenes;
- move from 16 isolated vertex snaps to clustered residual patches;
- run a medium-length recovery once multi-scene gate behavior is stable.
