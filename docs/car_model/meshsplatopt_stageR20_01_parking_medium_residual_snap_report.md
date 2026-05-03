# MeshSplatOpt R20.01 Parking Medium Residual Snap Report

Date: 2026-05-03

## Objective

Run a medium-budget W&B recovery for the R18 train-residual parking snap candidate and compare it to the existing same-protocol parking baseline at iteration 4000.

## Protocol

Candidate:

`outputs/carnet/meshsplatopt/stageR18_02_parking_train_residual_snap_selection/gate/candidate_model`

Recovery output:

`outputs/carnet/meshsplatopt/stageR20_01_parking_train_residual_snap_medium_freeze_skip_delaunay_2000to4000`

Training:

- load iteration: `2000`
- train until: `4000`
- iterations: `2000`
- densify until: `2000`
- restricted Delaunay: skipped
- GPU: `4`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/tu85uksa`

Exit codes:

- train: `0`
- render: `0`
- metrics: `0`
- geometry eval: `0`

## Medium-Budget Result

Comparison against existing parking baseline:

`outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000`

| run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal mean deg ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 2000->4000 | `14.251087` | `0.383800` | `0.569749` | `0.324794` | `3.636891` | `51.043451` |
| residual snap 2000->4000 | `14.207231` | `0.383298` | `0.570288` | `0.323844` | `3.589209` | `51.225949` |

Residual snap minus baseline:

- PSNR: `-0.043857`
- SSIM: `-0.000501`
- LPIPS: `+0.000539`
- AbsRel: `-0.000951`
- Depth MAE: `-0.047682`
- Normal mean deg: `+0.182499`

## Decision

`MEDIUM_RESIDUAL_SNAP_DEPTH_GAIN_RENDER_QUALITY_FAIL`

The medium run confirms that the train-residual snap edit is not a robust appearance-quality improvement at 4000 iterations. It does improve depth metrics, especially Depth MAE, but loses PSNR, SSIM, LPIPS, and normal mean angle. This reinforces the R19 conclusion: isolated vertex snaps are useful as a gate-safe residual/depth repair primitive, but they are too sparse to become the main paper result. The next method step should be clustered patch repair or fill/split proposals under the same CSEF gate.
