# MeshSplatOpt R28-R30 Full Sparse Recovery Report

Date: 2026-05-03

## Goal

Stress-test the parking recovery path at full budget after medium-run sparse depth recovery showed the first credible gains. The key question is whether the benefit comes from the boundary grid-fill edit or from sparse COLMAP depth supervision during recovery.

## R28 Full-Budget Results

Reference full baseline:

- R16.03 baseline full: `outputs/carnet/meshsplatopt/stageR16_03_parking_baseline_freeze_densify_skip_delaunay_2000to7000/recovery_model`
- Metrics: PSNR `15.570565`, SSIM `0.448212`, LPIPS `0.528052`

Grid-fill plus sparse depth:

- R28.01 output: `outputs/carnet/meshsplatopt/stageR28_01_parking_grid_fill_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `94pkp05l`
- Metrics: PSNR `15.770156`, SSIM `0.459545`, LPIPS `0.519976`
- Geometry: AbsRel `0.240156`, Depth MAE `3.189930`, normal angle `46.143910`

Matched baseline plus sparse depth:

- R28.02 output: `outputs/carnet/meshsplatopt/stageR28_02_parking_baseline_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `zm1ztyf4`
- Metrics: PSNR `15.822877`, SSIM `0.458552`, LPIPS `0.519231`
- Geometry: AbsRel `0.231866`, Depth MAE `3.089107`, normal angle `45.929940`

Lower sparse weight on grid-fill:

- R28.03 output: `outputs/carnet/meshsplatopt/stageR28_03_parking_grid_fill_sparse_depth_lam0p002_2000to7000/recovery_model`
- W&B: `7u0onsok`
- Metrics: PSNR `15.741236`, SSIM `0.455811`, LPIPS `0.520650`

Decision: `SPARSE_DEPTH_FULL_PASS_GRID_FILL_REJECTED`. Full-budget sparse depth is a strong recovery method, but the current boundary grid-fill edit does not improve the full-budget baseline+sparse control.

## R29 Loss-Space Diagnostic

The sparse depth loss was extended with optional `depth`, `relative`, `log`, and `inverse` spaces via:

- `--sparse_colmap_depth_loss_space`
- `--sparse_colmap_depth_robust_beta`

Results:

- R29.01 relative, lambda `0.05`, beta `0.02`: PSNR `15.643266`, SSIM `0.454726`, LPIPS `0.522929`
- R29.02 log, lambda `0.02`, beta `0.02`: PSNR `15.608345`, SSIM `0.452642`, LPIPS `0.525190`

Decision: `LOSS_SPACE_DIAGNOSTIC_REJECTED_FOR_PARKING_FULL`. The original metric-depth Smooth L1 loss remains the best validated sparse-depth variant on this scene.

## R30 Long-Run Follow-Up

R30.01 continues the strongest R28.02 checkpoint from iteration 7000 to 12000 with the validated metric-depth sparse supervision:

- Output: `outputs/carnet/meshsplatopt/stageR30_01_parking_baseline_sparse_depth_lam0p005_7000to12000/recovery_model`
- W&B group: `meshsplatopt_r30_long_sparse_depth`
- W&B: `9oi1skys`
- Metrics at 12000: PSNR `16.872860`, SSIM `0.514039`, LPIPS `0.475757`
- Geometry at 12000: AbsRel `0.192306`, Depth MAE `2.873014`, normal angle `42.638562`

R30.02 continues R30.01 from iteration 12000 to 16000:

- Output: `outputs/carnet/meshsplatopt/stageR30_02_parking_baseline_sparse_depth_lam0p005_12000to16000/recovery_model`
- W&B group: `meshsplatopt_r30_long_sparse_depth`
- W&B: `6gsab26p`
- Metrics at 16000: PSNR `17.081682`, SSIM `0.531858`, LPIPS `0.458050`
- Geometry at 16000: AbsRel `0.185581`, Depth MAE `2.961570`, normal angle `41.859201`

## Current Interpretation

The strongest validated full result is R30.02. Compared with R16.03 full baseline, it improves PSNR by `+1.511117`, SSIM by `+0.083646`, and LPIPS by `-0.070003`, while substantially improving sparse COLMAP depth agreement. Compared with R28.02, extending the validated sparse-depth recovery from 7000 to 16000 adds another `+1.258805` PSNR, `+0.073306` SSIM, and `-0.061182` LPIPS. The main paper method should pivot from the boundary fill edit to long-horizon sparse-geometry-guided recovery unless a later structural edit beats the R30 control.
