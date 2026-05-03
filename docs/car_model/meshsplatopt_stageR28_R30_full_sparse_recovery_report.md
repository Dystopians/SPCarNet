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

R31.01 continues R30.02 from iteration 16000 to 20000:

- Output: `outputs/carnet/meshsplatopt/stageR31_01_parking_baseline_sparse_depth_lam0p005_16000to20000/recovery_model`
- W&B: `ekcjc7qi`
- Metrics at 20000: PSNR `17.027088`, SSIM `0.532724`, LPIPS `0.455719`
- Geometry at 20000: AbsRel `0.187616`, Depth MAE `3.017283`, normal angle `41.740965`
- Decision: `RENDER_EARLY_STOP_AT_16000`. The 20000-step run improves LPIPS and normal angle slightly, but loses PSNR and sparse depth versus 16000.

## R31 Cross-Scene Generalization

Courtyard Stage35 sparse-depth continuation:

- Output: `outputs/carnet/meshsplatopt/stageR31_02_courtyard_stage35_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `s35bmzau`
- Baseline at 2000: PSNR `15.383161`, SSIM `0.508091`, LPIPS `0.584694`
- Sparse recovery at 7000: PSNR `16.313482`, SSIM `0.547770`, LPIPS `0.520214`
- Delta: PSNR `+0.930322`, SSIM `+0.039679`, LPIPS `-0.064480`
- Geometry at 7000: AbsRel `0.127543`, Depth MAE `1.571374`, normal angle `30.207450`

Bonsai Stage35 sparse-depth continuation:

- Output: `outputs/carnet/meshsplatopt/stageR31_03_bonsai_stage35_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `3wygm9u4`
- Baseline at 2000: PSNR `12.267367`, SSIM `0.277617`, LPIPS `0.611939`
- Sparse recovery at 7000: PSNR `20.299246`, SSIM `0.606873`, LPIPS `0.388372`
- Delta: PSNR `+8.031878`, SSIM `+0.329256`, LPIPS `-0.223567`
- Geometry at 7000: AbsRel `0.130567`, Depth MAE `1.452105`, normal angle `34.987466`

Decision: `CROSS_SCENE_SPARSE_RECOVERY_PASS`. Sparse-geometry-guided recovery now has positive evidence on parking, courtyard, and bonsai.

## R32 Trusted Sparse Correspondence Sampling

R32 adds a targeted sparse-supervision sampling improvement. Instead of always subsampling visible COLMAP points uniformly, sparse depth training and geometry evaluation can now prioritize low-reprojection-error COLMAP tracks:

- `--sparse_colmap_depth_sample_mode random`: previous default.
- `--sparse_colmap_depth_sample_mode low_error`: choose the lowest-error visible tracks.
- `--sparse_colmap_depth_sample_mode mixed_low_error --sparse_colmap_depth_low_error_fraction 0.5`: use half trusted low-error tracks and half random tracks.

Parking low-error-only continuation from R30.01, 12000 to 16000:

- Output: `outputs/carnet/meshsplatopt/stageR32_01b_parking_sparse_depth_low_error_lam0p005_12000to16000/recovery_model`
- W&B: `m8fu6936`
- Metrics at 16000: PSNR `17.086828`, SSIM `0.532577`, LPIPS `0.457497`
- Geometry at 16000: AbsRel `0.185512`, Depth MAE `2.966934`, normal angle `41.771796`

Parking mixed trusted/random continuation from R30.01, 12000 to 16000:

- Output: `outputs/carnet/meshsplatopt/stageR32_02b_parking_sparse_depth_mixed_low_error_lam0p005_12000to16000/recovery_model`
- W&B: `j58gdh9q`
- Metrics at 16000: PSNR `17.105490`, SSIM `0.532643`, LPIPS `0.457859`
- Geometry at 16000: AbsRel `0.184374`, Depth MAE `2.957988`, normal angle `41.764144`

Decision: `TRUSTED_MIXED_SPARSE_SAMPLING_PASS`. Mixed trusted/random sampling is the new strongest parking result. Relative to R30.02, it improves PSNR by `+0.023808`, SSIM by `+0.000785`, LPIPS by `-0.000191`, AbsRel by `-0.001207`, Depth MAE by `-0.003582`, and normal angle by `-0.095057`.

## Current Interpretation

The strongest validated parking render result is now R32.02b at 16000. Compared with R16.03 full baseline, it improves PSNR by `+1.534925`, SSIM by `+0.084431`, and LPIPS by `-0.070193`, while substantially improving sparse COLMAP depth agreement. R31 adds an important early-stop result and two cross-scene positives. R32 adds a concrete algorithmic improvement on top of long-horizon sparse-geometry-guided recovery: confidence-aware sparse correspondence sampling.
