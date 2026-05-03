# MeshSplatOpt R27 Sparse-Depth Recovery Report

Date: 2026-05-03

## Summary

Stage R27 found the first strong medium-budget parking repair result after the R24-R26 boundary-fill diagnostics.

The winning recipe is:

- R26 grid `FILL_PATCH` edit: `+51` vertices, `+106` triangles;
- nearest-face checkpoint initialization from R24;
- frozen topology recovery from 2000 to 4000;
- low-weight sparse COLMAP depth recovery:
  - `--enable_sparse_colmap_depth_loss`
  - `--lambda_sparse_colmap_depth 0.005`
  - `--sparse_colmap_depth_start_iter 2000`
  - `--sparse_colmap_depth_warmup_iters 50`
  - `--sparse_colmap_depth_min_matches 16`
  - `--sparse_colmap_depth_enable_in_final_finetune`

Decision: `SPARSE_DEPTH_REPAIR_MEDIUM_PASS`.

This result should be framed carefully: most of the large gain comes from sparse-depth recovery, but the edit+sparse run still beats a matched baseline+sparse run on render and geometry metrics.

## Implementation

Updated `scripts/car_model/meshsplatopt_run_teacher_recovery.py`:

- added `--train_extra_args`;
- parses shell-style extra training flags with `shlex.split`;
- records extra flags in `real_tiny_recovery_report.json`.

This makes recovery diagnostics explicit and reproducible without hard-coding new train options in the runner.

## Runs

### R27.01 high-weight sparse-depth diagnostic

Run:

- output: `outputs/carnet/meshsplatopt/stageR27_01_parking_boundary_grid_fill_sparse_depth_2000to2200`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/hrug0itm`
- lambda: `0.05`

Result: `FAIL`.

- PSNR: `12.315643`
- SSIM: `0.297245`
- LPIPS: `0.622109`
- AbsRel: `0.411106`
- DepthMAE: `4.342819`
- normal: `52.800506`

The sparse-depth term was too strong and damaged both render and geometry metrics.

### R27.02 low-weight short diagnostic

Run:

- output: `outputs/carnet/meshsplatopt/stageR27_02_parking_boundary_grid_fill_sparse_depth_lam0p005_2000to2200`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ogabx44c`
- lambda: `0.005`

Result: `SHORT_PASS`.

| Method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| R17 baseline | 12.331465 | 0.298222 | 0.622323 | 0.409263 | 4.300273 | 52.595639 |
| R22 fan fill | 12.354150 | 0.298658 | 0.621934 | 0.410232 | 4.302468 | 52.328850 |
| R26 grid fill | 12.347396 | 0.298338 | 0.622142 | 0.408940 | 4.303084 | 52.550777 |
| R27 grid fill + sparse-depth 0.005 | 12.362178 | 0.299357 | 0.621872 | 0.407613 | 4.307866 | 52.595478 |

This was the best short-run render and AbsRel result among the parking repair variants.

### R27.03 medium edit+sparse

Run:

- output: `outputs/carnet/meshsplatopt/stageR27_03_parking_boundary_grid_fill_sparse_depth_lam0p005_2000to4000`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/81hryi53`

Result: `MEDIUM_PASS`.

| Method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| R15 baseline | 14.251087 | 0.383800 | 0.569749 | 0.324794 | 3.636891 | 51.043451 |
| R26 grid fill | 14.212496 | 0.383164 | 0.570729 | 0.329141 | 3.667578 | 51.594204 |
| R27 grid fill + sparse-depth 0.005 | 14.325891 | 0.385450 | 0.567749 | 0.306381 | 3.605697 | 49.906129 |

Delta versus R15 baseline:

- PSNR: `+0.074804`
- SSIM: `+0.001650`
- LPIPS: `-0.001999`
- AbsRel: `-0.018413`
- DepthMAE: `-0.031194`
- normal: `-1.137322`

### R27.04 matched baseline+sparse control

Run:

- output: `outputs/carnet/meshsplatopt/stageR27_04_parking_baseline_sparse_depth_lam0p005_2000to4000`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/b726rga8`

Result: `CONTROL_STRONG`.

| Method | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
|---|---:|---:|---:|---:|---:|---:|
| baseline + sparse-depth 0.005 | 14.301250 | 0.384772 | 0.567846 | 0.309894 | 3.666060 | 50.012948 |
| grid fill + sparse-depth 0.005 | 14.325891 | 0.385450 | 0.567749 | 0.306381 | 3.605697 | 49.906129 |

Delta of edit+sparse versus matched sparse baseline:

- PSNR: `+0.024641`
- SSIM: `+0.000678`
- LPIPS: `-0.000097`
- AbsRel: `-0.003513`
- DepthMAE: `-0.060363`
- normal: `-0.106820`
- topology: `+51` vertices, `+106` triangles

## Interpretation

R27 changes the status of the parking repair line:

- R24-R26 showed safe topology-adding machinery but weak medium results.
- R27 shows that low-weight sparse geometry recovery turns the same repair into a medium-budget improvement.
- The matched control shows sparse recovery is the dominant contributor, but the grid fill edit still adds measurable benefit under identical recovery settings.

This is not yet enough for a full NeurIPS claim by itself. It is, however, a real medium-budget gain and a credible next anchor for cross-scene testing.

## Next Steps

1. Run the same sparse-depth recovery control on at least one geometry-observable public scene where residual snap was available.
2. Add edit-region metrics so the `+106` triangle fill can be evaluated locally, not only through global PSNR.
3. Promote `--train_extra_args` use into a named recovery profile once cross-scene evidence supports it.
