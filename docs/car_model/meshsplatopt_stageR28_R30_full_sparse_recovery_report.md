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

## R33 Cross-Scene Trusted Sampling Check

R33 reruns the R31 cross-scene 2000-to-7000 sparse-depth setting with `mixed_low_error` sampling to test whether R32 generalizes as a render improvement or mainly as a geometry-confidence regularizer.

Courtyard Stage35 mixed trusted/random sparse-depth continuation:

- Output: `outputs/carnet/meshsplatopt/stageR33_01_courtyard_stage35_mixed_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `s1po8x07`
- Metrics at 7000: PSNR `16.304310`, SSIM `0.545805`, LPIPS `0.521787`
- Geometry at 7000: AbsRel `0.123796`, Depth MAE `1.536491`, normal angle `29.875990`
- Delta versus R31.02 random sampling: PSNR `-0.009172`, SSIM `-0.001965`, LPIPS `+0.001573`, AbsRel `-0.003747`, Depth MAE `-0.034883`, normal angle `-0.331460`

Bonsai Stage35 mixed trusted/random sparse-depth continuation:

- Output: `outputs/carnet/meshsplatopt/stageR33_02_bonsai_stage35_mixed_sparse_depth_lam0p005_2000to7000/recovery_model`
- W&B: `xj2ng1s1`
- Metrics at 7000: PSNR `20.279762`, SSIM `0.605154`, LPIPS `0.390035`
- Geometry at 7000: AbsRel `0.128458`, Depth MAE `1.417768`, normal angle `35.109088`
- Delta versus R31.03 random sampling: PSNR `-0.019484`, SSIM `-0.001719`, LPIPS `+0.001663`, AbsRel `-0.002109`, Depth MAE `-0.034337`, normal angle `+0.121622`

Decision: `TRUSTED_SAMPLING_GEOMETRY_PASS_RENDER_MIXED`. The trusted/random sampler is not a universal cross-scene render win at the tested 50/50 mixture, but it consistently improves sparse depth AbsRel and Depth MAE on both cross-scene checks and improves courtyard normal agreement. The paper should report R32 as the parking render-best setting and describe trusted sampling as a geometry-confidence knob whose mixture requires validation per scene.

## R34-R35 Parking Trusted-Fraction Ablation

R34 and R35 refine the parking trusted/random mixture around the R32 best setting. All rows continue R30.01 from 12000 to 16000 with `lambda=0.005`, online W&B, independent render metrics, and COLMAP sparse geometry evaluation at 16000.

| Row | trusted fraction | W&B | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal angle |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R34.01 | `0.25` | `jfcn9ug0` | `17.098461` | `0.531578` | `0.458490` | `0.184467` | `2.964016` | `41.684424` |
| R32.02b | `0.50` | `j58gdh9q` | `17.105490` | `0.532643` | `0.457859` | `0.184374` | `2.957988` | `41.764144` |
| R35.01 | `0.625` | `t8y6ryn9` | `17.105064` | `0.532436` | `0.457493` | `0.183602` | `2.959589` | `41.472216` |
| R34.02 | `0.75` | `ympoevql` | `17.099464` | `0.532346` | `0.457681` | `0.183488` | `2.959905` | `41.606181` |

Decision: `TRUSTED_FRACTION_PARETO_PASS`. The `0.50` mixture remains the parking render-table choice for PSNR and SSIM. The `0.625` mixture is a stronger Pareto point for perceptual/geometry quality: it nearly matches the best PSNR (`-0.000425`) while improving LPIPS by `-0.000366`, AbsRel by `-0.000772`, and normal angle by `-0.291927` versus R32.02b. The paper should report `0.50` as render-best and `0.625` as the geometry-balanced variant.

## R36-R38 Trusted-Fraction and Lambda Refinement

R36 tests whether the parking `0.625` trusted/random fraction gives useful cross-scene behavior:

| Row | Scene | fraction | W&B | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal angle |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R36.01b | courtyard | `0.625` | `qguqasou` | `16.376713` | `0.548868` | `0.520534` | `0.126731` | `1.564874` | `29.581638` |
| R36.02b | bonsai | `0.625` | `xq21lzsm` | `20.267965` | `0.605809` | `0.391068` | `0.130851` | `1.441631` | `35.098674` |

Decision: `CROSS_SCENE_FRACTION_TUNING_PARTIAL_PASS`. Fraction `0.625` is the current courtyard render-best setting, beating R31.02 by PSNR `+0.063231` and SSIM `+0.001098`, while improving normal angle by `-0.625812`. Bonsai remains better with the original random sparse sampling for render metrics, so the cross-scene claim should be scene-dependent trusted-fraction selection.

R37 probed an error-stratified sampler, but it was not retained because it degraded courtyard and did not improve bonsai render. Its results are logged in `SPCarNet_research_log.md` as a controlled negative.

R38 lowers the sparse depth lambda from `0.005` to `0.003` in the parking 12000-to-16000 setting:

| Row | fraction | lambda | W&B | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal angle |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R38.01 | `0.50` | `0.003` | `yo6oxofn` | `17.124186` | `0.533355` | `0.456678` | `0.183460` | `2.945563` | `41.679295` |
| R38.02 | `0.625` | `0.003` | `j8t2tyc9` | `17.107119` | `0.532528` | `0.456906` | `0.183256` | `2.933642` | `41.632026` |

Decision: `NEW_PARKING_RENDER_AND_GEOMETRY_BEST`. R38.01 is the new parking table row. Relative to R32.02b, it improves PSNR by `+0.018696`, SSIM by `+0.000712`, LPIPS by `-0.001181`, AbsRel by `-0.000915`, Depth MAE by `-0.012425`, and normal angle by `-0.084849`. R38.02 is the geometry-biased variant and is useful for a Pareto table: compared with R38.01, it further improves AbsRel by `-0.000204`, Depth MAE by `-0.011921`, and normal angle by `-0.047269`.

## R39 Parking Lambda Fine Sweep and Result Collector

R39 checks the local lambda curve around R38.01 using the same `0.50` trusted/random fraction:

| Row | fraction | lambda | W&B | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal angle |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R39.01 | `0.50` | `0.002` | `jqcn7cwc` | `17.142246` | `0.534422` | `0.456627` | `0.181240` | `2.825327` | `41.812617` |
| R38.01 | `0.50` | `0.003` | `yo6oxofn` | `17.124186` | `0.533355` | `0.456678` | `0.183460` | `2.945563` | `41.679295` |
| R39.02 | `0.50` | `0.004` | `o9f9e03g` | `17.088505` | `0.532507` | `0.457398` | `0.184764` | `2.956959` | `41.736803` |
| R32.02b | `0.50` | `0.005` | `j58gdh9q` | `17.105490` | `0.532643` | `0.457859` | `0.184374` | `2.957988` | `41.764144` |

Decision: `NEW_STRONGEST_PARKING_RESULT_AND_LAMBDA_CURVE_PASS`. R39.01 is the new parking table row and the strongest validated render/depth result. Relative to R38.01, it improves PSNR by `+0.018061`, SSIM by `+0.001067`, LPIPS by `-0.000051`, AbsRel by `-0.002219`, and Depth MAE by `-0.120237`. Relative to R32.02b, it improves PSNR by `+0.036756`, SSIM by `+0.001779`, LPIPS by `-0.001232`, AbsRel by `-0.003134`, and Depth MAE by `-0.132661`.

Reproducible table generation is now handled by:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_collect_sparse_recovery_results.py
```

The collector writes paper-facing JSON/CSV/Markdown tables under `outputs/carnet/meshsplatopt/sparse_recovery_tables`.

## R40 Low-Lambda Pareto and Cross-Scene Breakthrough

R40 extends the lambda sweep below R39 and applies the stronger low-lambda setting to the tuned courtyard fraction.

| Row | scene | fraction | lambda | W&B | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal angle |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R40.01 | parking | `0.50` | `0.001` | `czebaxco` | `17.145630` | `0.534154` | `0.456297` | `0.181336` | `2.849124` | `42.151608` |
| R39.01 | parking | `0.50` | `0.002` | `jqcn7cwc` | `17.142246` | `0.534422` | `0.456627` | `0.181240` | `2.825327` | `41.812617` |
| R40.02 | courtyard | `0.625` | `0.002` | `coqls9rm` | `16.801973` | `0.559031` | `0.508579` | `0.106783` | `1.388936` | `29.394197` |
| R43.02b | courtyard long | `0.625` | `0.002` | `cla3utia` | `17.793036` | `0.560976` | `0.496724` | `0.158907` | `1.991568` | `28.950016` |
| R43.01b | parking long | `0.50` | `0.001` | `mhz6t8ps` | `16.249155` | `0.511035` | `0.477426` | `0.193679` | `3.018124` | `43.714506` |
| R44.01 | parking decay long | `0.50` | `0.001` | `c1rxa6q6` | `17.169540` | `0.548714` | `0.441888` | `0.187067` | `2.919396` | `42.218251` |
| R44.02 | courtyard decay long | `0.625` | `0.002` | `5tleod3c` | `17.829701` | `0.561812` | `0.493252` | `0.147102` | `1.915970` | `26.520612` |
| R36.01b | courtyard | `0.625` | `0.005` | `qguqasou` | `16.376713` | `0.548868` | `0.520534` | `0.126731` | `1.564874` | `29.581638` |
| R41.01 | bonsai | `0.50` | `0.002` | `poh8k4be` | `21.601114` | `0.677450` | `0.347170` | `0.161510` | `1.824463` | `36.047671` |
| R42.01 | bonsai | `0.625` | `0.002` | `l2inxutg` | `21.543251` | `0.672968` | `0.349113` | `0.161678` | `1.824630` | `35.622191` |
| R31.03 | bonsai | random | `0.005` | `3wygm9u4` | `20.299246` | `0.606873` | `0.388372` | `0.130567` | `1.452105` | `34.987466` |

Decision: `LOW_LAMBDA_CROSS_SCENE_STRONG_PASS`. R40.01 is the parking render/LPIPS Pareto row; R39.01 remains the cleaner parking depth row. R40.02 is the stronger paper milestone because it improves every tracked courtyard metric relative to R36.01b: PSNR `+0.425260`, SSIM `+0.010163`, LPIPS `-0.011955`, AbsRel `-0.019948`, Depth MAE `-0.175938`, and normal angle `-0.187441`. Relative to the original courtyard random sparse-depth row R31.02, R40.02 improves PSNR by `+0.488491`, SSIM by `+0.011261`, LPIPS by `-0.011635`, AbsRel by `-0.020760`, Depth MAE by `-0.182438`, and normal angle by `-0.813253`.

R41.01 removes the remaining bonsai render weakness: relative to R31.03 random sparse-depth it improves PSNR by `+1.301868`, SSIM by `+0.070577`, and LPIPS by `-0.041202`. The tradeoff is geometry: AbsRel increases by `+0.030943`, Depth MAE by `+0.372358`, and normal angle by `+1.060204`. R42.01 checks whether raising the trusted fraction to `0.625` repairs this geometry tradeoff. It does not recover depth, and relative to R41.01 it gives up PSNR `-0.057863`, SSIM `-0.004483`, and LPIPS `+0.001943`, while only improving normal angle by `-0.425480`. This should be used as a render/geometry Pareto branch rather than an all-metric win.

R43 adds the requested long-horizon evidence. Parking `16000->30000` is not beneficial: R43.01b drops PSNR by `-0.896475` and worsens LPIPS by `+0.021129` relative to R40.01, so the validated parking budget remains 16000. Courtyard `7000->20000` is useful for a render-best row: R43.02b improves PSNR by `+0.991062` and LPIPS by `-0.011855` relative to R40.02, but it worsens AbsRel by `+0.052124` and Depth MAE by `+0.602633`. The paper should therefore present budget-aware Pareto rows rather than claiming monotonic long-training improvement.

R44 turns the R43 diagnosis into a mechanism: sparse COLMAP depth loss is useful early, but should decay after the sparse geometry has anchored the recovery. R44.01 repairs the parking long-horizon failure, improving PSNR by `+0.920386`, SSIM by `+0.037679`, and LPIPS by `-0.035538` versus R43.01b. R44.02 improves every courtyard metric versus R43.02b, including normal angle by `-2.429404` degrees. The direct clean-baseline visual comparison is saved under `outputs/carnet/meshsplatopt/baseline_vs_method_qualitative`.

## Current Interpretation

The strongest validated parking render result is now R44.01, while R39.01 remains the depth-balanced parking row. Compared with R16.03 full clean baseline, R44.01 improves PSNR by `+1.598975`, SSIM by `+0.100502`, LPIPS by `-0.086165`, AbsRel by `-0.070748`, and normal angle by `-7.571498`. R31 adds an important early-stop result and two cross-scene positives. R32 adds a concrete algorithmic improvement on top of long-horizon sparse-geometry-guided recovery: confidence-aware COLMAP sparse correspondence sampling. R33-R36 narrow the claim into a stronger paper-safe version: trusted sampling is a tunable confidence knob, not a universal constant fraction. R38-R44 then convert that tuning into a stronger lambda, budget, and decay curve: `lambda=0.001-0.002` is now the validated low-lambda regime, R44.02 is the courtyard long-horizon render/normal row, R40.02 remains the all-metric courtyard depth row, and R41.01 is the large bonsai render Pareto row.
