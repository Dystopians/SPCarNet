# MeshSplatOpt Stage R14.10 Medium Area-Outlier Pilot Report

Date: 2026-05-02

## Gate

`SOFT PASS_SINGLE_SCENE`.

This stage runs the first W&B-logged medium-budget MeshSplatOpt candidate on `parking_phone_tiny`. It starts from the R14.9 render-gated automatic area-outlier edit at iteration 200, resumes training to iteration 2000, renders independently, runs `metrics.py`, and evaluates sparse COLMAP geometry.

The result is strong on one scene, but R14 still needs a second scene before a full `PASS`.

## W&B

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/81kwhzr3
```

## Training Command

```bash
WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online CUDA_VISIBLE_DEVICES=4 \
  /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_run_teacher_recovery.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/gate/candidate_model \
  --edit_json outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/selected_edit.json \
  --output_dir outputs/carnet/meshsplatopt/stageR14_10_medium_area_outlier_recovery \
  --iterations 1800 \
  --load_iteration 200 \
  --run_real_tiny \
  --gpu 4 \
  --python /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  --wandb_project spcarnet_meshprior \
  --wandb_group meshsplatopt_r14_medium_pilot \
  --wandb_name meshsplatopt_r14_10_parking_area_outlier_recovery_2000iter_gpu4
```

## Independent Render Metrics

| row | iteration | triangles | vertices | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| current branch baseline | `2000` | `782982` | `820107` | `11.599437713623047` | `0.2702677547931671` | `0.6347319483757019` |
| MeshSplatOpt area-outlier + recovery | `2000` | `783509` | `822064` | `13.276764869689941` | `0.30384060740470886` | `0.6081721186637878` |

Delta candidate minus baseline:

| metric | delta |
|---|---:|
| triangles | `+527` |
| vertices | `+1957` |
| PSNR | `+1.6773271560668945` |
| SSIM | `+0.03357285261154175` |
| LPIPS | `-0.026559829711914062` |

## Sparse COLMAP Geometry

Comparable `--max_points_per_view 500`:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| current branch baseline | `21911` | `0.42787965657189714` | `4.414160625200222` | `52.565184963415106` |
| MeshSplatOpt area-outlier + recovery | `21911` | `0.3640420630578014` | `3.806375643108584` | `52.672900862227785` |

Delta candidate minus baseline:

| metric | delta |
|---|---:|
| AbsRel | `-0.06383759351409572` |
| Depth MAE | `-0.6077849820916381` |
| normal mean deg | `+0.10771589881267915` |

## Decision

`SOFT PASS_SINGLE_SCENE`.

The medium candidate improves all independent render metrics and sparse depth geometry over the current-branch 2000iter baseline on `parking_phone_tiny`, with a small topology increase and a small normal-angle regression. This is the first evidence that the R14 real checkpoint edit/recovery path can produce a meaningful medium-budget result.

It is not a full R14 `PASS` because the prompt requires at least two scenes and stronger baselines such as Stage35/PRISM retained refresh where compatible.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_10_medium_area_outlier_recovery/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_10_medium_area_outlier_recovery/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_10_medium_area_outlier_recovery/recovery_model/geometry_eval_colmap/iter_2000_max500.json`
- `outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/geometry_eval_colmap/iter_2000_max500.json`
