# MeshSplatOpt Stage R14.17 Bonsai Snap Post-Edit Recovery Diagnostic Report

Date: 2026-05-02

## Gate

`PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`.

This stage runs a W&B-logged 200-step recovery diagnostic on `bonsai` after the R14.15 accepted non-delete `SNAP_VERTICES` area-outlier edit. It resumes the edited checkpoint from iteration 2000 and trains to iteration 2200.

This is not an equal-budget medium result because no `bonsai` iteration-200 checkpoint was available locally.

## W&B

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/8qdzfu6h
```

## Command

```bash
WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online CUDA_VISIBLE_DEVICES=4 \
  /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_run_teacher_recovery.py \
  --model_path outputs/carnet/meshsplatopt/stageR14_15_bonsai_snap_outlier_nondelete/gate/candidate_model \
  --edit_json outputs/carnet/meshsplatopt/stageR14_15_bonsai_snap_outlier_nondelete/selected_snap_edit.json \
  --output_dir outputs/carnet/meshsplatopt/stageR14_17_bonsai_snap_postedit_recovery_diagnostic \
  --iterations 200 \
  --load_iteration 2000 \
  --run_real_tiny \
  --gpu 4 \
  --python /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  --wandb_project spcarnet_meshprior \
  --wandb_group meshsplatopt_r14_nondelete_snap_recovery_diag \
  --wandb_name meshsplatopt_r14_17_bonsai_snap_recovery_200step_gpu4
```

## Results

| row | iteration | triangles | vertices | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline sparse-depth | `2000` | `2487474` | `2478890` | `12.201611518859863` | `0.20731531083583832` | `0.6242585182189941` |
| snap post-edit gate | `2000` | `2487474` | `2478890` | `12.201420783996582` | `0.20730163156986237` | `0.6242029070854187` |
| snap recovery diagnostic | `2200` | `2487474` | `2478890` | `13.273988723754883` | `0.24039088189601898` | `0.6116319894790649` |

Sparse COLMAP geometry:

| row | points | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|
| baseline sparse-depth | `18500` | `0.49587362441894434` | `4.907808996255763` | `50.118300749023625` |
| snap recovery diagnostic | `18500` | `0.47445281696526337` | `4.772623802825101` | `49.315686202793366` |

## Comparison To R14.13 Delete Recovery

| row | iteration | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | normal mean deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| R14.13 delete recovery | `2200` | `13.276382446289062` | `0.24055197834968567` | `0.6113873720169067` | `0.4733479577347401` | `4.762276469029142` | `49.21947049923495` |
| R14.17 snap recovery | `2200` | `13.273988723754883` | `0.24039088189601898` | `0.6116319894790649` | `0.47445281696526337` | `4.772623802825101` | `49.315686202793366` |

The non-delete snap recovery is stable and close to the delete recovery row, but it is not stronger than R14.13 on this 200-step diagnostic.

## Decision

`PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`.

The accepted non-delete checkpoint edit can be resumed with W&B online, rendered, evaluated, and geometrically checked after recovery. It improves render and sparse geometry metrics versus the 2000iter baseline, but it uses 200 extra training steps and should not be reported as an equal-budget result.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_17_bonsai_snap_postedit_recovery_diagnostic/teacher_recovery_run_report.json`
- `outputs/carnet/meshsplatopt/stageR14_17_bonsai_snap_postedit_recovery_diagnostic/recovery_model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_17_bonsai_snap_postedit_recovery_diagnostic/recovery_model/geometry_eval_colmap/iter_2200_max500.json`
