# MeshPrior Parking Medium Baseline 2000-Iter Report

Date: 2026-05-01

## Scope

This report compares the first medium-budget `parking_phone_tiny` baselines:

- clean/original Mesh Splatting candidate from `origin/main@1a714f3`
- current `clean-submit` branch engineering baseline

The user's correction is adopted here: a 200-iteration run is only a smoke test. The paper baseline should be clean/original Mesh Splatting on the same data, budget, and metric scripts.

## W&B Accountability

The current-branch 2000-iteration run used training-time online W&B.

- run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nk2w04wn`
- group: `parking_current_branch_baseline`
- name: `parking_current_branch_gpu1_2000iter`

The clean `origin/main` script does not expose the current branch's `--enable_wandb` flags. That run was therefore logged after training with the external summary logger.

- run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`
- group: `parking_origin_main_baseline`
- name: `parking_origin_main_2000iter_external_log`

This distinction matters. Future current-branch training runs must use training-time W&B. If a historical baseline cannot do that, the report must explicitly say so and log externally immediately.

## Commands

Current branch 2000-iteration training:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model --images images --eval --iterations 2000 --test_iterations 1000 2000 --save_iterations 2000 --checkpoint_iterations 2000 --resolution 4 --enable_wandb --wandb_project spcarnet_meshprior --wandb_group parking_current_branch_baseline --wandb_name parking_current_branch_gpu1_2000iter --wandb_scalar_log_interval 20 --wandb_disable_fixed_views --scene_name parking_phone_tiny_current_branch_2000iter
```

Current branch render metrics:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model --images images --eval --iteration 2000 --skip_train --quiet
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model
```

Current branch COLMAP geometry proxy:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python evaluate_geometry_colmap.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model --images images --eval --iteration 2000 --max_points_per_view 500 --output outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/geometry_eval_colmap/iter_2000.json
```

Origin/main commands are recorded in `docs/car_model/meshprior_parking_origin_main_baseline_report.md`.

## Training Internal Metrics

These values come from the training script's internal test evaluation and must not be mixed with `render.py + metrics.py` values without labels.

| branch | iter | test L1 | test PSNR | test SSIM | test LPIPS | test FPS |
|---|---:|---:|---:|---:|---:|---:|
| origin/main clean candidate | 2000 | 0.1105293389 | 16.4619565010 | 0.4846517714 | 0.5333475658 | 271.3129810583 |
| current branch engineering | 2000 | 0.1104943447 | 16.4415020589 | 0.4834401826 | 0.5322314313 | 257.5665033592 |

Interpretation: the two internal render-quality numbers are nearly tied. Current branch has slightly lower PSNR/SSIM, slightly better LPIPS, and lower FPS.

## Post-Render Metrics

These values come from `render.py + metrics.py`.

| branch | iter | PSNR | SSIM | LPIPS | triangles | vertices |
|---|---:|---:|---:|---:|---:|---:|
| origin/main clean candidate | 2000 | 11.0476598740 | 0.2199306488 | 0.6417058110 | 39079 | 58458 |
| current branch engineering | 2000 | 11.5994377136 | 0.2702677548 | 0.6347319484 | 782982 | 820107 |

Interpretation: current branch is better on this post-render metric path, but uses far more triangles and vertices. This is not an efficiency-normalized win.

## COLMAP Geometry Proxy

COLMAP does not provide ground-truth scene geometry. This proxy compares rendered geometry against sparse COLMAP evidence and PCA-estimated local normals.

| branch | depth count | depth MAE | depth AbsRel | normal mean angle deg | normal median angle deg |
|---|---:|---:|---:|---:|---:|
| origin/main clean candidate | 19776 | 13.7902993339 | 5.6119052058 | 52.1989385790 | 52.9868341712 |
| current branch engineering | 21911 | 4.4141606252 | 0.4278796566 | 52.5651849634 | 53.6820151423 |

Interpretation: current branch is much better on the sparse depth proxy but slightly worse on normal-angle proxy. Because the proxy depends on sparse COLMAP visibility and point filtering, this supports diagnosis rather than a final paper claim.

## Cleanup Safety Check

The current-branch 2000 run did not execute destructive final cleanup:

- `final_cleanup_enabled`: `false`
- `cleanup_executed`: `false`
- triangles: `782982 -> 782982`
- vertices: `820107 -> 820107`

This confirms the earlier final-cleanup bug is not recurring in this medium run.

## Decision

Stage gate: `SOFT PASS`.

The medium baselines are now useful enough for engineering decisions and are W&B-recorded. They are not sufficient for a final method claim because:

1. the current branch is not yet a MeshPrior proposal-applied method at 2000 iterations;
2. the current branch's quality improvement is paired with a very large topology increase;
3. the true paper baseline still needs longer or budget-matched runs before headline claims.

Recommended next step: build a 2000-iteration MeshPrior variant with training-time W&B, using the same data and metric scripts, and include topology budget or cleanup control in the comparison.
