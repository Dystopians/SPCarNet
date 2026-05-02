# Stage26 Cross-Scene Method Evidence Report

Date: 2026-05-02

## Summary

M26 is a `SOFT PASS`.

The current PRISM topology-retention method trains and validates on two public COLMAP-style scenes: Mip-NeRF 360 `bonsai` and ETH3D `courtyard`. Both scenes are geometry-observable under the PRISM sparse-COLMAP validation path. Against an aligned current-branch sparse-depth baseline, PRISM keeps or improves most render metrics and records accepted topology edits on both scenes.

The result is not yet a final paper claim. The 2000-iteration cross-scene schedule uses late PRISM and a small `0.5%` candidate ratio, so direct candidate-edit topology reduction is small. Some larger checkpoint-topology deltas come from schedule effects and final-checkpoint accounting, not only from candidate deletion. The next step should tune the cross-scene schedule before launching expensive full-budget sweeps.

## Evidence Root

- output root: `outputs/carnet/meshprior/stage26_cross_scene/`
- collector: `scripts/car_model/meshprior_collect_stage26_cross_scene.py`
- summary JSON: `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_summary.json`
- summary CSV: `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_runs.csv`
- paired deltas CSV: `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_paired_deltas.csv`
- generated summary MD: `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_summary.md`

## Runs

All four training runs used online W&B, `CUDA_VISIBLE_DEVICES=1`, `2000` iterations, sparse COLMAP depth loss, and no final cleanup pruning. PRISM rows additionally used the M24.2 freeze-after-first-commit topology-retention schedule.

| scene | variant | W&B | training PSNR/SSIM/LPIPS | independent PSNR/SSIM/LPIPS | W&B triangles | checkpoint triangles | PRISM decisions | validation |
|---|---|---|---|---|---:|---:|---|---|
| Mip-NeRF 360 `bonsai` | baseline sparse depth | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xdct9uys` | `23.6034 / 0.7977 / 0.2422` | `12.2016 / 0.2073 / 0.6243` | `1357104` | `2487474` | n/a | n/a |
| Mip-NeRF 360 `bonsai` | M24.2 PRISM | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dmasxcej` | `23.6994 / 0.8004 / 0.2386` | `12.1712 / 0.2378 / 0.6182` | `1350319` | `1350319` | `1` commit, `3` rollback, `2` no-candidate retries | `4/4` observable, `2/4` pass |
| ETH3D `courtyard` | baseline sparse depth | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mdan8yc2` | `19.6829 / 0.6507 / 0.4295` | `14.9462 / 0.4388 / 0.5924` | `220339` | `410254` | n/a | n/a |
| ETH3D `courtyard` | M24.2 PRISM | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r9zgtuyp` | `19.6932 / 0.6518 / 0.4284` | `15.0614 / 0.4734 / 0.5838` | `217058` | `217058` | `3` commits, `0` rollback, `4` no-candidate retries | `5/5` observable, `3/5` pass |

## Paired Deltas

Positive PSNR/SSIM is better. Negative LPIPS is better.

| scene | training delta | independent delta | W&B triangle delta | checkpoint triangle delta | interpretation |
|---|---|---|---:|---:|---|
| Mip-NeRF 360 `bonsai` | `+0.0960 PSNR`, `+0.0027 SSIM`, `-0.0036 LPIPS` | `-0.0304 PSNR`, `+0.0305 SSIM`, `-0.0060 LPIPS` | `-0.50%` | `-45.72%` | render quality is mostly retained/improved, but direct W&B topology reduction is tiny; checkpoint accounting differs from runtime W&B topology. |
| ETH3D `courtyard` | `+0.0103 PSNR`, `+0.0011 SSIM`, `-0.0011 LPIPS` | `+0.1152 PSNR`, `+0.0347 SSIM`, `-0.0087 LPIPS` | `-1.49%` | `-47.09%` | cross-scene positive render result; three accepted PRISM edits, but W&B topology reduction is still modest. |

## PRISM Decisions

Mip-NeRF 360 `bonsai`:

- iteration `1751`: committed candidate prune, `1357104 -> 1350319`
- iterations `1842`, `1852`: no candidate
- iterations `1862`, `1863`, `1864`: counterfactual rollback

ETH3D `courtyard`:

- iteration `1751`: committed candidate prune, `220345 -> 219244`
- iterations `1842`, `1852`: no candidate
- iteration `1862`: committed candidate prune, `219244 -> 218148`
- iterations `1953`, `1963`: no candidate
- iteration `1973`: committed candidate prune, `218148 -> 217058`

## Validation Notes

Both PRISM rows were geometry-observable:

- `bonsai`: `4` validation snapshots, all geometry-observable, `2` passed the PRISM validation gate.
- `courtyard`: `5` validation snapshots, all geometry-observable, `3` passed the PRISM validation gate.

No Tanks and Temples geometry claim is made in M26. The current Tanks mirror path remains trainable but not sparse-geometry-validatable because it lacks true COLMAP 2D-3D tracks.

## Commands

Exact training commands are recorded in:

- `outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_baseline_sparse_depth_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage26_cross_scene/mipnerf360_bonsai_m24_2_prism_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage26_cross_scene/eth3d_courtyard_baseline_sparse_depth_2000iter/logs/train_command.txt`
- `outputs/carnet/meshprior/stage26_cross_scene/eth3d_courtyard_m24_2_prism_2000iter/logs/train_command.txt`

Independent render metrics were generated with:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -m <model> --eval --iteration 2000 --skip_train --quiet

CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m <model>
```

Collector:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_collect_stage26_cross_scene.py
```

## Gate

`SOFT PASS`.

M26 verifies that the current method runs across two public geometry-observable scenes with W&B, independent render metrics, PRISM decisions, checkpoint accounting, and validation reports. It does not yet prove the final NeurIPS-strength claim because the direct W&B topology reduction is only `0.5%` to `1.5%` at this medium budget, and baseline checkpoint-topology accounting still differs from runtime W&B topology. The next prompt should tune cross-scene topology pressure and metric accounting before full-budget claims.
