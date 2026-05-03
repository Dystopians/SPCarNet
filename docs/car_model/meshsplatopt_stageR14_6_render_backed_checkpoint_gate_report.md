# MeshSplatOpt Stage R14.6 Render-Backed Checkpoint Gate Report

Date: 2026-05-02

## Gate

`PASS`.

R14.6 adds a reusable checkpoint-level gate that compares a baseline model and an edited candidate using independent render metrics, sparse COLMAP geometry metrics, and checkpoint topology. It can also materialize a candidate model from a checkpoint plus `MeshEdit` JSON.

## Implementation

New entrypoint:

```bash
scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py
```

The script:

- optionally applies an edit into a checkpoint-copy candidate model;
- records `nvidia-smi`;
- runs or reuses `render.py`;
- runs or reuses `metrics.py`;
- runs or reuses `evaluate_geometry_colmap.py`;
- reads checkpoint triangle/vertex counts;
- writes `render_backed_checkpoint_gate_report.json`;
- fails closed if render or geometry evidence is missing.

## Validation Command

```bash
CUDA_VISIBLE_DEVICES=4 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py \
  --baseline_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model \
  --candidate_model outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/model \
  --output_root outputs/carnet/meshsplatopt/stageR14_6_render_backed_checkpoint_gate \
  --iteration 200 \
  --gpu 4 \
  --max_points_per_view 500
```

## Result

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| triangles | `64497` | `64498` | `+1` |
| vertices | `193491` | `193494` | `+3` |
| PSNR | `10.949986457824707` | `10.949986457824707` | `0.0` |
| SSIM | `0.2898596525192261` | `0.2898596525192261` | `0.0` |
| LPIPS | `0.6441746354103088` | `0.6441746354103088` | `0.0` |
| AbsRel | `0.32417137460470213` | `0.32417137460470213` | `0.0` |
| Depth MAE | `3.6485552222775537` | `3.6485552222775537` | `0.0` |
| normal mean deg | `51.68797353552561` | `51.68793149935674` | `-0.00004203616886400141` |

## Decision

`PASS`.

The current fill dry-run candidate is accepted by render-backed checkpoint validation. This is still a path-validation result, not a repair-quality claim. The remaining R14 blockers are real edit selection on public scenes, teacher recovery around accepted edits, and a medium-budget W&B-logged comparison against Stage35/PRISM baselines.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR14_6_render_backed_checkpoint_gate/render_backed_checkpoint_gate_report.json`
- `outputs/carnet/meshsplatopt/stageR14_6_render_backed_checkpoint_gate/logs/nvidia_smi.log`
