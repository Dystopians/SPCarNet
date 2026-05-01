# MeshPrior Stage 11 Scene Experiment Report

| Field | Value |
|---|---|
| Stage | M11 / scene experiment |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage11_scene_experiment_design.md` |
| Run | `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun` |

## 1. Exact Commands

The command record was written to:

```text
outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/commands.sh
```

Commands executed:

```bash
git status --short
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
nvidia-smi
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_pipeline.py --scene_source synthetic --scene_model synthetic --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt --output_dir outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun --proposal_types protect prune fill --mode dry_run --require_gate_pass
micromamba run -n mesh_splatting python scripts/car_model/meshprior_collect_scene_experiment.py --run_dir outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun --gpu_used none
```

## 2. GPU Used

GPU used:

```text
none
```

`nvidia-smi` was visible only with elevated permissions. It showed GPU 1 and GPU 5 at 0 percent utilization, but every GPU already had active processes and memory allocations. To satisfy the "ensure GPU idle" requirement, no full training and no online wandb run were launched.

## 3. Runtime

The dry-run pipeline completed during the interactive command poll interval. Runtime was not separately instrumented with `/usr/bin/time`; observed wall-clock was under two seconds for the pipeline command.

## 4. Wandb

Wandb package availability:

```text
installed
```

Wandb run:

```text
not started
```

Reason: no fully idle GPU was available, and this M11 run used dry-run scene artifacts.

## 5. Metrics Table

Metrics source:

```text
outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/metrics.json
```

| Metric | Value |
|---|---:|
| `pipeline_status` | `PASS` |
| `proposal_count` | `1` |
| `accepted_count` | `1` |
| `rejected_count` | `0` |
| `triangle_count_delta_sum` | `4.0` |
| `boundary_edge_delta_sum` | `4.0` |
| `component_count_delta_max` | `0.0` |
| `floater_count_delta_max` | `0.0` |
| `free_space_violation_delta_max` | `0.0` |
| `COLMAP sparse AbsRel` | `null` |
| `sparse DepthMAE` | `null` |
| `sparse normal mean angle` | `null` |
| `PSNR / SSIM / LPIPS / MAE` | `null` |
| `controlled FPS` | `null` |

Accepted proposal:

```text
synthetic_fill_0000
```

Gate evidence:

- boundary edges improved from `4` to `0`;
- component count stayed `1`;
- floater delta stayed `0`;
- free-space violation delta stayed `0`.

## 6. Qualitative Artifacts

No rendered qualitative artifacts were generated. The run is topology/geometry dry-run only.

## 7. Failures and Suspected Causes

Real scene training/evaluation was not launched.

Cause:

- no fully idle GPU was available;
- no concrete real COLMAP scene/model pair was selected for M11;
- M10 runner currently provides dry-run proposal gates, not scene-optimizer application.

## 8. Decision

Decision:

```text
PASS for dry-run scene experiment
```

Continue to M12 with the following constraint: M12 should treat the current evidence as dry-run topology evidence only. It should not claim render/geometry metric improvement until a real scene checkpoint is evaluated.
