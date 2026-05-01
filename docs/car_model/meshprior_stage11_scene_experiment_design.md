# MeshPrior Stage 11 Design — Scene Experiment

| Field | Value |
|---|---|
| Stage | M11 / scene experiment |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M10 dry-run pipeline |

## 1. Scene Selection

Primary experiment scene:

```text
synthetic local-hole scene from the M10 pipeline
```

Reason:

- M10 already produces a controlled before/after proposal set;
- M9 scene gates can evaluate accepted/rejected proposals;
- no repository-local real COLMAP scene/model pair is currently selected;
- GPU inventory shows no fully idle GPU, so real training is deferred.

If a real scene path is later selected, it must provide:

- COLMAP source directory;
- trained mesh-splatting model directory;
- available checkpoint iteration;
- test split if rendering or COLMAP geometry eval is required.

## 2. Valid Baselines

This M11 run records these groups:

- original synthetic damaged mesh baseline;
- baseline + geometry/topology eval only;
- region mining artifact;
- protect/prune proposal dry-run;
- fill proposal accepted by scene gate.

Not run in this stage:

- full scene training;
- online wandb logging;
- real render PSNR/SSIM/LPIPS;
- COLMAP sparse geometry eval.

Reason: no fully idle GPU was available at launch time.

## 3. Checkpoints / Iterations

Synthetic dry-run has no scene checkpoint. The run records:

```text
checkpoint: dry_run_synthetic
iteration: 0
```

For future real-scene experiments, compare:

- loaded baseline checkpoint;
- post-proposal accepted dry-run state;
- optional post-recovery training checkpoint.

## 4. Primary Metrics

Available in this run:

- triangle count before/after;
- boundary edge count before/after;
- hole-boundary score;
- connected component delta;
- floater count delta;
- free-space violation delta;
- accepted/rejected proposal counts.

Unavailable in this run:

- COLMAP sparse AbsRel;
- sparse DepthMAE;
- sparse normal mean angle;
- PSNR / SSIM / LPIPS / MAE;
- controlled FPS.

Unavailable metrics are reported as `null`, not omitted.

## 5. Wandb Naming

If online wandb is used later:

```text
project: spcarnet_meshprior
run name: meshprior_m11_<scene>_<mode>_<YYYYMMDD_HHMMSS>
```

For this run, wandb is not started because no fully idle GPU is available. The required `wandb_url.txt` is therefore omitted.

## 6. Command Sequence

Preflight:

```bash
git status --short
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
nvidia-smi
```

Smoke:

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
```

Experiment:

```bash
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_pipeline.py \
  --scene_source synthetic \
  --scene_model synthetic \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --output_dir outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun \
  --proposal_types protect prune fill \
  --mode dry_run \
  --require_gate_pass
```

Then collect `metrics.json` and `summary.md` from the pipeline artifacts.

## 7. Decision Rule

PASS if:

- dry-run scene experiment completes;
- metrics are copied without cherry-picking;
- at least one proposal is accepted by scene evidence;
- no disconnected floater or free-space regression is introduced.

SOFT FAIL if all proposals are rejected.
