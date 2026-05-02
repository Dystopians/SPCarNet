# MeshPrior Stage 19 Clean Baseline Audit

Date: 2026-05-01

## Goal

Confirm whether `origin/main@1a714f3` is a valid clean MeshSplatting baseline for the parking experiments.

## Remote Audit

Local remotes:

```text
origin  https://github.com/Dystopians/mesh-.git
spcarnet https://github.com/Dystopians/SPCarNet.git
```

Official MeshSplatting remote check:

```bash
git ls-remote https://github.com/meshsplatting/mesh-splatting.git HEAD refs/heads/main refs/heads/master
```

Result:

```text
1a714f33dd758a42be8fa86e1041c3c67df0d0a8 HEAD
1a714f33dd758a42be8fa86e1041c3c67df0d0a8 refs/heads/main
```

Local `origin/main` is also:

```text
1a714f33dd758a42be8fa86e1041c3c67df0d0a8
```

Therefore, the already used clean baseline worktree at `/tmp/mesh-splatting-origin-main` was based on the same commit as the official MeshSplatting `main` branch at audit time.

## README Evidence

`origin/main:README.md` states that the repository is the official implementation for:

```text
MeshSplatting: Differentiable Rendering with Opaque Meshes
```

It also instructs users to clone:

```bash
git clone https://github.com/meshsplatting/mesh-splatting --recursive
```

This matches the remote audited above.

## Current Branch Difference

The current `clean-submit` branch contains substantial SP-CarNet / MeshPrior additions and changes to the training stack. Relevant changed files relative to `origin/main` include:

- `train.py`
- `arguments/__init__.py`
- `scene/__init__.py`
- `scene/cameras.py`
- `scene/dataset_readers.py`
- `scene/triangle_model.py`
- `evaluate_geometry_colmap.py`
- `scripts/car_model/*`
- `ss3dm_prior/*`
- `docs/car_model/*`

This confirms that `clean-submit` is not a clean MeshSplatting baseline and should not be used as the paper baseline.

## Existing Baseline Runs

The clean baseline runs already completed from the isolated worktree:

- worktree: `/tmp/mesh-splatting-origin-main`
- commit: `origin/main@1a714f33dd758a42be8fa86e1041c3c67df0d0a8`
- 2000-iteration output: `outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model`
- W&B external summary: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`

The clean script lacks current-branch training-time W&B integration, so the external summary record remains the correct logging method for this historical baseline unless W&B support is backported to a separate official-baseline worktree.

## Baseline Metrics

Training-internal test metrics at iteration `2000`:

- PSNR: `16.4619565010`
- SSIM: `0.4846517714`
- LPIPS: `0.5333475658`
- FPS: `271.3129810583`

Post-render metrics at iteration `2000`:

- PSNR: `11.0476598740`
- SSIM: `0.2199306488`
- LPIPS: `0.6417058110`
- triangles: `39079`
- vertices: `58458`

COLMAP geometry proxy at iteration `2000`:

- depth MAE: `13.7902993339`
- depth AbsRel: `5.6119052058`
- normal mean angle: `52.1989385790`

## Decision

Stage gate: `PASS`.

`origin/main@1a714f33dd758a42be8fa86e1041c3c67df0d0a8` is confirmed as the official MeshSplatting `main` commit at audit time and is a valid clean baseline for the current parking experiments.

Remaining caveat: this is still only a 2000-iteration medium-budget run on one scene. It is clean and valid, but not yet a long-budget or multi-scene paper baseline.
