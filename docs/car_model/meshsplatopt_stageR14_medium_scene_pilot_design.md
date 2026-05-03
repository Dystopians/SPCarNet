# MeshSplatOpt Stage R14 Medium Scene Pilot Design

Date: 2026-05-02

## Intended Goal

Run the first medium-budget public-scene repair pilot after synthetic repair is validated.

## Required Scenes

Candidate scenes:

- Mip-NeRF 360 `bonsai`;
- ETH3D `courtyard`;
- `parking_phone_tiny`.

## Required Methods

1. current best clean / sparse-depth baseline;
2. Stage35 PRISM retained relaxed baseline;
3. delete-only PRISM-Budget baseline;
4. topology baseline from R6 where compatible;
5. MeshSplatOpt full repair without giant-hole fill;
6. MeshSplatOpt full repair with certified giant-hole fill.

## Required Metrics

- independent `render.py + metrics.py`: PSNR, SSIM, LPIPS;
- sparse COLMAP geometry proxy: AbsRel, DepthMAE, normal mean angle;
- triangle and vertex counts;
- defect counts and repair certificates;
- accepted/rejected edit table;
- W&B training/recovery links;
- runtime and memory where available.

## Pre-Launch Hard Requirement

R14 may not launch a medium public-scene GPU sweep until the full MeshSplatOpt path can apply, validate, and recover edits on a real Mesh Splatting checkpoint. Current blockers:

- R10 counterfactual gate is mesh-level/synthetic and does not call `render.py`;
- R11 teacher recovery is `SOFT PASS` contract-only, with no real recovery run;
- R12 state machine runs on generic numpy mesh states, not Mesh Splatting checkpoint state dicts;
- R13 benchmark is synthetic and not a substitute for public-scene render validation.

## GPU Policy

Before any future R14 GPU run:

```bash
nvidia-smi
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

As of this design check, GPU 4 is the relatively light option among available devices, but no R14 training was launched because the method path is not yet connected to real checkpoint rendering/recovery.

## Decision

`STOP_BEFORE_GPU`.

This is a methodological stop, not a repository failure. Running 2000-iteration public-scene training now would produce baseline/PRISM evidence, not a valid MeshSplatOpt full-repair comparison.
