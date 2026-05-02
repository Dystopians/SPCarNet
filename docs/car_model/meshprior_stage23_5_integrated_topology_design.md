# MeshPrior Stage 23.5 Integrated Topology-Control Design

Date: 2026-05-02

## Task Definition

After the user's clarification, the task is:

> optimize a MeshSplatting scene mesh from posed images and COLMAP evidence, not pure radar-only mesh reconstruction.

The input assumption is reasonable for this codebase:

- posed multi-view images;
- COLMAP cameras / sparse points;
- an initial MeshSplatting triangle-splat scene;
- optional object/shape priors and region evidence.

## Goal

Move M21.5 from post-hoc checkpoint-copy pruning toward training-time topology control.

This stage uses the existing PRISM scheduler in `train.py` as the minimal integrated controller:

1. collect triangle statistics during training;
2. propose low-value triangle pruning;
3. run counterfactual gate on calibration views;
4. save pre/post round checkpoints and metadata;
5. recover/fine-tune after accepted topology edits;
6. preserve final-cleanup audit and W&B logging.

## Smoke Scope

The first smoke is intentionally short: `800` iterations on `parking_phone_tiny`.

It is not expected to beat the 7000-iteration runs. It verifies that the integrated topology-control mechanism runs end-to-end on the clarified camera/COLMAP MeshSplatting task.

## Gate

`PASS` if the smoke produces:

- W&B training run;
- PRISM round metadata;
- final cleanup summary;
- checkpoint at the target iteration;
- independent render metrics;
- COLMAP proxy geometry metrics;
- collector summary.

`SOFT PASS` if training/eval completes but no prune is committed because the gate rejects it.

`FAIL` if training crashes, W&B is absent without documented fallback, or collector/eval artifacts are missing.
