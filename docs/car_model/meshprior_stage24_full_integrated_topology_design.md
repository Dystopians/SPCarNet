# Stage24 Full Integrated Topology-Control Design

Date: 2026-05-02

## Goal

Run a full-budget integrated topology-control experiment on `parking_phone_tiny` and compare it against the already completed clean/current 7000-iteration baselines.

The target remains posed multi-view images plus COLMAP/camera geometry plus Mesh Splatting scene optimization.

## Why Stage24 Follows Stage23.6

Stage23.6 v1 failed as a useful method row because `orientation_keep=1.0` protected every triangle when the threshold was `0.85`.

Stage23.6 v2 fixed the schedule by setting `--prism_keep_orientation_threshold 1.1`. It committed two training-time PRISM topology edits with counterfactual acceptance and no rollback:

- iteration `551`: `64497 -> 63853`
- iteration `922`: `63853 -> 63215`

The 2000-iteration result improved over the current-branch 2000 row while using substantially fewer triangles, but it remained below Stage17 quality and exposed topology growth after later densification. Therefore Stage24 should keep topology-control active across the full 7000-iteration budget instead of only early training.

## Stage24 Run

- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage24_full_integrated_topology/full_v1_7000iter/`
- iterations: `7000`
- GPU: `1`
- W&B project: `spcarnet_meshprior`
- W&B group: `parking_stage24_full_integrated_topology`
- W&B name: `full_v1_7000iter`

PRISM schedule:

- geometry acquisition until iteration `300`
- stats collection `250`
- dead rounds `0`
- candidate rounds `18`
- candidate prune ratio per round `0.015`
- recovery after commit `250`
- post-commit recollect `120`
- score recompute interval `100`
- final cleanup disabled

The repeated candidate rounds are intentional: the 2000-iteration run showed later densification can add topology after early PRISM edits, so the full run must continue scheduled topology decisions through most of training.

## Baselines

Use the existing W&B-recorded 7000-iteration baselines:

- clean `origin/main`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n`
- current branch: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m`
- failed Stage17 MeshPrior resume: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w3kczubb`

Do not rerun these unless their artifacts are missing. They already have render metrics, COLMAP proxy geometry, topology counts, and W&B records.

## Gate

`PASS` if full training completes with online W&B, at least one PRISM topology edit commits or rejects with explicit metadata, final cleanup is recorded, independent render metrics and COLMAP proxy geometry exist, and the report compares against clean/current 7000 baselines with topology visible.

`SOFT PASS` if the mechanism works but the result is not better than the topology-controlled diagnostic row.

`FAIL` if training crashes, W&B is missing without fallback, or final evaluation artifacts cannot be generated.
