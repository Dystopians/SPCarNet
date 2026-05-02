# MeshPrior Stage 21 Long-Budget Single-Scene Diagnostic Design

Date: 2026-05-02

## Scope

Stage 21 runs a single-scene long-budget diagnostic on `parking_phone_tiny`. This stage is explicitly not a final paper claim because M20 found no second suitable parking-lot scene.

## Budget

Use `7000` iterations. This is the smallest long-budget option listed in the remaining-work prompts and is appropriate for a first aligned diagnostic before any sweep.

## Runs

All runs use the same dataset view:

```text
outputs/carnet/meshprior/parking_phone_tiny/dataset_view
```

Run roots:

- clean official MeshSplatting: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/origin_main_7000iter/model`
- current branch engineering: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/current_branch_7000iter/model`
- MeshPrior variant: `outputs/carnet/meshprior/parking_phone_tiny/stage21_long_budget/stage17_meshprior_7000iter/model`

The MeshPrior variant is initialized from the same MeshPrior-cleaned checkpoint used in Stage17 at `iteration_200`, then resumed to `7000`.

## W&B

- current branch and MeshPrior runs must use training-time online W&B.
- clean official MeshSplatting `origin/main` lacks current W&B flags and must be externally logged immediately after training.

## Metrics

For every completed run:

- training internal test metrics;
- `render.py + metrics.py`;
- `evaluate_geometry_colmap.py`;
- checkpoint triangle and vertex counts;
- final-cleanup summary where available;
- W&B URL or external W&B summary URL.

## Gate

`PASS` if all three aligned runs finish and comparison artifacts are generated.

`SOFT PASS` if at least one aligned pair finishes and missing rows are documented.

`FAIL` if logs, checkpoints, W&B records, or metric JSONs are missing for a claimed completed row.
