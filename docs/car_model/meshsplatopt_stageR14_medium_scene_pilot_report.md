# MeshSplatOpt Stage R14 Medium Scene Pilot Report

Date: 2026-05-02

## Gate

`STOP_BEFORE_GPU`.

## Reason

The prompt requires a medium public-scene pilot comparing full MeshSplatOpt repair against Stage35, delete-only PRISM, topology baselines, and certified giant-hole variants. The current codebase is not ready to produce that row honestly:

- R10 validates generic mesh edits but does not render calibration/held-out views.
- R11 recovery is a `SOFT PASS` cache contract, not a real teacher-guided recovery run.
- R12 portfolio/state machine operates on generic mesh arrays, not checkpoint/radiance state.
- There is no real-scene apply/recover/evaluate loop that can produce accepted MeshSplatOpt edits with independent render metrics.

Launching medium GPU training at this point would consume compute but not answer the R14 method question.

## Checks Performed

Repository status before R14 report:

```text
 ? submodules/effrdel
 ? submodules/simple-knn
?? docs/NeurIPSRepairPrompts.md
```

GPU availability check:

```text
0: 16946 / 49140 MiB, util 47%
1: 44508 / 49140 MiB, util 94%
2: 41159 / 49140 MiB, util 0%
3: 41150 / 49140 MiB, util 0%
4: 9895 / 49140 MiB, util 14%
5: 46258 / 49140 MiB, util 100%
6: 41672 / 49140 MiB, util 0%
7: 39858 / 49140 MiB, util 0%
```

If R14 were ready to run, GPU 4 would be the preferred low-occupancy choice at this check.

## Existing Baseline Evidence

Stage35 retained-refresh public-scene artifacts exist locally for `bonsai` and `courtyard` and remain valid baselines. They do not provide full MeshSplatOpt repair rows.

## Required Next Work Before Resuming R14

1. Add a checkpoint adapter that maps MeshSplatOpt edits to Mesh Splatting checkpoint geometry/state.
2. Add render-backed `counterfactual_edit_gate` integration using calibration/held-out views.
3. Run at least one real tiny teacher recovery using W&B online.
4. Produce real accepted/rejected edit audit trails for a public-scene checkpoint.
5. Only then launch medium 2000-iteration public-scene comparison with W&B.

## Decision

`STOP_BEFORE_GPU`.

Do not proceed to R15. R14 should resume after the real checkpoint/render/recovery integration exists.
