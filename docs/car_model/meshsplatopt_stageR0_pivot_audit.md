# MeshSplatOpt Stage R0 Pivot Audit

Date: 2026-05-02

## Gate

`PASS`.

The repository compiles, the pivot is explicitly bidirectional repair rather than pruning-only, and the next stage should proceed.

## Repository State

| item | value |
|---|---|
| branch | `neurips-meshsplatopt-repair` |
| commit | `6344a0c` |
| python | `Python 3.13.2` |
| compile gate | `PASS`: `python -m compileall scripts/car_model ss3dm_prior utils -q` |

Dirty files at audit time:

```text
 ? submodules/effrdel
 ? submodules/simple-knn
?? docs/NeurIPSRepairPrompts.md
```

The dirty state is limited to untracked submodule working trees and the new prompt file. No tracked source file was dirty before this audit.

## Documents Read

- `docs/car_model/reports/meshprior_prism_deep_retrospective.md`
- `docs/car_model/reports/meshprior_prism_final_handoff.md`
- `docs/car_model/reports/meshprior_prism_reviewer_risk_checklist.md`
- `docs/car_model/meshprior_stage1_scene_meshprior_RFC.md`
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`
- `docs/car_model/meshprior_stage24_2_topology_retention_report.md`
- `docs/car_model/meshprior_stage35_retained_refresh_report.md`
- `docs/car_model/meshprior_stage36_metric_reconciliation_report.md`
- `docs/prompts.md`
- `docs/car_model/meshprior_remaining_work_prompts.md`

## Current PRISM Strengths

PRISM has strong infrastructure:

- opt-in training-time topology edits;
- proposal, gate, rollback, and retained-topology audit records;
- W&B logging for meaningful training runs;
- final cleanup and topology accounting fixes;
- independent `render.py + metrics.py` evaluation discipline;
- sparse COLMAP geometry proxy evaluation;
- reproducible report collectors and failure-case documentation;
- Stage24.2 single-scene topology-retention evidence with a large topology drop and no render regression;
- Stage35 public-scene retained-relaxed evidence with explicit active versus rolled-back edit accounting.

These are useful baselines and reusable safety mechanisms for MeshSplatOpt.

## Current PRISM Weaknesses

PRISM is still too narrow for a top-tier method claim:

- The dominant operation is delete/prune; the system is not yet a general scene repair optimizer.
- Public-scene effect sizes are small, especially `bonsai` Stage35 versus Stage33.
- `courtyard` improves PSNR/SSIM/topology but has an LPIPS tradeoff.
- Existing gates certify conservative removals better than constructive repairs.
- The old object-prior narrative is mostly proposal infrastructure, not a demonstrated repair method.
- Large holes, dents, rough surfaces, ground/wall misalignment, vehicle discontinuities, and appearance recovery are not first-class optimized operations.
- Current novelty risks being read as an over-engineered pruning controller unless reframed around a unified evidence field and reversible edit calculus.

## Why Stage35 Is A Baseline, Not The Final Method

Stage35 is a strong safety and accounting baseline because it proves retained relaxed topology edits can survive final validation and be audited. It is not the final method because:

- it remains delete-centric;
- the `bonsai` improvement is about `+0.067 dB` PSNR and `-512` triangles versus Stage33, too small for a main method claim;
- it does not add or deform geometry to repair missing scene evidence;
- it does not address giant ground voids or constructive hole completion;
- it does not provide a unified calculus across delete, collapse, snap, split, fill, and appearance recovery.

Future comparisons must keep Stage35 as a named baseline: conservative retained PRISM, not MeshSplatOpt.

## Required New Operations

MeshSplatOpt must support bidirectional, reversible scene surgery:

| operation | role |
|---|---|
| `protect` | preserve supported geometry from unsafe edits |
| `delete/prune` | remove unsupported floaters and redundant topology |
| `collapse/merge` | reduce topology while preserving supported surfaces |
| `snap/deform` | correct dents, rough surfaces, and plane/object misalignment |
| `split/subdivide` | allocate topology where the current mesh under-explains evidence |
| `fill/patch` | repair small holes and certified giant ground voids |
| `appearance reset/recovery` | restore radiance after topology or geometry repair |

Every operation must be reversible or explicitly fail its stage gate.

## Pivot Lock

The new method is:

> MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting.

The core abstraction is the Counterfactual Surface Evidence Field (CSEF), which scores positive surface evidence, negative free-space evidence, explanation debt, prior support, topology cost, and uncertainty. Edits are proposed from this field and committed only when counterfactual render, geometry, changed-pixel, free-space, topology, and budget gates pass.

This is not a better pruning schedule. It is a constrained repair optimizer whose edit space includes destructive and constructive operations.

## Go / No-Go

Recommendation: `PROCEED_TO_R1`.

Rationale: the codebase has enough PRISM safety infrastructure to support the pivot, but the old empirical story is insufficient. The next step should be paper-facing RFC and novelty lock before major implementation.
