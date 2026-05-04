# Final Stage F1 Method Spec

Date: 2026-05-04

## Decision

`PASS`.

MeshSplatOpt can be explained as a compact-repair optimizer without overclaiming rejected edit primitives. The current load-bearing evidence is clean-to-compact topology optimization with strict recovery and certification. Snap, fill, object-prior, and giant-void operations remain optional repair branches until equal-budget controls prove their benefit.

## Title Options

1. **MeshSplatOpt: Counterfactually Certified Compact-Repair Optimization for Mesh Splatting**
2. **Evidence-Calibrated Compact Repair for Mesh Splatting**
3. **From Overcomplete Mesh Splatting to Compact Certified Scene Meshes**

Recommended title: **MeshSplatOpt: Counterfactually Certified Compact-Repair Optimization for Mesh Splatting**.

## One-Paragraph Story

MeshSplatOpt starts from the observation that strong Mesh Splatting checkpoints can become overcomplete: they render well, but carry excessive or poorly calibrated topology. The method builds a Counterfactual Surface Evidence Field (CSEF), proposes topology compaction or repair edits from scene evidence, certifies each accepted state with rollback-compatible counterfactual render and sparse-geometry gates, freezes topology during recovery, and reports the resulting quality/topology Pareto frontier against the strongest matched clean baseline. The currently validated branch is clean-to-compact optimization: R53.01 improves independent PSNR, SSIM, LPIPS, sparse depth, and normal agreement over the best parking clean-long baselines while removing 70 percent of triangles; public-scene matched evidence is positive on bonsai and mixed elsewhere, so the paper must present scene-sensitive certification rather than universal edit success.

## Main Claim

MeshSplatOpt improves Mesh Splatting by compacting overcomplete clean meshes into compact evidence-consistent topology, then recovering appearance and sparse geometry under strict topology freeze and counterfactual validation.

This is a compact-repair claim, not a local snap/fill claim. A compact-repair result is accepted only when it is compared against the strongest available matched clean baseline for the same scene and budget, using independent `render.py + metrics.py` metrics plus sparse COLMAP geometry when available.

## Core Abstraction: CSEF

The Counterfactual Surface Evidence Field is the common interface for compaction and repair:

```text
CSEF(x, n, region) = {
  positive_surface_evidence,
  negative_free_space_evidence,
  explanation_debt,
  prior_support,
  topology_cost,
  uncertainty
}
```

Current evidence-backed interpretation:

- `positive_surface_evidence`: multi-view visibility, sparse COLMAP support, low-reprojection-error tracks, stable render contribution, local normal consistency, boundary support.
- `negative_free_space_evidence`: camera rays or sparse observations that reject a proposed surface or topology edit.
- `explanation_debt`: pixels, sparse tracks, normals, or local regions that the current mesh under-explains.
- `prior_support`: object, ground, plane, smoothness, or semantic support used to rank proposals but not to commit geometry by itself.
- `topology_cost`: triangle, vertex, memory, and rendering cost introduced or removed by an edit.
- `uncertainty`: weak coverage, unreliable sparse tracks, large view disagreement, or prior-only support.

Edit acceptance maximizes evidence improvement while penalizing free-space violation, hallucination risk, and topology cost:

```text
maximize  evidence_debt_reduction(edit)
        + render_quality_gain(edit)
        + geometry_consistency_gain(edit)
        - free_space_violation(edit)
        - hallucination_risk(edit)
        - topology_cost(edit)
```

subject to render, geometry, changed-pixel, free-space, rollback, topology, and budget gates.

## Algorithm

**A. Train or load a strong clean long baseline.**

Use the best available clean Mesh Splatting checkpoint for the scene. The comparison baseline must be the strongest clean long or matched continuation row, never a short clean run when the method has a longer budget.

**B. Build CSEF.**

Gather render evidence, checkpoint topology, sparse COLMAP tracks, visibility and reprojection confidence, local normal/depth agreement, topology cost, and uncertainty flags. Area can be one signal, but not the whole method.

**C. Generate compaction and repair edit candidates.**

The current validated candidate family is clean-to-compact topology reduction:

```text
clean 22k
-> prune 65/70/75/80/90 percent by evidence-compatible criterion
-> checkpoint materialization
-> strict topology freeze
-> recovery
-> independent render and sparse-geometry evaluation
```

Future repair candidate families include snap, fill, collapse, split, object-prior repair, and ground-void repair, but they must pass the same counterfactual gates before promotion.

**D. Apply counterfactual certification and rollback.**

Every accepted candidate needs an audit trail: proposal JSON, before snapshot, after snapshot, rollback path, topology delta, risk summary, and independent validation when rendered. Rejected candidates remain first-class evidence.

**E. Run strict topology-frozen recovery.**

Use `--freeze_topology_updates --skip_restricted_delaunay` for fixed-topology recovery. `--skip_restricted_delaunay` alone is not enough because the standard pruning/densification branch can still change topology.

**F. Report Pareto frontier and repair diagnostics.**

Report quality versus triangles, not only a single best row. Separate training-time metrics from independent metrics. Separate oracle diagnostics, prior-only diagnostics, and inference-time results.

## Current Load-Bearing Empirical Branch

The main empirical branch is clean-to-compact recovery.

| row | role | status |
| --- | --- | --- |
| clean 22k / clean 30k | strongest parking clean-long baselines | required baseline |
| R44.01 | historical sparse-depth long row | render-losing vs clean 22k; topology/normal Pareto only |
| R48.01 prune80 recovery | compact Pareto row | beats clean 22k on PSNR/SSIM/depth while using 20 percent of triangles; LPIPS slightly worse |
| R53.01 prune70 recovery | headline parking row | beats clean 22k/30k on all tracked independent metrics while using 30 percent of triangles |
| R55.01 prune65 recovery | LPIPS/normal Pareto row | strongest LPIPS/normal among promoted compact rows, with more triangles |
| R58 bonsai prune70 matched | public-scene all-metric positive | supports transfer but not universality |
| R57 courtyard / R60 counter | controlled negatives | show area-only compaction needs a selector |
| R59 room | render-positive, geometry-negative | useful Pareto evidence, not all-metric pass |

The strongest validated paper result is R53.01. It supersedes R44 because R44 loses to clean 22k on render and depth.

## Repair Branch Policy

Snap, fill, object-prior, and ground-void edits are optional branches until they pass equal-budget controls. They should be reported as:

- reversible edit interfaces;
- safety and rollback infrastructure;
- diagnostics for future repair;
- negative evidence where equal-budget controls fail.

They should not be described as the current source of the main quality gain. A repair primitive can become a headline result only if it beats a matched no-edit or clean baseline under independent metrics and sparse geometry, with prior-only support clearly labeled.

## What Not To Claim

- Do not claim current snap/fill improves full-budget quality.
- Do not claim R44 beats clean long on render.
- Do not compare a long method run against clean 7k as the headline baseline.
- Do not claim prior-only giant void fill reconstructs observed geometry.
- Do not hide scene-specific sparse sampling or compaction tuning.
- Do not claim universal cross-scene clean-to-compact dominance from the current evidence.

## Novelty Statement

MeshSplatOpt is not training-time pruning alone, sparse depth alone, or posthoc decimation alone. The contribution is the certified compact-repair loop:

```text
CSEF-scored edit proposal
-> counterfactual gate and rollback
-> topology-frozen recovery
-> sparse-geometry-guided validation
-> quality/topology Pareto certification
```

This loop makes topology reduction and repair auditable against downstream rendering and sparse geometry, rather than treating triangle count as a standalone compression target.

## Required Baselines

- Clean Mesh Splatting long baseline.
- Clean Mesh Splatting same-iteration continuation.
- Stage35 PRISM.
- Delete-only PRISM.
- Clean-to-compact area prune without recovery.
- Clean-to-compact with recovery.
- Sparse-depth-only recovery.
- Posthoc simplification, QEM, or area prune.
- No-freeze recovery.
- Snap/fill branches if promoted.

## Required Figures And Tables

- Method diagram: clean baseline, CSEF, candidate proposal, gate, rollback, topology-frozen recovery, certification.
- CSEF/edit calculus diagram: evidence, debt, topology cost, uncertainty, and gate outcomes.
- Pareto frontier: independent quality metrics versus triangle count.
- Clean long versus R53 qualitative montage.
- Failure table: R44, R45/R46, R51/R52, R56, R57, R60, snap/fill controls.
- Ablation table: no recovery, no freeze, sparse-depth-only, area-prune-only, compaction budgets, selector variants.

## F1 Gate

`PASS`.

The method can be explained in one paragraph, and the story does not overclaim rejected edit primitives. Proceed to F2 baseline registry and metric-integrity collector.
