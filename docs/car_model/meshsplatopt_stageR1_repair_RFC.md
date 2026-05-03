# MeshSplatOpt Stage R1 Repair RFC

Date: 2026-05-02

## Gate

`PASS`.

This RFC separates pruning, repair, and hallucination risk, and defines MeshSplatOpt as a bidirectional repair method rather than a pruning schedule.

## Method Name

**MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting**

## Core Innovation

MeshSplatOpt introduces the **Counterfactual Surface Evidence Field (CSEF)**:

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

Every topology or geometry edit is proposed from this field and certified before commit. The field is not a triangle-pruning score. It is an edit-neutral representation that can support deletion, collapse, snapping, splitting, filling, protection, and appearance recovery.

The edit objective is:

```text
maximize  evidence_debt_reduction(edit)
        + render_quality_gain(edit)
        + geometry_consistency_gain(edit)
        - free_space_violation(edit)
        - hallucination_risk(edit)
        - topology_cost(edit)
```

subject to counterfactual render, geometry, changed-pixel, free-space, topology, and budget/state-machine gates.

## Why This Is Not Just Engineering

MeshSplatOpt reformulates mesh-splatting repair as **evidence-debt minimization under counterfactual validation constraints**. Existing PRISM asks which triangles can be safely removed. MeshSplatOpt asks which reversible local edit best reduces surface evidence debt while preserving held-out rendering, sparse geometry, free space, and topology integrity.

The edit proposals are not hard-coded patches. Each proposal is scored by:

- positive surface evidence;
- negative free-space evidence;
- explanation debt;
- prior support;
- topology cost;
- uncertainty.

The same calculus handles destructive and constructive actions:

- deletion and collapse repay topology cost where surface evidence is weak;
- snap/deform repairs supported but misaligned surfaces;
- split/subdivide allocates topology where the current mesh under-explains evidence;
- fill/patch repairs holes only when evidence certificates are strong enough;
- appearance recovery repairs radiance after geometry changes.

This makes the method a certified scene-repair optimizer, not an accumulation of pruning heuristics.

## Problem Classes

MeshSplatOpt targets scene defects that pruning-only methods cannot solve:

| problem | desired behavior |
|---|---|
| surface floaters | delete, collapse, or snap only if evidence supports removal or attachment |
| local dents / depressions | snap/deform toward local plane, sparse support, or prior surface |
| rough broken surfaces | smooth, snap, split, or locally remesh under render/geometry gates |
| car surface discontinuities | protect supported car surfaces and propose bounded snap/fill/split repairs |
| ground/wall misalignment | fit plane or height-field support and snap/deform conservatively |
| giant ground voids / parking-lot holes | fill only with boundary, plane, camera, semantic, or sparse support |
| appearance ghosting after geometry repair | run teacher-guided appearance reset/recovery on edited regions |

## Edit Operations

| operation | role | default risk |
|---|---|---:|
| `protect` | prevent supported regions from unsafe pruning or deformation | low |
| `delete/prune` | remove unsupported triangles, floaters, or topology debt | medium |
| `collapse/merge` | reduce redundant local topology while preserving support | medium |
| `snap/deform` | correct dents, rough surfaces, and misalignment | medium |
| `split/subdivide` | add topology where evidence debt is high | high |
| `fill/patch` | close small holes or certified large voids | high |
| `appearance reset/recovery` | recover color/radiance after accepted geometry edits | medium |

Every edit must carry an audit record, a rollback snapshot, evidence summary, risk summary, topology delta, and gate report.

## Evidence Certificates

MeshSplatOpt accepts an edit only through certificates that match the edit risk:

| certificate | purpose |
|---|---|
| render certificate | verifies held-out/calibration rendering does not regress beyond thresholds |
| sparse depth certificate | checks agreement with COLMAP sparse tracks or depth proxy |
| normal certificate | verifies local normal consistency and avoids severe orientation errors |
| free-space certificate | rejects surfaces that violate camera rays or known empty space |
| boundary-loop certificate | verifies hole loops are coherent, supported, and not open-world boundaries |
| semantic/object certificate | records car/object/ground support without allowing prior-only commits |
| plane/ground certificate | supports ground/wall repair through robust local plane or height-field evidence |
| uncertainty certificate | marks low-coverage and weak-evidence proposals as uncertain or diagnostic-only |

## Giant-Hole Policy

MeshSplatOpt separates three cases:

1. **Observed hole**: A hole with boundary-loop support, neighboring surface support, sparse/depth evidence, multi-view coverage, or semantic/plane support. It may be filled if render, geometry, free-space, topology, and uncertainty gates pass.
2. **Prior-supported void**: A weakly observed void with strong plane, object, or semantic prior support. It may produce a reversible proposal for diagnostics or visualization, but headline metrics must label it as prior-supported and uncertain.
3. **Unknown unobserved void**: A region outside trajectory or with insufficient evidence. It must not be silently filled in the main method.

Large parking-lot ground holes may be filled only when there is enough boundary, plane/height-field, multi-view, semantic, sparse, or neighboring-surface evidence. If the region is truly out-of-trajectory, the method must report uncertainty rather than claiming reconstruction.

## Paper Claim

> MeshSplatOpt repairs mesh-splatting scene geometry by proposing bidirectional topology/geometry edits from a surface evidence field and certifying them through counterfactual rendering and geometry validation.

The shorter framing is:

> MeshSplatOpt is not a better pruning heuristic; it is a counterfactually certified surface-repair optimizer.

## Baselines And Ablations

Required baselines:

- Mesh Splatting original;
- Stage35 PRISM retained relaxed baseline;
- delete-only PRISM;
- post-hoc mesh decimation / QEM;
- classical hole filling without render gate;
- plane fill without free-space gate;
- object prior fill without scene gate.

Required ablations:

- no teacher recovery;
- no CSEF debt term;
- no rollback;
- no negative free-space evidence;
- no sparse geometry gate;
- no changed-pixel gate;
- delete/collapse only;
- snap only;
- fill only;
- MeshSplatOpt without giant-hole fill;
- MeshSplatOpt with prior-only diagnostic fill, labeled separately.

## Kill Criteria

| condition | decision |
|---|---|
| method remains delete-only after R6 | `STOP` |
| giant-hole repair cannot produce a valid candidate on synthetic damage by R8 | `STOP` or demote hole repair |
| medium public scenes do not show repair-quality gains or topology-quality Pareto gains by R13/R14 | stop main-conference framing |
| prior-only fills are used as headline reconstruction evidence | `STOP` until reports are corrected |
| independent `render.py + metrics.py` and training metrics are mixed | `STOP` until tables are regenerated |

## Research Contract

MeshSplatOpt may use priors to propose edits, but scene evidence must dispose of them. A prior can raise candidate priority, initialize geometry, or explain uncertainty. It cannot directly commit geometry. Every accepted repair must leave:

- proposal JSON;
- before snapshot;
- after snapshot;
- rollback path;
- gate report;
- independent metrics when rendered;
- W&B link for training or recovery runs;
- explicit uncertainty and prior-only labels where applicable.
