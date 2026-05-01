# MeshPrior Stage 8 Design — Guarded Fill Proposals

| Field | Value |
|---|---|
| Stage | M8 / guarded fill |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M7 conservative snap |

## 1. Goal

M8 proposes local hole-fill patches under strict evidence gates. It must not become a general hallucination step.

The default output is a fill proposal. Scene application remains deferred to later evidence and rollback gates.

## 2. Hole-Boundary Detection

Boundary edges are mesh edges with exactly one incident face. Boundary loops are ordered from those edges.

Candidate loops are rejected when:

- they are too small to define a patch;
- ordering is ambiguous or non-manifold;
- the loop is too large relative to the mesh extent;
- multiple disconnected loops would need a single joint patch.

## 3. Local Field Extraction

For each candidate loop, M8 builds a local bounding box around the loop with configurable padding. The decoder is sampled on a small regular grid inside that box.

The first implementation stores sampled logits/support and uses them as evidence for whether a simple boundary-supported patch is allowed. Full local marching-cubes extraction is intentionally deferred until the scene gates can reject hallucinated surfaces.

## 4. Patch Clipping

The initial patch is a boundary fan:

```text
boundary loop vertices + one centroid vertex
```

The patch is clipped by construction:

- all new triangles use only loop vertices and the loop centroid;
- no far-away vertices are introduced;
- candidate patch extent must stay inside the padded loop bbox;
- patch support is scored from decoder values at the centroid and boundary samples.

## 5. Free-Space Avoidance

`evaluate_fill_risk` accepts an optional free-space violation function. A proposal is rejected when the patch increases mean free-space violation.

Until scene-level free-space evidence is wired in, synthetic benchmark free-space violation is explicitly reported as `0.0` only for controlled analytic cases.

## 6. Uncertainty and Ambiguity

Fill is rejected when:

- posterior uncertainty is above threshold;
- field support is below threshold;
- more than one plausible loop must be filled together;
- the loop geometry is non-manifold or cannot be ordered.

Ambiguous hidden-side completion is not solved in M8. Later stages may revive it with stronger evidence gates.

## 7. Rollback Conditions

A downstream optimizer must roll back a fill proposal when:

- boundary edge count does not decrease;
- disconnected components increase;
- free-space violation increases;
- visible preservation drops beyond tolerance;
- patch area is too large relative to the candidate hole boundary.

## 8. Stage Gate

M8 passes if:

- a controlled synthetic hole is closed;
- no disconnected far-away patch is created;
- benchmark free-space violation does not increase;
- the benchmark compares damaged input, guarded fill, and snap+fill.
