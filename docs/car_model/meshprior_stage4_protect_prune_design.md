# MeshPrior Stage 4 Design — Safe Protect/Prune Proposals

| Field | Value |
|---|---|
| Stage | M4 / protect-prune proposals |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M3 scene-region posterior inference |

## 1. Goal

M4 introduces the first safe MeshPrior proposal types:

- `protect`: mark valid object-like triangles as geometry that scene pruning/compaction should preserve.
- `prune`: mark unsupported or object-prior-inconsistent triangles as removable candidates.

M4 does not move vertices, does not add geometry, and does not fill holes.

## 2. Proposal Definitions

For each mined region with a posterior latent:

1. sample points on each triangle,
2. query the SP-CarNet shape field at those points,
3. aggregate triangle-level support and violation scores,
4. lower confidence when posterior/canonicalization uncertainty is high,
5. emit triangle-level scores and proposal records.

## 3. Triangle Scoring

For triangle `t` with samples `x_t`:

```text
surface_support_t = mean(sigmoid(f(x_t; z)))
prior_violation_t = 1 - surface_support_t
observed_support_t = region evidence or 1.0 when unavailable
uncertainty_penalty = clamp(region_uncertainty, 0, 1)

protect_score_t = surface_support_t * observed_support_t * (1 - uncertainty_penalty)
prune_score_t = clamp(prior_violation_t + free_space_violation_t + low_observed_support_t - protect_score_t, 0, 1)
```

M4 currently sets `free_space_violation_t=0` unless explicit free-space samples are provided by a later stage.

## 4. Shape-Field Support

The support function consumes either:

- a real `SPCarShapeFieldDecoder` plus latent `z`, or
- an analytic callable used by smoke tests.

The output is interpreted as occupancy logits and converted to probabilities with sigmoid.

## 5. Free-Space Violation

If free-space queries are available later, a triangle receives higher prune score when its samples lie in known free-space and the object field claims support there. M4 leaves this hook explicit but inactive for the initial implementation.

## 6. Uncertainty Handling

Uncertainty lowers both protect confidence and aggressive prune confidence:

- posterior uncertainty from M3,
- canonicalization confidence,
- missing region evidence.

The default M4 implementation accepts a scalar region uncertainty and clamps it to `[0, 1]`.

## 7. Downstream Contract

Outputs:

```text
triangle_scores.npz
proposals.json
summary.csv
```

`triangle_scores.npz` is optimizer-neutral. PRISM integration is deferred to M5.

## 8. Stage Gate

M4 passes only if:

- smoke test passes,
- synthetic valid object-surface triangles receive higher protect scores,
- synthetic floater triangles receive higher prune scores,
- no vertex movement or fill is performed,
- the score contract can be loaded by a downstream optimizer.
