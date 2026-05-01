# MeshPrior Stage 7 Design — Conservative Snap Proposals

| Field | Value |
|---|---|
| Stage | M7 / conservative snap |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M6 synthetic damage benchmark |

## 1. Goal

M7 is the first MeshPrior stage that proposes vertex movement. It must be conservative and risk-gated.

The output is a snap proposal, not an unconditional scene edit. Real scene application is deferred to later scene gates.

## 2. Eligible Vertices

Default eligible vertices:

- not on boundary loops,
- not explicitly protected,
- not high observed support unless proposed displacement is tiny,
- in a low-to-moderate uncertainty region,
- finite field gradient available.

Boundary vertices are fixed by default.

## 3. Snap Direction

For occupancy fields, M7 uses the gradient of a surface objective:

```text
loss(x) = (sigmoid(f(x; z)) - iso_level)^2
direction = -grad_x loss
```

This moves points toward the occupancy iso-surface without assuming a signed-distance field.

## 4. Maximum Movement

Default max displacement is conservative:

```text
max_disp = 0.02 canonical units
```

The implementation clips every vertex displacement to `max_disp`.

Synthetic benchmark integration uses a stricter default:

```text
snap_max_disp = 0.005 canonical units
```

This keeps protect/prune visibility metrics stable while still allowing a measurable correction on vertex-noise cases.

## 5. Preservation Rules

- Boundary vertices remain fixed unless explicitly allowed.
- Protected vertices remain fixed.
- High observed-support vertices can only move under future tighter thresholds; initial implementation treats them as fixed if provided.
- High uncertainty sets displacement to zero.

## 6. Risk Evaluation

M7 reports:

- mean displacement,
- max displacement,
- moved vertex fraction,
- before/after analytic surface distance when a surface distance function is supplied,
- before/after free-space violation when a free-space violation function is supplied.

M7 also exposes an acceptance gate. A downstream optimizer should reject and therefore roll back a snap proposal when:

- free-space violation increases,
- visible-preservation drop is above the configured tolerance,
- a required analytic surface-distance improvement is absent.

If snap harms visible preservation by more than 5% in later synthetic benchmark integration, the snap line must stop and write failure analysis.

## 7. Stage Gate

M7 passes if:

- smoke test passes,
- snap reduces synthetic distance-to-surface,
- boundary/protected vertices stay fixed,
- no displacement exceeds `max_disp`,
- no scene geometry is modified by default.
