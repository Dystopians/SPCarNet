# MeshPrior Stage 9 Design — Scene Gates and Rollback

| Field | Value |
|---|---|
| Stage | M9 / scene gate and rollback |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M8 guarded fill |

## 1. Goal

M9 decides whether a MeshPrior proposal may be accepted in a scene. Object-level shape confidence is never sufficient by itself.

The first implementation is a dry-run gate that works without full differentiable scene optimization.

## 2. Gate Inputs

Required:

- mesh before proposal;
- mesh after proposal;
- proposal id and proposal type.

Optional:

- object-level confidence and uncertainty;
- free-space violation function or precomputed free-space scores;
- rendering metric deltas;
- COLMAP sparse depth deltas;
- sparse normal proxy deltas;
- FPS or runtime proxy.

## 3. Gate Metrics

Dry-run metrics:

- triangle count delta;
- boundary edge count delta;
- hole-boundary score delta;
- connected component count delta;
- floater/component count delta;
- free-space violation delta;
- local geometry displacement proxy.

Future full-scene metrics:

- rendering metric delta;
- COLMAP sparse depth AbsRel / MAE delta;
- sparse normal proxy delta;
- controlled FPS proxy.

## 4. Acceptance Thresholds

Hard rejects:

- free-space violation increases above tolerance;
- connected component count increases;
- triangle count grows above the configured ratio;
- no scene-side metric improves;
- object confidence is high but scene evidence is neutral or negative.

Default dry-run accept:

- no hard reject;
- at least one scene-side metric improves, such as boundary edge count decreasing;
- object uncertainty is below threshold when provided.

## 5. Rollback Data Structure

Rollback snapshot is an NPZ containing:

```text
vertices
faces
metadata_json
```

The snapshot is written before proposal application and can be restored into a mesh tuple for downstream tools.

## 6. Dry-Run Mode

Dry-run mode runs entirely on mesh arrays and optional proposal NPZs. It writes:

```text
gate_report.json
gate_report.md
rollback_snapshot.npz
```

This lets M9 validate gate behavior before scene rendering or training is available.

## 7. Object vs Scene Evidence

Object-level evidence:

- posterior confidence;
- shape-field support;
- proposal type;
- uncertainty.

Scene-level evidence:

- topology changes;
- free-space change;
- rendering/depth/normal deltas when available;
- local geometry proxy.

Acceptance requires scene-level evidence. Object evidence can tighten or explain the decision but cannot accept a proposal alone.
