# MeshPrior Stage 15 Design — Retrieval-Deformation Fallback

| Field | Value |
|---|---|
| Stage | M15 / retrieval-deformation fallback |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M14 paper roadmap and claim-risk analysis |

## 1. Motivation

M14 concludes `MORE_SCENE_EVIDENCE_REQUIRED`. One plausible risk is that the learned implicit shape field is too blurry for reliable scene proposals. A retrieval-deformation fallback tests a different prior: retrieve a train-only complete car anchor and optionally deform it conservatively toward observed scene evidence.

This is not a headline replacement yet. It is a serious fallback that must be measured against the Stage 3 posterior-style MeshPrior proposal metrics.

## 2. Leakage Policy

The anchor bank is train-only:

- anchors may only come from records with `split=train`;
- validation/test object IDs must never be inserted into the bank;
- retrieval excludes an anchor with the same `object_id` as the query if such an ID is provided;
- the loader refuses banks containing non-train splits.

## 3. Anchor Bank

Each anchor stores a fixed-size canonical point sample:

```text
object_id
split=train
points[N, 3]
metadata
```

The first implementation uses clean train object points when an object index is available. The smoke path builds a synthetic box anchor to keep the contract testable without large data.

## 4. Retrieval-Only Baseline

Retrieval-only must be measured before deformation. It selects the nearest anchor by symmetric Chamfer L1 between normalized observed points and anchor points.

The retrieval result records:

- best anchor ID;
- best score;
- second score;
- margin;
- uncertainty;
- mean nearest distance.

## 5. Optional Smooth Deformation

The optional deformation is conservative:

- each anchor point moves a clipped fraction toward the nearest observed point;
- max displacement is bounded;
- the deformation row is killed if it reduces valid-surface preservation or increases free-space violation.

This stage does not implement a neural deformation model. A neural deformation model is only justified if retrieval-only is clearly promising.

## 6. Proposal Contract

Retrieval exports MeshPrior-compatible proposal types:

- `protect`;
- `prune`;
- `snap`;
- `fill_candidate`;
- `uncertainty`.

Scene gates still decide acceptance. Retrieval cannot directly insert geometry into a scene.

## 7. Benchmark

The M15 benchmark compares:

- `stage3_posterior_proxy`;
- `retrieval_only`;
- `retrieval_deform`.

Primary metrics:

- floater prune precision/recall;
- valid-surface protect recall;
- hole boundary score;
- free-space violation;
- snap/deform displacement diagnostics.

## 8. Stage Gate

M15 passes if:

- a train-only anchor bank can be built;
- retrieval-only is measured before deformation;
- smoke test exports all required proposal types;
- synthetic metrics are written;
- the report states whether to pivot, keep as baseline, or kill deformation.
