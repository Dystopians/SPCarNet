# MeshPrior Stage 13 Design — Evaluation Protocol

| Field | Value |
|---|---|
| Stage | M13 / evaluation protocol and matrix |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M12 prior calibration plus cleanup repair |

## 1. Datasets and Splits

Object prior:

- MeshFleet car object cache, using existing SP-CarNet object index and validation split.
- Existing available metrics come from SP-CarNet posterior encoder validation output.

Synthetic mesh repair:

- controlled box damage benchmark from M6-M8;
- damage types: `local_hole`, `floater`, `vertex_noise`, `density_imbalance`;
- calibration case: `vertex_noise` snap-risk benchmark.

Scene-level:

- dry-run synthetic local-hole scene from M10/M11;
- short real COLMAP video smoke run from the cleanup repair validation, used only as diagnostic scene training evidence.

Real parking-scene claims are not made until a selected parking COLMAP scene and trained baseline are evaluated.

## 2. Object-Level Metrics

Primary:

- reconstruction Chamfer L1;
- hidden-surface Chamfer L1;
- visible-preservation error;
- zero-corruption Chamfer;
- free-space violation rate;
- mesh extraction success rate.

Diagnostic:

- mesh IoU;
- latent retrieval error;
- runtime / inference time where available.

## 3. Synthetic Damage Metrics

Primary:

- hole closure / boundary-edge reduction;
- floater prune precision and recall;
- valid surface protect recall;
- free-space violation;
- triangle count delta.

Diagnostic:

- moved vertex fraction;
- snap surface-distance delta;
- component/floater count delta.

## 4. Scene-Level Metrics

Primary:

- COLMAP sparse AbsRel;
- sparse DepthMAE;
- sparse normal mean angle;
- PSNR / SSIM / LPIPS / MAE when render eval exists;
- accepted/rejected proposal counts;
- triangle count.

Diagnostic:

- controlled FPS;
- render time;
- car ROI hole/floater metrics.

## 5. Primary vs Diagnostic

Primary metrics determine claims. Diagnostic metrics explain failure modes or runtime tradeoffs.

Object Chamfer alone is never sufficient to claim scene optimization success.

## 6. Inference-Time vs Oracle-Only

Inference-time:

- posterior encoder output;
- protect/prune/snap/fill proposals;
- scene gates using scene evidence available at deployment;
- calibrated snap profile.

Oracle-only:

- best-of-K selection using clean GT;
- synthetic labels used to compute recall/precision;
- damage labels;
- failure classification from GT clean meshes.

Oracle metrics may be reported but cannot be headline comparisons.

## 7. Checkpoint Selection Rules

- Use last checkpoint only if it is the predetermined evaluation checkpoint.
- Do not choose checkpoint by test metric.
- For smoke runs, label iteration and dataset as diagnostic.
- For scene training, include cleanup policy in the row.

## 8. Seed Protocol

Default seeds:

```text
0,1,2
```

Smoke mode may use seed `0` only. Reports must state when fewer seeds are available.

## 9. Failure-Case Reporting

Every report writes:

```text
outputs/carnet/meshprior/reports/failure_cases.md
```

Failures include:

- missing metric files;
- failed mesh extraction;
- rejected proposals;
- free-space violation increase;
- topology/floater regression;
- real-scene metric unavailable.

## 10. Scientific Safeguards

- Missing experiments are marked `MISSING`.
- Oracle rows are labeled `oracle_only=true`.
- Dry-run rows are labeled `dry_run=true`.
- Synthetic and scene metrics are not mixed into one headline table.
- Rejected proposals count as safety evidence, not automatic failure.
