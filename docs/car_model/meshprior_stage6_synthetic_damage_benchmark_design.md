# MeshPrior Stage 6 Design — Synthetic Mesh-Damage Benchmark

| Field | Value |
|---|---|
| Stage | M6 / synthetic damage benchmark |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M5 optimizer adapter |

## 1. Goal

Before touching real parking-lot scenes, M6 builds a controlled synthetic benchmark for proposal behavior under known mesh damage.

This benchmark is not a replacement for scene metrics. It is a safety filter for protect/prune/snap/fill development.

## 2. Damage Types

Initial M6 supports:

- `local_hole`: remove faces near a point or selected face ids.
- `floater`: add disconnected triangles away from the object.
- `vertex_noise`: perturb vertices.
- `density_imbalance`: duplicate/subdivide-like local face density proxy.

Later stages may add side-panel, roof/cabin, wheel, and oversimplified-patch damage once part annotations or heuristics exist.

## 3. Metrics

M6 reports:

- hole boundary edge count,
- hole boundary score,
- floater prune precision,
- floater prune recall,
- valid surface protect recall,
- triangle count delta,
- visible preservation proxy,
- free-space violation placeholder,
- mesh extraction success placeholder.

Object Chamfer is not the headline for M6 because this stage evaluates proposal signals.

## 4. Baselines

M6 baseline rows:

- damaged input,
- protect/prune proposals with analytic support field,
- optional Stage-3 posterior proposal row when a checkpoint and object data are explicitly supplied.

v0.7/v0.8.2 rows are deferred until their evaluation wrappers are wired into the common report matrix.

## 5. Oracle Separation

Known synthetic damage labels are used only for evaluation:

- floater face labels,
- valid surface face labels,
- removed-hole face labels.

They are not passed into proposal scoring.

## 6. Success Threshold

M6 passes if the synthetic benchmark:

- generates damage,
- computes finite metrics,
- protect/prune scores identify floater triangles,
- protect/prune scores preserve valid surface triangles,
- writes machine-readable and markdown reports.

## 7. Stage Gate

M6 passes only after:

- smoke test passes,
- report generator produces a markdown report,
- inference-time metrics and oracle label metrics are separated in the JSON.
