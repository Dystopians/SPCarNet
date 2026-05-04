# MeshSplatOpt Bibliography Plan

## Required Citation Buckets

1. 3D Gaussian Splatting and follow-up splatting renderers.
2. Mesh Splatting or mesh-based splat primitives used by this repository.
3. Classical mesh simplification, especially QEM/decimation.
4. Classical mesh repair and hole filling.
5. Neural rendering with depth or COLMAP regularization.
6. Multi-view consistency, free-space constraints, and geometry priors.
7. Safe/counterfactual model editing or validation-by-intervention.

## Bib Risk

The paper cannot rely on novelty from "mesh repair" alone because the current validated real-scene result is compact-recovery. The bibliography should support a claim that MeshSplatOpt combines simplification, counterfactual certification, and recovery in a rendering-aware optimizer.

## Must-Add Baseline Citations

- QEM mesh simplification.
- Screened Poisson or common hole-fill references if giant-hole repair is discussed.
- 3DGS and mesh-splatting base method.
- Depth-regularized neural rendering / sparse COLMAP supervision.

