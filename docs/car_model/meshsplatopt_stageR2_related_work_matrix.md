# MeshSplatOpt Stage R2 Related Work And Novelty-Threat Matrix

Date: 2026-05-02

## Gate

`PASS`.

The matrix names concrete threats and required baselines for pruning, simplification, hole filling, and geometry-aware repair.

## Required Conclusion

Training-time pruning alone is not novel. Mesh/triangle pruning alone is not novel. Geometry priors alone are not novel. Counterfactual validation alone is not enough unless it is tied to bidirectional repair and evidence-debt minimization.

The strongest novelty target is:

> unified CSEF + reversible bidirectional edit calculus + certified huge-hole repair.

## Neural Rendering And Splatting Foundations

| method/group | what it does | threatens | what it does not cover | required baseline/ablation | novelty threat |
|---|---|---|---|---|---|
| NeRF | optimizes an implicit radiance field from posed images | differentiable scene optimization framing | explicit mesh topology surgery, rollback, hole repair | cite as foundation; not a direct baseline | Low |
| Instant-NGP | accelerates neural field training with hash grids | efficient scene optimization | mesh-splat edit calculus and topology accounting | cite as speed foundation | Low |
| 3DGS | optimizes explicit Gaussians with densification/pruning | explicit primitive optimization and pruning | mesh/triangle surgery, boundary-loop repair, giant-hole certification | 3DGS-style pruning/compression baselines where feasible | Medium |
| Mesh Splatting | renders/optimizes mesh or triangle splats | direct base method | certified bidirectional repair | clean Mesh Splatting baseline | High |
| Triangle Splatting | triangle primitive rendering | triangle-level edit target | CSEF evidence field and counterfactual rollback | triangle-splat clean baseline | High |
| 2D Triangle Splatting | 2D triangle/surface splat formulation | surface-aligned primitive representation | giant void fill and reversible repair portfolio | include if implementation is compatible | Medium |

## Mesh-Aware Splatting And Geometry

| method/group | what it does | threatens | what it does not cover | required baseline/ablation | novelty threat |
|---|---|---|---|---|---|
| SuGaR | aligns Gaussians to surfaces and extracts meshes | surface-aware radiance geometry | edit certification, rollback, hole-specific evidence debt | surface-aware Gaussian/mesh extraction comparison | High |
| MeshGS | binds Gaussian/radiance primitives to mesh structures | mesh-aligned splatting | reversible multi-operation repair | mesh-aware splatting baseline if runnable | High |
| 2DGS | uses 2D Gaussians and geometry regularizers | geometry regularization and surfaces | edit-level audit trails and giant-hole policy | no-CSEF geometry-regularized baseline | Medium |
| DN-Splatter | adds depth/normal supervision for geometry | depth/normal priors | topology surgery and free-space-certified fill | depth/normal regularization ablation | Medium |
| mesh-embedded Gaussian methods | couple splats to mesh surfaces | mesh/radiance coupling | evidence-debt edit selection and rollback | strongest compatible mesh-Gaussian baseline | High |

## 3DGS Compression And Pruning

| method/group | what it does | threatens | what it does not cover | required baseline/ablation | novelty threat |
|---|---|---|---|---|---|
| LightGaussian | compresses/prunes Gaussian scenes | compactness claim | constructive mesh repair | compression Pareto baseline | Medium |
| Compact3DGS | compact representation/pruning | topology/memory efficiency framing | holes, dents, boundary loops | compression baseline | Medium |
| EfficientGS | efficient rendering/training | efficiency metrics | certified geometry edits | runtime/memory comparison | Medium |
| Mini-Splatting | reduces Gaussian count | pruning novelty | mesh fill/snap/split | same-budget pruning baseline | Medium |
| EAGLES | efficient Gaussian representation | compact explicit primitive story | mesh edit certificates | compression baseline if available | Medium |
| RadSplat | radiance-splat efficiency | splat compactness | surface evidence debt | cite/compare where compatible | Medium |
| LP-3DGS | lightweight/pruned 3DGS | pruning and compression | bidirectional repair | budgeted pruning baseline | Medium |
| MaskGaussian | mask-guided Gaussian pruning | learned/removable masks | mesh topology operations | mask/prune ablation | Medium |
| PUP 3D-GS | pruning/updating primitives | adaptive pruning | giant-hole fill | pruning baseline | Medium |
| GaussianPOP | optimized/pruned Gaussian pipeline | primitive reduction | scene evidence debt | compression baseline | Medium |
| GaussianSpa | sparse Gaussian optimization | sparsity | mesh repair | sparse baseline | Medium |
| SafeguardGS | safer Gaussian pruning | safety-gated pruning | bidirectional mesh surgery | safety-pruning baseline | High |

## Classical Mesh Processing

| method/group | what it does | threatens | what it does not cover | required baseline/ablation | novelty threat |
|---|---|---|---|---|---|
| QEM edge collapse | simplifies meshes by quadric error | collapse/merge operation | render/geometry counterfactual gates | QEM/post-hoc simplification baseline at 90/75/50/25% | High |
| constrained Delaunay triangulation | fills/triangulates planar domains | hole fill geometry | scene evidence certification | hole fill without render/free-space gate | High |
| classical hole filling | closes boundary loops | small and large hole claims | multi-view evidence, prior-only labeling | classical hole-fill baseline | High |
| Poisson/screened Poisson | reconstructs surfaces from points/normals | geometry repair/completion | mesh-splat training loop and rollback | Poisson reconstruction baseline if points/normals exist | Medium |
| isotropic/adaptive remeshing | redistributes mesh elements | split/collapse topology optimization | render/scene validation | remesh-only ablation | Medium |
| Laplacian/ARAP deformation | smooths/deforms surfaces | snap/deform operation | free-space and render certificate | smooth/deform-only baseline | Medium |

## Multi-View Geometry And Priors

| method/group | what it does | threatens | what it does not cover | required baseline/ablation | novelty threat |
|---|---|---|---|---|---|
| COLMAP SfM/MVS | sparse/dense multi-view geometry | positive surface and sparse depth evidence | differentiable edit portfolio | sparse-depth baseline and sparse-geometry gate ablation | High |
| plane/Manhattan/ground priors | regularize man-made scenes | ground/wall repair and void filling | hallucination-safe counterfactual validation | plane fill without free-space gate | High |
| object shape priors | propose canonical object geometry | vehicle repair and fill | scene evidence disposal | object-prior fill without scene gate | High |
| monocular depth/normal priors | external learned geometry signals | depth/normal certificate | pure COLMAP evidence claim | optional add-on ablation, labeled separately | Medium |

## Novelty Threat Summary

The main threats are strong:

- Mesh Splatting is the base method and must be the clean baseline.
- Stage35 retained PRISM must be treated as the strongest internal delete-centric baseline.
- QEM and classical hole filling directly threaten collapse and fill operations.
- SuGaR/MeshGS/2DGS/DN-Splatter threaten the surface-aware geometry story.
- COLMAP/depth/plane/object priors threaten the evidence source story.

MeshSplatOpt remains novel only if experiments show that a unified CSEF-driven, reversible, certified edit portfolio solves defects that single-operation baselines do not: especially snap/fill repairs, giant-hole handling, and prior hallucination rejection.

## Paper-Safe Positioning

Do not claim:

- pruning is new;
- mesh simplification is new;
- plane fitting or object priors are new;
- counterfactual validation alone is the contribution.

Claim only if supported:

- CSEF provides a common evidence contract for destructive and constructive mesh-splat edits;
- reversible edit calculus enables rollback and audit across delete, collapse, snap, split, fill, and appearance recovery;
- giant voids are filled only when certified, and unknown voids are rejected or labeled prior-only diagnostic;
- the full method outperforms delete-only and classical repair baselines on repair metrics or topology-quality Pareto curves.
