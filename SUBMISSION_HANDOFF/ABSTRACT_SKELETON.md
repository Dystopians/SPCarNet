# GEMS Submission Handoff — Abstract Skeleton

Generated 2026-07-04 for Stage3 closure. This is a numbered slot template, not
final prose.

1. **Problem slot.** Mesh-splat representations can render well after
   compression, but photometric metrics alone do not reveal geometry or
   downstream-consumption reliability.

2. **Method slot.** We introduce GEMS, a train-evidence-guided mesh-splat
   compaction and evaluation framework: evidence pruning, topology-frozen
   features-only recovery, and a single-mouth measurement suite.

3. **Rendering/efficiency result slot.** At B50, GEMS-core preserves or improves
   rendering quality in the reported regimes while reducing triangle count and
   resource use; quote exact scene counts and deltas from `T1_main_pareto.md`
   and `T4_efficiency.md`.

4. **Analysis result slot.** Train-evidence signals predict but do not certify
   test-time or planning safety; quote evidence-vs-error correlations from
   `analysis/e2geo_evidence_vs_error/summary.json`.

5. **Downstream result slot.** Four one-time train-evidence occupancy consumers
   fail the frozen parking-grade bar; quote the R3-FINAL 0/100 found result and
   route-family summary from `RESULTS/CONSUMPTION_IMPOSSIBILITY.md`.

6. **Boundary slot.** We do not claim state-of-the-art novel-view rendering,
   geometry improvement over clean MeshSplatting, or successful closed-loop
   planning; the contribution is compactness plus a reproducible reliability
   audit.

7. **Takeaway slot.** Compact mesh-splat rendering can be made efficient and
   evidence-locked, but collision-grade world models require representation
   improvements beyond the tested MeshSplatting checkpoints.

## Language Rules

- Use "preserves", "bounds", "exposes", and "falsifies".
- Avoid "solves", "safe", "guarantees", "state-of-the-art", and "robust"
  unless directly supported by the table being quoted.
- Every numeric clause must point to T1-T7, F1-F8, or the R3-FINAL addendum.
