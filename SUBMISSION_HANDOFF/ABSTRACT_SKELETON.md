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

---

## STAGE-4 RE-SLOT (2026-07-10) — evidence-cached rendering leads

1. **Problem slot.** Compact mesh-splat scenes render below the radiance-field
   state of the art; prior attempts to close the gap by baking or distilling
   training-view evidence into the representation fail (quote the Stage-1/2/3
   falsification corpus: baking 4.76%, distillation 5–16%, residual cosine
   ≈0.21).

2. **Method slot.** We CACHE the evidence instead of baking it: per-scene
   evidence-cached rendering (ECR) ships {mesh checkpoint + train-view cache
   + frozen transport} and corrects each novel view at render time via
   depth-consistent K-source warping, learned per-pixel fusion, and
   residual-vs-RGB routing — trained train-only, audited per row by a
   structural no-test-GT audit (69/69 GREEN).

3. **Quality result slot.** Full9: +1.666 dB [+1.567,+1.766] over the primary
   anchor; +0.361 dB [+0.316,+0.407] AND −0.0169 LPIPS over the strong
   Phase-J floor (CIs excl. 0; 8/9–9/9 per-scene wins); ladder rungs each
   CI-gated (quote `final_stack_tables.md` T-ECR-1/3).

4. **Compression result slot.** At 50% triangles the full stack exceeds the
   FULL-budget anchor by +1.488 dB / −0.0748 LPIPS (9/9) — compaction and
   quality are no longer a trade (quote L6 tie-back).

5. **Honest-cost slot.** The gain is bought with storage and transport time:
   TOTAL artifact 1.7–3.5 GB/scene, ~1.3–3 fps end-to-end; at matched TOTAL
   storage 3DGS remains +0.3..+1.5 dB ahead at 40–70× fps (quote
   e07_matched_total_3dgs.md + the L5 Pareto); a generative single-step
   enhancer given the same evidence is PSNR-negative on all 3 tested scenes
   (quote difix_table.md).

6. **Safety/analysis slot.** The routing is gated structurally: evidence RGB
   cannot appear where the transport lacks support — 1 transport-negative
   view in 139 (−0.06 dB), coverage gaps degrade to the base render; the
   net's explicit confidence inputs are provably redundant (E-08 ablation,
   CIs incl. 0).

7. **Boundary slot.** Per-scene method (no cross-scene generalization);
   train-view evidence is a declared render-time input of the shipped
   artifact; geometry/downstream consumption unchanged by construction (the
   R3 impossibility is cited as scope).

8. **Takeaway slot.** For mesh-first pipelines, caching evidence beats baking
   it: an audited train-only transport turns a compact mesh artifact into a rendering
   system that surpasses its strongest baked baseline — at an honestly
   reported storage/latency price.

9. **Editing slot (added 2026-07-12, Route A).** The cache stays consistent
   under mesh editing: per-pixel evidence invalidation (one face-ID pass +
   masks at the transport's warp) renders deletions and recolors without
   stale-content leakage — where BOTH naive strategies fail on one class
   each (cache rebuild ghosts deletions back, +3.09 dB; stale caches
   repaint recolors, +1.96) — preserving unaffected quality to ≤0.02 dB at
   ~1/20 rebuild cost (quote `analysis/edit_aware/*`; motion/deformation
   are declared boundaries).
