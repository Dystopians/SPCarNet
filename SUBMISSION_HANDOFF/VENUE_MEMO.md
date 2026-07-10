# GEMS Submission Handoff — Venue Memo

Generated 2026-07-04 for Stage3 closure. Narrative prose remains human work;
this memo only locks claim-safe positioning and evidence pointers.

## Outcome Assumption

R3-FINAL outcome: **IMPOSSIBILITY x4 route families**. The visibility-carved
three-state consumer failed as a hard non-near-miss, so the paper should not be
positioned as a positive closed-loop planning system. The honest positioning is
an evidence-locked compact representation and measurement-suite paper:

- GEMS-core gives half-budget rendering/efficiency gains or iso-quality regimes.
- The suite exposes geometry and downstream-consumption failure modes that
  photometric metrics hide.
- Compaction preserves downstream proxy outcomes; it does not repair baseline
  checkpoint geometry.
- The R3 consumption axis is a citable negative, not a missing experiment.

## Deadline Table

All dates below are planning placeholders and must be human-verified.

| venue | approximate deadline | fit under R3-FINAL FAIL | note |
|---|---:|---|---|
| 3DV 2027 | Aug 25-28, 2026 [HUMAN-VERIFY] | strong | Best match for representation, geometry diagnostics, and reproducible evidence pack. |
| ICRA 2027 | Sep 15, 2026 [HUMAN-VERIFY] | medium | Viable only if framed as a planning-consumption analysis/negative with careful robotics caveats. |
| WACV 2027 R2 | Aug-Sep 2026 [HUMAN-VERIFY] | strong | Good fit for practical system, efficiency, and honest limitations. |
| CVPR 2027 | mid-Nov 2026 [HUMAN-VERIFY] | high variance | Needs sharper novelty framing and stronger visual figures; avoid overclaiming downstream. |
| NeurIPS 2027 D&B | May 2027 [HUMAN-VERIFY] | medium-strong | Fits measurement-suite/data-and-benchmark framing if artifacts are packaged cleanly. |

## Recommendation

Top two primary targets under the current evidence:

1. **3DV 2027**: lead with compact mesh-splat representation plus geometry
   reliability audit. This venue tolerates systems-plus-analysis if the
   evaluation is dense and reproducible.
2. **WACV 2027 R2**: lead with practical deployment tradeoffs: 50% triangle
   budget, quality/efficiency, and failure taxonomy.

ICRA is a secondary target if the mentor wants a robotics audience. CVPR is
possible but high-risk unless the story is made more principled and the figures
make the evidence-vs-error insight visually obvious.

## Claim to Section Map

### 3DV / WACV 8-page skeleton

| section | budget | claim-safe content |
|---|---:|---|
| Introduction | 0.8 page | Photometric mesh-splat compactness is not enough; geometry/downstream reliability must be measured. |
| Related Work | 1.0 page | Mesh/point/splat compression, neural rendering reliability, occupancy consumers; no SOTA NVS claim. |
| Method | 1.6 pages | Evidence prune, features-only recovery, metrics suite, static consumers as analysis. |
| Experiments | 2.4 pages | T1/T2 rendering, T3 geometry, T4 efficiency, T5 downstream, T6 ablations, T7 robustness. |
| Failure/Negative Results | 1.0 page | E2/E3/R3 failure mechanisms; R3-FINAL impossibility as first-class result. |
| Limitations | 0.6 page | Per-scene optimization, no 3DGS SOTA claim, no closed-loop positive, no high-speed driving. |
| Conclusion | 0.2 page | Compactness can be preserved, but geometry-grade world models remain unresolved. |

## Non-Claims to Preserve

- No state-of-the-art novel-view-quality claim versus 3DGS-style methods.
- No claim that GEMS improves geometry versus clean MeshSplatting.
- No claim of parking-grade closed-loop planning success.
- No claim that moderate-budget importance pruning always dominates random
  after safe fine-tuning on small scenes.
- The baseline end-phase decline is a property of the baseline trainer, not a
  GEMS contribution.

---

## STAGE-4 REFRESH (2026-07-10) — the positive branch is now REAL

The Stage-3 memo above assumed the paper's spine was "compact representation +
measurement suite + citable negatives." Stage-4 (ECR) changes the outcome
assumption: there is now a CI-backed POSITIVE system result, and the paper
should lead with it.

### New outcome summary (all through the single mouth, PROTOCOL 1.2.0)

- **Headline system result:** per-scene evidence-cached rendering (ECR): the
  shipped artifact {mesh checkpoint + train-view evidence cache + certified
  transport} renders full9 at **+1.666 dB [+1.567,+1.766]** over the PRIMARY
  anchor and **+0.361 dB [+0.316,+0.407] / ΔLPIPS −0.0169** over the strong
  Phase-J floor (9/9 and 8/9 per-scene CI-wins) — and the ladder generalizes
  to 3/5 held-out suite scenes outright (town06/toy honest boundary cases).
- **Compression headline (the L6 row):** at HALF the triangle budget the full
  stack still beats the FULL-BUDGET anchor by **+1.488 dB / −0.0748 LPIPS**
  (9/9) — "compact AND better" replaces "compact at iso-quality".
- **Method story (ladder, each rung CI-gated):** frozen Phase-J transport
  (v0 floor) → joint (K,α) multiband aggregation (v1) → learned per-pixel
  fusion (v2) → residual-vs-RGB routing (v3). Two pre-registered negatives
  are part of the story: distilled-base composition is SUBSUMED by the
  transport (L1), and the net's explicit confidence input channels are
  REDUNDANT (E-08) — safety lives in the STRUCTURAL compose gate (1
  transport-negative view in 139; coverage gaps degrade to base).
- **Honest cost:** TOTAL artifact 1.7–3.5 GB/scene and ~1.3–3 fps end-to-end
  (vs 3DGS ~40–70× faster and ahead +0.32..+1.53 dB at matched TOTAL
  storage, reported plainly); cache-quality Pareto = L5 (in flight).
- **External cells measured:** Difix3D+ single-step on the same base renders
  is PSNR-negative on all 3 scenes and dominated by ECR on both metrics
  (A11 is now answered with numbers, not positioning).
- The R3 impossibility, geometry suite, and Stage-1/2/3 falsification corpus
  remain first-class SCOPE/motivation (the corpus is exactly WHY evidence is
  cached rather than baked).

### Revised venue fits

| venue | fit under Stage-4 | note |
|---|---|---|
| 3DV 2027 | **strong, now as a positive-system paper** | representation + certified-transport method + reliability audit; leads with CR1/CR3. |
| WACV 2027 R2 | strong | deployment trade story (storage-vs-quality Pareto, honest fps) is stronger with L6. |
| CVPR 2027 | **materially improved** | there is now a method core (routing ladder) + striking quals (β/confidence maps); risk shifts to novelty positioning vs ULR/Deep-Blending lineage — REBUTTAL A10 is the key paragraph. |
| ICRA 2027 | unchanged (medium) | still not a closed-loop story; DS-1 retry may sharpen the scope sentence either way. |
| NeurIPS 2027 D&B | medium-strong | unchanged; the frozen protocol + 69-audit trail is the D&B asset. |

### Revised section map (delta from the 8-page skeleton above)

- Method (1.6 → 2.0 pages): the transport + ladder is now the core; move the
  measurement-suite mechanics to a compact subsection + appendix.
- Experiments: T-ECR-1/2/3 + Pareto + matched-storage 3DGS + Difix cell +
  E9-style failure cases with β/conf visualizations (FIGURE_NOTES).
- Failure/Negative Results (keep 1.0 page): L1 subsumption, E-08 conf-input
  null, town06/toy boundary cases, R3-FINAL as scope.
- Non-claims: unchanged EXCEPT "no positive system claim" is DELETED; add
  "per-scene method, no cross-scene generalization claim; train-view
  evidence is a declared render-time input."
