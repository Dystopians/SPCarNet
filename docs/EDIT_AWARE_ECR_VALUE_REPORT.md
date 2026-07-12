# EDIT-AWARE ECR (Route A) — VALUE REPORT — FINAL

Finalized 2026-07-12 (temporal-after-edit lands as the last appendix number; all other evidence banked).
Companions: FEASIBILITY_AUDIT (data model), PROTOCOL (frozen eval), EXECUTION_PLAN.
Every number below from `analysis/edit_aware/*/edit_eval.{md,json}` — nothing hand-typed.

## VERDICT: **CONDITIONAL-GO** — validated as the ECR paper's "why retain the mesh" section; not yet a standalone central contribution

- **Overall technical feasibility: 85/100** (deletion+recolor implemented end-to-end: ~450 lines across
  4 tools + a 20-line default-off hook at the single warp site, bit-stable when off).
- **Overall scientific value: 78/100** (upgraded from the provisional 70: the prototype produced TWO
  non-obvious, CI-backed findings — the rebuild inversion and the deletion/recolor protection asymmetry —
  that elevate this above "engineering plumbing").

## The evidence matrix (the story in one table)

| Edit class | C2 stale cache | C4 full rebuild (photos unmasked) | C5 local invalidation (ours) |
|---|---|---|---|
| **Deletion** (garden table 2.04M faces; toy car_0 711k) | mild leak (garden 0.0088) — *accidentally* protected: stale depths fail the z-test | **CATASTROPHIC**: ghost +3.088 [+2.568,+3.616]; leak 0.0840 = 13× ours; visible translucent table apparition | clean: leak 0.0065/0.0032; preservation −0.020 [−0.034,−0.008] dB (garden) / −0.002 (toy) on TRUE GT outside the region |
| **Recolor** (same table, DC shift) | **BAD**: ghost +1.964 [+1.869,+2.061]; leak 0.0346 = 10× ours — old color repaints (depth-CONSISTENT stale evidence sails through the z-test) | ≡ C2 by construction (photos stale + depths valid either way) — not separately run, documented | clean: leak 0.0034; preservation +0.015 [+0.005,+0.023] |

**Finding 1 (rebuild inversion):** the "obvious" fix — rebuild the cache from the edited checkpoint — is
the WORST action: refreshing depths makes the stale photographs depth-consistent, so the transport
faithfully paints the deleted object back. Staleness lives in the photographs, not the cache files.
**Finding 2 (protection asymmetry):** the depth z-test accidentally shields DELETION (geometry-inconsistent
staleness) but gives ZERO protection for appearance edits — so no naive strategy survives both classes,
while per-pixel evidence invalidation survives both with unaffected-region quality preserved to ≤0.02 dB
and ECR's photometric gain intact (+1.65 dB over the edited base outside the region).

## Per-class final scores and verdicts

| Class | Feasibility | Value | Verdict |
|---|---|---|---|
| 1. Deletion | **85** | **75** | **VALIDATED** (2 scenes, frozen protocol, CIs, panels) |
| 2. Recolor | **75** | **60** | **VALIDATED** as the decisive control (same mechanism; its value is proving Finding 2) |
| 3. Rigid translation | 55 | 65 | NOT ATTEMPTED — deferred with a written path (transform-vs-mask ablation; dual occlusion; baked-shadow caveat) |
| 4. Deformation/topology | 35 | 60 | **DECLARED BOUNDARY** — unsupportable from original photographs without hallucination beyond small deformations |

## Cost / locality (honest)

Garden: 108 s / 1053 MB rewritten vs ~40 min full routed rebuild (~20× wall-clock); toy: 34 s / 231 MB
(~30×). Caveat reported plainly: both edits target centrally-visible objects, so ALL train views were
affected (161/161, 72/72) — view-locality is weak for central objects; savings come from bytes-per-view
(GT photographs untouched, no α recalibration, no net retrain) and wall-clock. A peripheral-object edit
would show stronger view-locality; not claimed without measurement.

## Missing technical capabilities (exact, from the audit)

Stored per-pixel 3D-exact barycentrics (needed only for class 4); an oriented-region edit UI beyond
box/cylinder; per-object segmentation (edits currently geometric-region-based); translation's evidence
transform machinery. None block the validated classes.

## Minimum publishable Route-A method (as validated)

"Per-source-pixel evidence-validity masks, derived from one face-ID pass of the original checkpoint over
the training views and honored multiplicatively at the transport's single warp site, make an evidence
cache edit-consistent: deletion and appearance edits render without stale-content leakage (10–13× lower
in-region deviation than rebuild/stale baselines, CIs excl. 0), preserve unaffected-region quality to
≤0.02 dB (true-GT), and cost ~1/20–1/30 of a cache rebuild — while the two naive strategies each fail on
one edit class."

## Strongest likely reviewer objection + response

*"Gaussian-editing systems (GaussianEditor-class) delete/recolor without any of this machinery — why is
this hard?"* Response: they have no evidence cache — nothing external to the representation can leak.
The stale-evidence problem is CREATED by evidence-cached rendering (which buys +1.67 dB, Stage-4), and
this work shows the cache can be made edit-consistent rather than forcing a choice between photometric
quality and editability. The face-identity of the mesh is what makes invalidation exact and cheap
(face-ID buffer = one render pass; identity = row index) — a capability with no direct 3DGS-sidecar
equivalent (Gaussians lack stable per-primitive image evidence attribution under densification). A fair
3DGS-editing context row is USEFUL for the paper's related-work honesty but is not a claim target; not
run this cycle (no evidence cache exists there to compare invalidation against).

## Does Route A answer "why retain the mesh?"

**Partially — and honestly: it answers "why retain face identity," which the mesh provides for free.**
The measured chain is: explicit faces → exact pixel-to-face attribution (rend_ids) → exact evidence
invalidation → edit-consistent evidence caching. That is a real, now-measured workflow advantage a
static rendering sidecar does not have. It does NOT by itself justify the mesh against 3DGS on
quality/speed (the frozen NON-CLAIMS trade stands); combined with the Stage-2/3 geometry/downstream
suite and L6 compaction, it completes the mesh-artifact story: measure it, compact it, transport
evidence over it, and now EDIT it without losing the evidence advantage.

## Recommended next action

**Integrate Route A as the editing section of the ECR paper** (one method subsection + the evidence
matrix + one figure row per edit class; ~0.75 page + supp): it strengthens the central thesis ("caching
evidence beats baking it — and the cache stays consistent under editing") without displacing it. Do NOT
build the standalone editing paper this cycle: two validated classes with one mechanism is a section,
not a spine; revisit standalone (with translation + peripheral-object locality + a 3DGS-editing context
row) only if the main paper lands and the mentor wants a follow-up. Route B (appearance baking) remains
falsified by the Stage-1/2/3 corpus; Route C (geometry refinement) remains open but is a different
program — neither displaces Route A's section-level value.

---

## PRE-SUBMISSION ADDENDUM (2026-07-12, GOAL #E-16 red-team + cleanup)

Numbers above predate the red-team corrections. The CANONICAL presentation is
`analysis/edit_aware/routeA_master_table.md` (packed at
`RESULTS/STAGE4_ECR/edit_aware/routeA_master_table.md`): in-region claims are ORACLE-PRIMARY
(synthetic scene only — verified oracle rebuild; real scenes report true-GT preservation + bounded
ghost metrics only); leak_R is SECONDARY (ρ=0.502 vs oracle error; penalizes legitimate improvement —
the garden proxy ordering INVERTS the oracle ordering); the 5-way novelty family holds at BOTH 95% and
99% CIs; the parent-mask builder bug affected NO banked result (all single-edit parents unedited; the
only chained cache was built post-fix with verified mask-superset inheritance).
