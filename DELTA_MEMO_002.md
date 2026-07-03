# CLAIMS-vs-EVIDENCE Delta Memo #002

2026-07-03 · covers GOALs #010–#017, #R-05..#R-09 · per Stage-1R R4 cadence

## Strengthened
- **C1′ is now a complete, three-regime, baseline-anchored story:** B50 iso across three suites; B25/B12.5 graceful degradation with FT value growing (9/9 CI at both budgets); **B12.5 evidence-vs-random dominance +5.23 dB mean (9/9 CI)**; **B3 QEM column: B5 dominates geometric decimation by 2.7–3.4 dB** (mechanism: vertex motion destroys converged appearance; features-FT cannot repaint) — the last naked C1′ arm is dressed. Importance-family ablation: the axis is flat (±0.05 dB) — *having* render evidence is load-bearing, which column barely matters; pipeline noise floor measured at 1.6e-5 dB.
- **C2 (measurement claim) gains replication:** geometric unreliability is now a family property (toy ×3 variants: d1 false-free 58.5–59.3%; SS3DM g1 0.55–0.75 dominated by far-field collapse; courtyard 65%). The evidence-vs-error analysis gives the v3xx machinery its honest role: residual evidence predicts test error (ρ≈0.7, 3/3) but is silent exactly in the coverage gaps where error is largest (5–11×) — the selection effect, now continuously quantified.
- **C4′ preservation-exactness at N=500:** planner outcomes outcome-level EXACT clean↔B50 (identical found/collision sets); new damning consumption datum: courtyard route-i's only 2/500 plans BOTH collide (infeasible AND unsafe); ESDF systematically under-estimates clearance 1.0–2.9 m.
- **A3 (cross-representation) has its context row:** garden 3DGS = +2.8 dB and 3× FPS at matched storage (compression rung ~lossless at 75% keep) — reported plainly; positioning rests on mesh output + geometry/downstream + preservation, not rendering parity. (bicycle/kitchen rows landing.)

## Weakened / bounded
- **B6R stays courtyard-scoped:** on SS3DM the E2R direction transfers (LPIPS/g1/g3-components better 3/3 CI, guard held) but the g3-fraction magnitude arm failed 0/3 — not claim-grade.
- D-2 P1 missed on toy_occl by 0.053 dB via one heavy-tail view (−6.21 dB, E9 family A) — the B50 iso bound formally excludes occlusion-heavy toy scenes.
- B5's LPIPS < B4's replicates on toy-family scenes (features-FT trades LPIPS for PSNR there) — a bounded caveat for the toy rows only (real scenes go the other way).

## Still open (the §10 tail)
- R1 bicycle/kitchen rows (in flight); folding all post-pack rows (B3/E6/D-2/R1/#015) into regenerated T1–T7 + CLAIMS_EVIDENCE_MATRIX (final assembly goal).
- EXPERIMENT_REPORT final assembly; F3 (pipeline diagram draft), F6 (downstream figure), F8 (failure board — E9 assets exist).
- T2 items to run-or-waive with notes: E7-SENS (3 seeds), E8-ROBUST, B6.25, flythrough videos, S-GEO B2 column, laptop bench (waived).
- REPRO_PACK re-verification after the final table regeneration.

## Steering
1. The paper's spine is now: three-regime compaction story + preservation-exactness + the honest consumption/geometry negatives with mechanisms + the measurement suite. Lead T1+F2.
2. The evidence-vs-error reliability curve (F-new from #015 plots) is a strong figure — consider promoting it to the main text as the "why evidence-guided pruning works and where it cannot" bridge.
3. All three tombstoned axes (geometry repair, teacher, consumption certification) share ONE mechanism family (train-coverage selection effects) — write them as one section, not three apologies.
