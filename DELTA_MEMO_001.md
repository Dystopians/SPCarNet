# CLAIMS-vs-EVIDENCE Delta Memo #001

2026-07-03 · covers GOALs #009, #R-00..#R-05 · per Stage-1R R4 cadence (one page; steer the writing early)

## Strengthened
- **C1′ (compaction):** now evidenced on THREE suites — S-REND B50 8/9 within −0.10 of legacy clean (4/9 vs the drift-repaired PRIMARY anchor; garden still +0.070 above it), and **S-GEO (SS3DM driving domain) B50 3/4 towns with positive means and LPIPS better 4/4**. The anchor question (reviewer attack A1) is closed pre-emptively with dual rows.
- **C4′ (downstream):** the strongest preservation evidence in the project — **B50 rendered depth is bit-identical to clean** (surf_depth per-pixel), planner outcomes identical (collision diff CI [0,0]). Plus two citable negatives with mechanisms: TSDF fusion worsens safety-critical false-free (grazing-ray bias); raw occupancy grids of both routes fail closed-loop planning (route-i spurious infeasibility from clean-model floater clutter; route-ii real collisions) → quantitative motivation for the certified sub-mesh (R3.b, elevated).
- **C2 (geometry measurement claim):** courtyard E2R-v1 is the project's first positive geometry intervention — g1 −28.5% (CI excl. 0), g3 −34/−41%, LPIPS better, visible cleanup — bounded to real-scene/single-scene status; enters MATRIX as B6R and will be tested on SS3DM towns (courtyard-like over-parameterization profile).

## Weakened
- **The old "beats clean at half triangles" headline** is anchor-dependent: vs PRIMARY (clean-fixed@30k, 9/9 CI-wins over clean@30k) the B50 floor is 4/9; the honest claim is "iso-or-better vs the deployed-default recipe (8/9); iso on 4/9 vs the repaired anchor; garden strictly better vs both". Also clean@26k > fixed@30k on 9/9 — the end-phase decline is NOT fully position-drift; reported as a baseline-trainer property.
- **B25 (75% reduction)**: floor only 2-3/9 S-REND (anchor-dependent) and 1/4 S-GEO — the iso-quality claim consolidates at B50; B25+ becomes the "graceful degradation + FT-value-grows" regime (B5>B4 9/9 at B25).
- **E2R joint bar FAILED** (toy guard −0.38; the fading version of the E2v3 selection effect) — geometry demotion stands for the general claim; only the bounded courtyard finding survives.

## Still naked (needs evidence or explicit scope-out)
- Aggressive budgets B12.5/B6.25 (E1′ says this is where evidence-guidance dominates — the claim's center of gravity; cells still TODO).
- SS3DM geometry table: g1 needs the GT-depth branch (converter POINTS2D empty); g4 needs GT-mesh alignment (cm + mirror) + ROI freeze + Town06 RAM plan.
- B2/B3 baseline columns (random+safe-FT, QEM) beyond courtyard; H1/R1 reference rows (unblocked now that the anchor is frozen).
- R3.b certified sub-mesh (elevated by R3.c) — the make-or-break for "consumable for planning".
- Pristine-submodule build check (reproducibility landmine, R4).
- E9-FAIL taxonomy (candidates accumulating: flowers, town06, toy E2R pinholes, toy 00035 coverage gap).

## Steering suggestions for writing
1. Lead with B50 iso-across-three-suites + FPS/disk halving + preservation-exactness (depth bit-identical, planner CI[0,0]) — this is unassailable.
2. Frame the trainer end-phase decline + features-only repair as a standalone finding box (9/9 CI-wins is a gift).
3. The downstream section writes itself as "how NOT to consume splats" (TSDF falsification + raw-grid planner failures) → R3.b as the constructive answer — pending its verdict.
4. Do not promise geometry improvement; the courtyard E2R box is a bounded observation.
