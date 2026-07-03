# GEMS — EXPERIMENT_REPORT.md (Stage 2, living skeleton)

Started 2026-07-03 (Stage-1R R4). Sections fill from `metrics.json`/analysis artifacts only (no hand-typed numbers in final tables — scripts pending in REPRO_PACK). Claim boundaries: `CLAIMS.md` (incl. NON-CLAIMS) applies verbatim.

## 1. Method summary
Evidence-guided triangle pruning under explicit budgets + drift-safe (features-only) fine-tuning on MeshSplatting checkpoints; single-mouth evaluation (`run_eval.py`, PROTOCOL v1.1.1); all deltas paired per-view bootstrap, 10k resamples, seed 0.

## 2. Results index (fills as MATRIX cells close)
- Anchors: clean@30k (legacy), clean-fixed@30k (PRIMARY; 9/9 CI-wins), clean@26k (context). LEDGER #R-01.
- S-REND Pareto: B50 (8/9 legacy floor, 4/9 primary), B25 (3/9 / 2/9), B12.5 (RUNNING). S-GEO: B50 3/4, B25 1/4. LEDGER #009/#R-05.
- Downstream trilogy: LEDGER #R-02/#R-03/#R-06; artifacts `analysis/r3{a,c,b}_*`.
- Geometry axis: E2/E2R history in KILL_REPORT.md + LEDGER #006/#R-04; courtyard B6R bounded positive.

## 3. NEGATIVE_RESULTS (first-class; every DONE-FAIL ≥1 GPUh)
1. E1/E1′ criteria (Stage One + R1-era verdicts; KILL_REPORT.md).
2. E2 loss routes ×2, E2v3 floater deletion, E2R joint bar (fading selection effect; courtyard positive bounded).
3. E3 teacher distillation (sunset; real sub-floor channel, capture ceiling ≈ view-conditioning).
4. 26k-sourcing (refuted −0.020 CI).
5. R3.a TSDF fusion (false-free ↑26–41%, mechanism: grazing-ray bias).
6. R3.b certification (sheds load-bearing surface; calibrate-once transfer failed honestly).
7. Trainer end-phase decline (baseline property; features-only repair 9/9, yet 26k > fixed-30k 9/9 — decline not fully position-drift).

## 4. RED-TEAM (Stage-1R R4; answers with evidence pointers)
- **A1 "your clean is decayed — re-anchor":** Done first, pre-emptively. PRIMARY anchor = clean-fixed@30k (9/9 CI-wins over clean@30k); ALL comparative tables dual-row (vs clean30k AND vs primary); headline claims name their anchor explicitly (CLAIMS v1.1). Residual honesty: clean@26k > fixed@30k 9/9 (26k is not compute-matched; reported).
- **A2 "importance ≈ random under FT":** E1′ verdict reported verbatim (courtyard B50 +0.13 CI incl. 0 with safe FT). The claim lives where measured: aggressive budgets/no-FT (random collapses 2.4–5.3 dB at B25-no-FT; B12.5/B6.25 cells in progress); moderate-budget value is the FT channel itself (B5>B4 9/9 at B25).
- **A3 "modest ratios vs GS-compression literature":** positioning = reliability/deployment for MESH-based splatting (plain checkpoint out, FPS/disk halved, preservation-exactness incl. bit-identical depth); R1-3DGS reference row pending (context only, non-claim).
- **A4 "'for parking' without geometry improvement":** answered by the R3 trilogy: preservation-exactness + falsified consumption routes with mechanisms + a frozen quantitative fix-target (≥30/100 @ ≤3 coll/100 on courtyard) + SS3DM driving-domain Pareto (B50 3/4) + bounded courtyard geometry positive (B6R). No aspiration in claims; NON-CLAIMS list the boundary verbatim.

## 5. Statistics & caveats
Paired per-view bootstrap everywhere; multiple-comparisons caveat: dozens of CIs are reported across this program; individual borderline CIs (|effect| near floor) should be read with Bonferroni-style skepticism — headline claims rest only on effects that are large, replicated across scenes/suites, or mechanism-backed. Courtyard rendering CIs are 5-view (underpowered by design; geometry/downstream metrics carry that scene). Per-scene breakdowns accompany every mean (LEDGER tables).

## 6. Limitations
Per-scene optimization setting; toy_parking's 62× over-parameterization makes it adversarial for junk-removal mechanisms (selection effect — documented thrice); SS3DM g1/g4 pending (GT-depth branch, mesh alignment); courtyard z_band approximate; GT-scan unscanned voxels count free (collision lower bounds).
