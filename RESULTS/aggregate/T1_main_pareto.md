# T1 — Main Pareto summary (E1-PARETO / E10)

_generated 2026-07-03T19:26:47.250398+00:00 by tools/gems/report/tables.py — every number computed from metrics.json-derived artifacts; none hand-typed._

> STATS: every delta is a paired per-view bootstrap 95% CI (10k resamples, seed 0, PROTOCOL section 5). MULTIPLE-COMPARISONS CAVEAT (E10): dozens of CIs are reported across this pack; borderline CIs (effect near a floor or CI edge near 0) should be read with Bonferroni-style skepticism — headline claims rest only on effects that are large, replicated across scenes/suites, or mechanism-backed. Courtyard rendering CIs are 5-view (underpowered by design). Reporting language per section 6: 'improves/reduces' ONLY when the CI excludes 0 AND the D3 floor is cleared; otherwise 'comparable'/'inconclusive'.
> Anchors: B0 = clean@30k (legacy/deployed default); B0' = clean-fixed@30k (PRIMARY anchor per LEDGER GOAL#R-01; exists on S-REND only — S-GEO/S-DEV anchor columns vs B0' are blank by construction).
> w/i/l = per-scene win/iso/loss counts by paired 95% CI (win = CI excludes 0 in the improving direction). iso-floor pass = scenes with mean dPSNR >= -0.1 AND mean dLPIPS <= +0.005 vs B0 (D3 compaction iso-quality floor).
> MISSING BY DESIGN (honesty, section 7.3/7.4): B1 no-op, B3 QEM+FT and H1/R1 reference rows have NOT been run (MATRIX: TODO); B2 exists at B12.5 on all 9 S-REND scenes but only on garden/toy/courtyard at B50/B25 (e1b era). B6/B7 appear only as diagnostic rows on dev scenes (both DEMOTED per CLAIMS.md). S-GEO B2 was never run.

| suite | budget | method | scenes | tri_ratio_vs_B0 | mean dPSNR vs B0 [dB] | PSNR w/i/l vs B0 | mean dLPIPS vs B0 | LPIPS w/i/l vs B0 | mean dPSNR vs B0' | PSNR w/i/l vs B0' | mean dLPIPS vs B0' | LPIPS w/i/l vs B0' | iso-floor pass (vs B0) | iso-floor pass (vs B0') | mean FPS x vs B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-REND | B50 | B5 | 9/9 | 0.500 | +0.050 | 5/3/1 | -0.004 | 9/0/0 | -0.092 | 1/2/6 | +0.002 | 2/2/5 | 8/9 | 4/9 | 1.29 |
| S-REND | B50 | B4 | 9/9 | 0.500 | -0.061 | 1/2/6 | +0.000 | 0/3/6 | -0.204 | 0/0/9 | +0.006 | 0/0/9 | 7/9 | 3/9 | 1.29 |
| S-REND | B50 | B2 | 1/9 | 0.500 | -3.334 | 0/0/1 | +0.149 | 0/0/1 | -3.403 | 0/0/1 | +0.152 | 0/0/1 | 0/1 | 0/1 | 1.38 |
| S-REND | B50 | B5-iter | 1/9 | 0.500 | +0.157 | 1/0/0 | -0.007 | 1/0/0 | +0.088 | 1/0/0 | -0.004 | 1/0/0 | 1/1 | 1/1 | 1.29 |
| S-REND | B25 | B5 | 9/9 | 0.250 | -0.168 | 1/1/7 | +0.002 | 1/4/4 | -0.311 | 0/1/8 | +0.008 | 0/0/9 | 3/9 | 2/9 | 1.61 |
| S-REND | B25 | B4 | 9/9 | 0.250 | -0.283 | 0/0/9 | +0.007 | 0/0/9 | -0.426 | 0/0/9 | +0.013 | 0/0/9 | 0/9 | 0/9 | 1.61 |
| S-REND | B25 | B2 | 1/9 | 0.250 | -5.017 | 0/0/1 | +0.212 | 0/0/1 | -5.086 | 0/0/1 | +0.215 | 0/0/1 | 0/1 | 0/1 | 1.73 |
| S-REND | B12.5 | B5 | 9/9 | 0.125 | -1.352 | 0/0/9 | +0.025 | 0/0/9 | -1.494 | 0/0/9 | +0.031 | 0/0/9 | 0/9 | 0/9 | 1.94 |
| S-REND | B12.5 | B4 | 9/9 | 0.125 | -1.510 | 0/0/9 | +0.034 | 0/0/9 | -1.653 | 0/0/9 | +0.040 | 0/0/9 | 0/9 | 0/9 | 1.93 |
| S-REND | B12.5 | B2 | 9/9 | 0.125 | -6.583 | 0/0/9 | +0.177 | 0/0/9 | -6.726 | 0/0/9 | +0.183 | 0/0/9 | 0/9 | 0/9 | 1.92 |
| S-GEO | B50 | B5 | 4/4 | 0.500 | -0.061 | 1/2/1 | -0.003 | 4/0/0 | — | — | — | — | 3/4 | — | 1.30 |
| S-GEO | B50 | B4 | 4/4 | 0.500 | -0.169 | 0/0/4 | +0.000 | 0/1/3 | — | — | — | — | 3/4 | — | 0.99 |
| S-GEO | B25 | B5 | 4/4 | 0.250 | -0.471 | 0/1/3 | -0.000 | 1/2/1 | — | — | — | — | 1/4 | — | 1.64 |
| S-GEO | B25 | B4 | 4/4 | 0.250 | -0.576 | 0/0/4 | +0.003 | 0/0/4 | — | — | — | — | 0/4 | — | 1.64 |
| S-DEV | B50 | B5 | 2/2 | 0.500 | -0.322 | 0/0/2 | +0.008 | 0/0/2 | — | — | — | — | 0/2 | — | 1.19 |
| S-DEV | B50 | B4 | 2/2 | 0.500 | -0.279 | 0/0/2 | +0.000 | 1/0/1 | — | — | — | — | 1/2 | — | 1.20 |
| S-DEV | B50 | B2 | 2/2 | 0.500 | -1.186 | 0/1/1 | +0.062 | 0/0/2 | — | — | — | — | 0/2 | — | 1.25 |
| S-DEV | B50 | B5-iter | 1/2 | 0.500 | -0.542 | 0/0/1 | +0.015 | 0/0/1 | — | — | — | — | 0/1 | — | 1.19 |
| S-DEV | B50 | B6R | 2/2 | 0.485 | -0.283 | 0/1/1 | +0.008 | 0/1/1 | — | — | — | — | 1/2 | — | 0.87 |
| S-DEV | B25 | B5 | 2/2 | 0.250 | -0.772 | 0/0/2 | +0.007 | 0/1/1 | — | — | — | — | 0/2 | — | 1.39 |
| S-DEV | B25 | B4 | 2/2 | 0.250 | -0.778 | 0/0/2 | +0.002 | 0/1/1 | — | — | — | — | 0/2 | — | 1.40 |
| S-DEV | B25 | B2 | 1/2 | 0.250 | -4.533 | 0/0/1 | +0.147 | 0/0/1 | — | — | — | — | 0/1 | — | 1.54 |
