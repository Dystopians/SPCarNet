# T1 — Main Pareto summary (E1-PARETO / E10)

_generated 2026-07-04T07:47:24.012716+00:00 by tools/gems/report/tables.py — every number computed from metrics.json-derived artifacts; none hand-typed._

> STATS: every delta is a paired per-view bootstrap 95% CI (10k resamples, seed 0, PROTOCOL section 5). MULTIPLE-COMPARISONS CAVEAT (E10): dozens of CIs are reported across this pack; borderline CIs (effect near a floor or CI edge near 0) should be read with Bonferroni-style skepticism — headline claims rest only on effects that are large, replicated across scenes/suites, or mechanism-backed. Courtyard rendering CIs are 5-view (underpowered by design). Reporting language per section 6: 'improves/reduces' ONLY when the CI excludes 0 AND the D3 floor is cleared; otherwise 'comparable'/'inconclusive'.
> Anchors: B0 = clean@30k (legacy/deployed default); B0' = clean-fixed@30k (PRIMARY anchor per LEDGER GOAL#R-01; exists on S-REND only — S-GEO/S-DEV anchor columns vs B0' are blank by construction).
> w/i/l = per-scene win/iso/loss counts by paired 95% CI (win = CI excludes 0 in the improving direction). iso-floor pass = scenes with mean dPSNR >= -0.1 AND mean dLPIPS <= +0.005 vs B0 (D3 compaction iso-quality floor).
> MISSING BY DESIGN / KNOWN GAPS (honesty, section 7.3/7.4; updated pack v3 after the GOAL#019 gap-closure runs): B1 no-op pass-through WAS RUN (GOAL#019): garden budget=1.0 reproduces the clean checkpoint EXACTLY (paired dPSNR identically zero, identical triangle count — B1 row in T2/T7). B6.25 far-end rows exist on garden/kitchen/ss3dm_town01 (GOAL#019); other scenes scope-frozen. B3 QEM+FT exists at B50 on garden/toy_parking/courtyard only (GOAL#013 DONE-PASS; breadth scope-frozen per the GOAL#019 run-instead-of-waive ruling, MATRIX E1 notes). B2 exists at B12.5 on all 9 S-REND scenes, at B50/B25 on garden/toy/courtyard (e1b era), and at B50 on all 4 SS3DM towns (GOAL#019: importance beats random on S-GEO 4/4 CI-excl-0 — see T6/F7); remaining B2 cells (B50/B25 on the other 6 S-REND scenes) scope-frozen. B6/B7 appear only as diagnostic rows on dev scenes (both DEMOTED per CLAIMS.md). B6R rows are ablation rows (E2R FAILED its joint bar; GOAL#R-04/#014) shown for the bounded-positive context. H1/R1 are context appendices below, not corpus rows.

| suite | budget | method | scenes | tri_ratio_vs_B0 | mean dPSNR vs B0 [dB] | PSNR w/i/l vs B0 | mean dLPIPS vs B0 | LPIPS w/i/l vs B0 | mean dPSNR vs B0' | PSNR w/i/l vs B0' | mean dLPIPS vs B0' | LPIPS w/i/l vs B0' | iso-floor pass (vs B0) | iso-floor pass (vs B0') | mean FPS x vs B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-REND | B50 | B5 | 9/9 | 0.500 | +0.050 | 5/3/1 | -0.004 | 9/0/0 | -0.092 | 1/2/6 | +0.002 | 2/2/5 | 8/9 | 4/9 | 1.29 |
| S-REND | B50 | B4 | 9/9 | 0.500 | -0.061 | 1/2/6 | +0.000 | 0/3/6 | -0.204 | 0/0/9 | +0.006 | 0/0/9 | 7/9 | 3/9 | 1.29 |
| S-REND | B50 | B2 | 1/9 | 0.500 | -3.334 | 0/0/1 | +0.149 | 0/0/1 | -3.403 | 0/0/1 | +0.152 | 0/0/1 | 0/1 | 0/1 | 1.38 |
| S-REND | B50 | B3 | 1/9 | 0.500 | -3.255 | 0/0/1 | +0.137 | 0/0/1 | -3.324 | 0/0/1 | +0.140 | 0/0/1 | 0/1 | 0/1 | 1.03 |
| S-REND | B50 | B5-iter | 1/9 | 0.500 | +0.157 | 1/0/0 | -0.007 | 1/0/0 | +0.088 | 1/0/0 | -0.004 | 1/0/0 | 1/1 | 1/1 | 1.29 |
| S-REND | B25 | B5 | 9/9 | 0.250 | -0.168 | 1/1/7 | +0.002 | 1/4/4 | -0.311 | 0/1/8 | +0.008 | 0/0/9 | 3/9 | 2/9 | 1.61 |
| S-REND | B25 | B4 | 9/9 | 0.250 | -0.283 | 0/0/9 | +0.007 | 0/0/9 | -0.426 | 0/0/9 | +0.013 | 0/0/9 | 0/9 | 0/9 | 1.61 |
| S-REND | B25 | B2 | 1/9 | 0.250 | -5.017 | 0/0/1 | +0.212 | 0/0/1 | -5.086 | 0/0/1 | +0.215 | 0/0/1 | 0/1 | 0/1 | 1.73 |
| S-REND | B12.5 | B5 | 9/9 | 0.125 | -1.352 | 0/0/9 | +0.025 | 0/0/9 | -1.494 | 0/0/9 | +0.031 | 0/0/9 | 0/9 | 0/9 | 1.94 |
| S-REND | B12.5 | B4 | 9/9 | 0.125 | -1.510 | 0/0/9 | +0.034 | 0/0/9 | -1.653 | 0/0/9 | +0.040 | 0/0/9 | 0/9 | 0/9 | 1.93 |
| S-REND | B12.5 | B2 | 9/9 | 0.125 | -6.583 | 0/0/9 | +0.177 | 0/0/9 | -6.726 | 0/0/9 | +0.183 | 0/0/9 | 0/9 | 0/9 | 1.92 |
| S-REND | B6.25 | B5 | 2/9 | 0.062 | -4.384 | 0/0/2 | +0.081 | 0/0/2 | -4.575 | 0/0/2 | +0.090 | 0/0/2 | 0/2 | 0/2 | 2.37 |
| S-REND | B6.25 | B4 | 2/9 | 0.062 | -4.693 | 0/0/2 | +0.102 | 0/0/2 | -4.885 | 0/0/2 | +0.111 | 0/0/2 | 0/2 | 0/2 | 2.38 |
| S-GEO | B50 | B5 | 4/4 | 0.500 | -0.061 | 1/2/1 | -0.003 | 4/0/0 | — | — | — | — | 3/4 | — | 1.30 |
| S-GEO | B50 | B4 | 4/4 | 0.500 | -0.169 | 0/0/4 | +0.000 | 0/1/3 | — | — | — | — | 3/4 | — | 0.99 |
| S-GEO | B50 | B2 | 4/4 | 0.500 | -1.102 | 0/0/4 | +0.028 | 0/0/4 | — | — | — | — | 0/4 | — | 1.29 |
| S-GEO | B50 | B6R | 3/4 | 0.482 | +0.076 | 1/2/0 | -0.005 | 3/0/0 | — | — | — | — | 3/3 | — | 1.34 |
| S-GEO | B25 | B5 | 4/4 | 0.250 | -0.471 | 0/1/3 | -0.000 | 1/2/1 | — | — | — | — | 1/4 | — | 1.64 |
| S-GEO | B25 | B4 | 4/4 | 0.250 | -0.576 | 0/0/4 | +0.003 | 0/0/4 | — | — | — | — | 0/4 | — | 1.64 |
| S-GEO | B6.25 | B5 | 1/4 | 0.062 | -3.255 | 0/0/1 | +0.082 | 0/0/1 | — | — | — | — | 0/1 | — | 2.44 |
| S-GEO | B6.25 | B4 | 1/4 | 0.062 | -3.364 | 0/0/1 | +0.090 | 0/0/1 | — | — | — | — | 0/1 | — | 2.49 |
| S-DEV | B50 | B5 | 4/4 | 0.500 | -0.433 | 0/0/4 | +0.011 | 0/0/4 | — | — | — | — | 0/4 | — | 1.21 |
| S-DEV | B50 | B4 | 4/4 | 0.500 | -0.424 | 0/0/4 | +0.001 | 1/0/3 | — | — | — | — | 1/4 | — | 1.21 |
| S-DEV | B50 | B2 | 2/4 | 0.500 | -1.186 | 0/1/1 | +0.062 | 0/0/2 | — | — | — | — | 0/2 | — | 1.25 |
| S-DEV | B50 | B3 | 2/4 | 0.500 | -1.751 | 0/1/1 | +0.049 | 0/0/2 | — | — | — | — | 0/2 | — | 1.00 |
| S-DEV | B50 | B5-iter | 1/4 | 0.500 | -0.542 | 0/0/1 | +0.015 | 0/0/1 | — | — | — | — | 0/1 | — | 1.19 |
| S-DEV | B50 | B6R | 2/4 | 0.485 | -0.283 | 0/1/1 | +0.008 | 0/1/1 | — | — | — | — | 1/2 | — | 0.87 |
| S-DEV | B25 | B5 | 2/4 | 0.250 | -0.772 | 0/0/2 | +0.007 | 0/1/1 | — | — | — | — | 0/2 | — | 1.39 |
| S-DEV | B25 | B4 | 2/4 | 0.250 | -0.778 | 0/0/2 | +0.002 | 0/1/1 | — | — | — | — | 0/2 | — | 1.40 |
| S-DEV | B25 | B2 | 1/4 | 0.250 | -4.533 | 0/0/1 | +0.147 | 0/0/1 | — | — | — | — | 0/1 | — | 1.54 |

---

## CONTEXT appendix (NOT corpus rows — no CIs vs GEMS)

### R1 — 3DGS + storage-matched opacity-prune reference (LEDGER GOAL#017; sanctioned outside-single-mouth exception; full table: `analysis/r1_3dgs_reference/r1_table.md`)

Context only per Stage2 prompt section 4: NOT a claim target; NON-CLAIMS already disclaim SOTA novel-view quality vs the 3DGS family. Same llff8 splits/resolutions (name-asserted); disk MB = each representation's shippable artifact.

| scene | method | PSNR | LPIPS | primitives | disk MB | FPS |
|---|---|---|---|---|---|---|
| garden | 3DGS 30k (vanilla) | 27.503 | 0.1062 | 4,158,575 gaussians | 983.6 | 100.8 |
| garden | 3DGS opacity-prune+FT5k (storage-matched) | 27.513 | 0.1064 | 3,106,001 gaussians | 734.6 | 140.3 |
| garden | GEMS B0 (corpus row, quoted) | 24.712 | 0.2163 | 11,568,056 triangles | 942.0 | 32.3 |
| garden | GEMS B5@B50 (corpus row, quoted) | 24.851 | 0.2101 | 5,784,028 triangles | 734.6 | 41.7 |
| bicycle | 3DGS 30k (vanilla) | 25.241 | 0.2088 | 4,925,145 gaussians | 1164.9 | 79.4 |
| bicycle | 3DGS opacity-prune+FT5k (storage-matched) | 25.263 | 0.2092 | 2,892,207 gaussians | 684.0 | 156.2 |
| bicycle | GEMS B0 (corpus row, quoted) | 23.021 | 0.3473 | 9,422,930 triangles | 908.1 | 38.6 |
| bicycle | GEMS B5@B50 (corpus row, quoted) | 23.135 | 0.3429 | 4,711,465 triangles | 684.0 | 49.6 |
| kitchen | 3DGS 30k (vanilla) | 30.803 | 0.1270 | 1,592,262 gaussians | 376.6 | 149.4 |
| kitchen | 3DGS opacity-prune+FT5k (storage-matched) | 30.803 | 0.1270 | 1,592,262 gaussians | 376.6 | 149.4 |
| kitchen | GEMS B0 (corpus row, quoted) | 27.296 | 0.2262 | 9,716,239 triangles | 708.7 | 26.4 |
| kitchen | GEMS B5@B50 (corpus row, quoted) | 27.449 | 0.2191 | 4,858,119 triangles | 527.3 | 33.8 |

A3 positioning (verbatim conclusion lives in r1_table.md): at matched artifact storage 3DGS renders these scenes 2.1–3.4 dB above GEMS B5@B50 at 3.1–4.4x FPS; GEMS's deliverables (mesh artifact, g1–g4/downstream consumability, preservation-exactness, 50%-reduction-at-iso) have no 3DGS-family equivalent artifact. R1 contextualizes, gates nothing.

### H1 — v106 historical context row (LEDGER GOAL#013; `analysis/h1_v106_context/h1_v106_context_row.json`)

v106 POD-MoE base-preserve: full9 mean PSNR 25.831 / SSIM 0.7608 / LPIPS 0.2684 (+0.680 dB vs its era's clean row). Different mechanism (baked render-time residual field), different protocol era (legacy metrics.py mouth), NO triangle budget, and a documented target-camera-sidecar dependency — NEVER CI-compared to GEMS rows; quotable but not re-runnable (checkpoints existed only on tmpfs).
