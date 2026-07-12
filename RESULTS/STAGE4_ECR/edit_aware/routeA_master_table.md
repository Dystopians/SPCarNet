# Route-A MASTER TABLE (pre-submission canonical form)

**Statistics:** pairing unit = test view; paired per-view bootstrap, 10,000 resamples, seed 0, percentile CIs. Effect directions: oracle PSNR_R higher = better edited-region fidelity to TRUE edited GT; ghost_psnr_R higher = MORE stale-content similarity (worse); psnr_U higher = better true-GT preservation outside the region. **Multiple comparisons:** the 5-way novelty family is reported at BOTH 95% and 99% CIs (Bonferroni alpha = 0.05/5 = 0.01). **Oracle scope:** true edited GT exists ONLY for the synthetic scene (verified rebuild); real-scene cells report content preservation + bounded ghost metrics — no edited-GT access is implied. leak_R is SECONDARY (rho = 0.502 vs oracle error; penalizes legitimate improvement).

## A. Oracle-scored novelty family (toy car_0 deletion; n = 18 test views)

| C5 (ours) minus | Δ oracle PSNR_R | 95% CI | 99% CI | excl. 0 @95 / @99 |
|---|---|---|---|---|
| ABL_dilate4 | +0.164 | [+0.025,+0.378] | [+0.011,+0.467] | Y / Y |
| ABL_dilate16 | +0.294 | [+0.092,+0.560] | [+0.054,+0.668] | Y / Y |
| ABL_box2d | +0.383 | [+0.160,+0.655] | [+0.116,+0.756] | Y / Y |
| TM_targetmask | +0.263 | [+0.132,+0.417] | [+0.098,+0.463] | Y / Y |
| C4_rebuild | +1.133 | [+0.395,+1.869] | [+0.148,+2.080] | Y / Y |

Honest non-member of the family: C5 − C2_stale = -0.028 [-0.077,+0.007] (tie on DELETION; C2 fails recolor +1.964 [+1.869,+2.061] and chained recolor +2.669 [+1.326,+4.085] — bounded ghost metric, real scene).

## B. Real-scene cells (bounded metrics; no edited GT)

| cell | n views | C5 leak_R (secondary) | ghost C5−C1 [95% CI] | U preservation C5−ORIG [95% CI] (true GT) |
|---|---|---|---|---|
| garden table delete (2,037,550 faces) | 24 | 0.0065 | +0.059 [+0.038,+0.081] | -0.020 [-0.034,-0.008] |
| garden table recolor | 24 | 0.0034 | +0.058 [+0.048,+0.069] | +0.015 [+0.005,+0.023] |
| garden pot delete (peripheral, 24,952 faces) | 24 | 0.0037 | +0.039 [+0.016,+0.069] | -0.000 [-0.000,-0.000] |
| garden chained delete->recolor | 24 | 0.0029 | +0.043 [+0.021,+0.069] | -0.000 [-0.000,-0.000] |
| toy car_1 delete (711,609+ faces) | 18 | 0.0025 | +4.240 [+0.153,+10.286] | -0.002 [-0.004,-0.001] |

## C. Update cost & affected-view reconciliation

| cell | affected / train views | note | bytes (dense) | bytes (sparse sidecar) | wall |
|---|---|---|---|---|---|
| garden table (central) | 161 / 161 | central object: visible in every train view | 1053 MB | n/a (dense run) | 108 s |
| garden pot (peripheral) | 57 / 161 | TRUE view-locality: 35% of views | 369 MB | n/a (dense run) | 42 s |
| toy car_0 | 72 / 72 | 72 = ALL of toy's TRAIN views (its 90 total views include 18 test; the dataset census's '76/90' counts coverage over ALL views incl. test) | 231 MB | **12.8 MB (validated bit-equal, same process)** | 34 s |

Reconciliation: 57/161 and 72/72 are DIFFERENT SCENES and denominators — garden has 161 train views (pot affects 57); toy_parking has 72 train views (car_0 affects all 72; the widely-quoted 76/90 figure is the dataset's whole-set coverage census including test views).
