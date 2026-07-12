# Cross-process sensitivity study (task 4)

Full re-render + re-eval of the toy oracle cell in a fresh process; conclusions must be rank/CI-stable despite quantization-level cross-process render nondeterminism.

- Method-ranking Spearman rho (oracle PSNR_R means, run1 vs run2, 9 methods): **0.983**
- Novelty-family CI stability (C5 minus alternative, oracle PSNR_R, 95% CI excl. 0?):

| comparison | run 1 | run 2 | conclusion stable |
|---|---|---|---|
| C5_minus_ABL_box2d | +0.383 [+0.160,+0.655] (Y) | +0.641 [+0.281,+1.068] (Y) | YES |
| C5_minus_ABL_dilate16 | +0.294 [+0.092,+0.560] (Y) | +0.213 [+0.092,+0.357] (Y) | YES |
| C5_minus_ABL_dilate4 | +0.164 [+0.025,+0.378] (Y) | +0.080 [+0.018,+0.163] (Y) | YES |
| C5_minus_C1_editedbase | +0.487 [+0.241,+0.771] (Y) | +0.913 [+0.365,+1.645] (Y) | YES |
| C5_minus_C2_stale | -0.028 [-0.077,+0.007] (N) | -0.010 [-0.029,+0.008] (N) | YES |
| C5_minus_C4_rebuild | +1.133 [+0.395,+1.869] (Y) | +0.753 [+0.299,+1.247] (Y) | YES |
| C5_minus_TM_targetmask | +0.263 [+0.132,+0.417] (Y) | +0.178 [+0.083,+0.300] (Y) | YES |

- Proxy correlation run2: rho = 0.265 (run1: 0.502)

**Verdict: ALL rankings and CI conclusions STABLE across processes.**

Note: the leak_R proxy correlation is itself UNSTABLE across processes (0.502 -> 0.265) — additional
evidence for its secondary status (the oracle-primary hierarchy is required, not optional).
