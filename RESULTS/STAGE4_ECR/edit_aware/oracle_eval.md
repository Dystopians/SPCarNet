# EXP-ORACLE — true-edited-GT scoring

| method | oracle PSNR_R ↑ | oracle MAE_R ↓ | oracle PSNR_U ↑ |
|---|---|---|---|
| ABL_box2d | 26.166 | 0.0266 | 31.305 |
| ABL_dilate16 | 26.254 | 0.0260 | 31.958 |
| ABL_dilate4 | 26.385 | 0.0255 | 32.042 |
| C1_editedbase | 26.062 | 0.0273 | 31.060 |
| C2_stale | 26.576 | 0.0250 | 32.048 |
| C4_rebuild | 25.416 | 0.0380 | 31.170 |
| C5_ours | 26.549 | 0.0251 | 32.047 |
| ORIG_ecr | 23.528 | 0.0428 | 32.049 |
| TM_targetmask | 26.285 | 0.0265 | 32.048 |

**Proxy validation:** Spearman ρ(leak_R, oracle MAE_R) = **0.502** (p = 2.19e-09, n = 126)

**Paired CIs, oracle PSNR_R (C5 − alternative):**
- C5_minus_ABL_box2d: +0.383 [+0.160,+0.655]
- C5_minus_ABL_dilate16: +0.294 [+0.092,+0.560]
- C5_minus_ABL_dilate4: +0.164 [+0.025,+0.378]
- C5_minus_C1_editedbase: +0.487 [+0.241,+0.771]
- C5_minus_C2_stale: -0.028 [-0.077,+0.007]
- C5_minus_C4_rebuild: +1.133 [+0.395,+1.869]
- C5_minus_TM_targetmask: +0.263 [+0.132,+0.417]
