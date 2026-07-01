# v338 Combined Ranker Rejection And Relaxation Diagnostic

Root: `outputs/carnet/spcarnet_v338b_rank1_cvar002_focus6_20260701`

Important: this is a read-only post-hoc diagnostic over saved v338b reports. It uses target GT only to evaluate what a relaxed policy would have done, so it is not a fair deployable selector.

## v338b Actual Outcome

| scene | views | promote | keep | top reject reasons |
|---|---:|---:|---:|---|
| stump | 16 | 0 | 16 | target_neighbor_rank:59, target_neighbor_margin:5 |
| treehill | 18 | 0 | 18 | target_neighbor_rank:54, fixed_when_incumbent_nonfixed:13, source_local:local_ssim:3, target_neighbor_margin:2 |
| room | 39 | 0 | 39 | target_neighbor_rank:149, fixed_when_incumbent_nonfixed:39, target_neighbor_margin:7 |
| bicycle | 25 | 0 | 25 | target_neighbor_rank:66, fixed_when_incumbent_nonfixed:25, target_neighbor_margin:8, source_local:local_psnr,local_ssim,local_cvar:1 |
| bonsai | 37 | 0 | 37 | target_neighbor_rank:109, fixed_when_incumbent_nonfixed:34, target_neighbor_margin:5 |
| kitchen | 35 | 0 | 35 | target_neighbor_rank:102, fixed_when_incumbent_nonfixed:35, target_neighbor_margin:3 |

## Top Relaxed Diagnostics By Macro PSNR

| config | dPSNR | dSSIM | promotions | bad promotions | nonneg PSNR scenes | nonneg SSIM scenes |
|---|---:|---:|---:|---:|---:|---:|
| r3_m-0.0001_lp-0.002_ls-5e-05 | +0.000240408751 | +0.000004195381 | 18 | 10 | 5/6 | 5/6 |
| r3_m-0.0001_lp0.0_ls-5e-05 | +0.000156450016 | +0.000002897987 | 17 | 10 | 5/6 | 5/6 |
| r3_m-0.0001_lp-0.005_ls-5e-05 | +0.000108022288 | +0.000002830768 | 24 | 14 | 4/6 | 5/6 |
| r3_m0.0_lp-0.002_ls-5e-05 | -0.000328726671 | -0.000000849721 | 12 | 7 | 4/6 | 5/6 |
| r3_m0.0_lp0.0_ls-5e-05 | -0.000412685407 | -0.000002147116 | 11 | 7 | 4/6 | 5/6 |
| r3_m0.0_lp-0.005_ls-5e-05 | -0.000461113135 | -0.000002214335 | 18 | 11 | 4/6 | 4/6 |
| r3_m-0.0001_lp-0.02_ls-5e-05 | -0.000568391143 | -0.000001740758 | 31 | 20 | 3/6 | 4/6 |
| r2_m-0.0001_lp-0.005_ls-5e-05 | -0.000609555992 | -0.000001599347 | 21 | 13 | 5/6 | 5/6 |

## Verdict

The best posterior relaxed setting gives only a tiny macro PSNR gain and still creates scene regressions plus many bad promotions. Therefore v338 should be treated as a diagnostic/negative result, not as a promoted method. The practical lesson is that the current bottleneck is candidate-generation capacity, not another target-neighbor rank threshold.
