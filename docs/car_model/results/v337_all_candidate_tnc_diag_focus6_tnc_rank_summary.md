# v337 All-Candidate TNC Focus6 Rank Summary

Root: `outputs/carnet/spcarnet_v337_all_candidate_tnc_diag_focus6_20260701`

## Macro

| scenes | views | available | TNC matches strict oracle | match frac | oracle-output PSNR | TNC-best-output PSNR | oracle rank | output rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 170 | 170 | 37 | 0.217647 | +0.009893758799 | -0.051566646277 | 3.894118 | 3.682353 |

## Per Scene

| scene | views | match frac | oracle-output PSNR | TNC-best-output PSNR | oracle rank | output rank | TNC best counts | strict oracle counts |
|---|---:|---:|---:|---:|---:|---:|---|---|
| stump | 16 | 0.500000 | +0.021858285120 | +0.001558000550 | 2.625000 | 2.125000 | `{'fixed': 11, 'hybrid': 3, 'learned': 2}` | `{'fixed': 9, 'learned': 5, 'mix0750': 2}` |
| treehill | 18 | 0.611111 | +0.015234013794 | -0.005465829862 | 2.388889 | 1.888889 | `{'fixed': 9, 'learned': 8, 'mix0250': 1}` | `{'fixed': 4, 'hybrid': 1, 'learned': 12, 'mix0250': 1}` |
| room | 39 | 0.025641 | +0.011576829779 | -0.045652079578 | 4.948718 | 4.435897 | `{'adaptive': 1, 'fixed': 31, 'hybrid': 1, 'learned': 4, 'mix0750': 2}` | `{'adaptive': 14, 'hybrid': 2, 'learned': 21, 'mix0750': 2}` |
| bicycle | 25 | 0.280000 | +0.010343502985 | -0.017025614542 | 3.080000 | 3.240000 | `{'fixed': 16, 'learned': 7, 'mix0250': 1, 'mix0750': 1}` | `{'fixed': 7, 'hybrid': 7, 'learned': 9, 'mix0250': 1, 'mix0750': 1}` |
| bonsai | 37 | 0.162162 | +0.007299128751 | -0.092994969647 | 4.189189 | 4.216216 | `{'fixed': 28, 'hybrid': 2, 'learned': 5, 'mix0250': 1, 'mix0750': 1}` | `{'fixed': 5, 'hybrid': 1, 'learned': 30, 'mix0750': 1}` |
| kitchen | 35 | 0.114286 | +0.002224071022 | -0.087028216981 | 4.342857 | 4.228571 | `{'fixed': 26, 'hybrid': 1, 'learned': 6, 'mix0250': 1, 'mix0750': 1}` | `{'fixed': 1, 'learned': 32, 'mix0750': 2}` |

## Verdict

Pure target-neighbor MAE ranking is not sufficient as a standalone selector on focus6. It matches the strict oracle on only a minority of views and its best-ranked candidate is worse than the selected output on average. Use this signal as a feature/certificate, not as the sole policy.
