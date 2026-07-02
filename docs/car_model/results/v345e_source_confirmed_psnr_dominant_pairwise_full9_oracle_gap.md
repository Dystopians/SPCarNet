# Support-Transport Oracle Gap Audit

Read-only diagnostic over metrics already serialized in support_transport_apply_report.json.

## Macro

| method | scenes | views | primary | selected mean | oracle mean | mean headroom | positive views | selected counts | best counts |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| v343e_replay | 9 | 246 | psnr_gain | 0.276768570 | 0.284990320 | +0.008221750 | 95 | adaptive:19, fixed:14, hybrid:39, learned:146, mix0750:1, source_trust:27 | adaptive:1, fixed:34, hybrid:3, learned:177, mix0250:1, mix0750:7, source_trust:23 |
| v345e | 9 | 246 | psnr_gain | 0.276976720 | 0.284990320 | +0.008013599 | 95 | adaptive:19, fixed:13, hybrid:39, learned:146, mix0250:1, mix0750:1, source_trust:27 | adaptive:1, fixed:34, hybrid:3, learned:177, mix0250:1, mix0750:7, source_trust:23 |

## v343e_replay

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 9 | 246 | 0.276768570 | 0.284990320 | +0.008221750 | +0.006842166 | 95 |
| candidate_psnr | max | 9 | 246 | 25.417162796 | 25.425384546 | +0.008221750 | +0.006842166 | 95 |
| ssim_gain | max | 9 | 246 | 0.003767040 | 0.003857935 | +0.000090894 | +0.000085947 | 116 |
| candidate_ssim | max | 9 | 246 | 0.840521045 | 0.840611939 | +0.000090894 | +0.000085947 | 116 |
| mse_reduction | max | 9 | 246 | 0.000172071 | 0.000181887 | +0.000009817 | +0.000007297 | 95 |
| candidate_mse | min | 9 | 246 | 0.004434015 | 0.004424198 | +0.000009817 | +0.000007297 | 95 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| fixed_scene_generated_source_summary_unlock | 16 | +0.020859960 | +0.067380167 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |
| knn | 17 | +0.002166590 | +0.036832026 | 1 | learned:17 | learned:16, source_trust:1 |
| pairwise | 2 | +0.010211082 | +0.017349122 | 2 | hybrid:1, source_trust:1 | mix0750:1, source_trust:1 |
| scene | 117 | +0.007549447 | +0.131198225 | 46 | fixed:9, hybrid:37, learned:71 | fixed:15, hybrid:2, learned:95, mix0250:1, mix0750:3, source_trust:1 |
| source_oracle_knn | 8 | +0.010084050 | +0.054011185 | 3 | fixed:1, learned:4, source_trust:3 | fixed:1, learned:5, source_trust:2 |
| source_reliability | 83 | +0.003954236 | +0.052045476 | 28 | adaptive:3, fixed:4, hybrid:1, learned:51, mix0750:1, source_trust:23 | fixed:11, hybrid:1, learned:51, mix0750:2, source_trust:18 |
| target_neighbor_unlock | 3 | +0.000000000 | +0.000000000 | 0 | learned:3 | learned:3 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.582901932 | 0.583337117 | +0.000435186 | 3 | fixed:4, hybrid:1, learned:32 | fixed:6, learned:30, mix0750:1 |  |
| counter | ok | 30 | psnr_gain | 0.426359637 | 0.427055163 | +0.000695526 | 2 | learned:30 | fixed:1, learned:28, mix0750:1 |  |
| flowers | ok | 22 | psnr_gain | 0.092359632 | 0.097795440 | +0.005435808 | 13 | hybrid:13, learned:9 | fixed:1, learned:20, mix0750:1 |  |
| garden | ok | 24 | psnr_gain | 0.147204102 | 0.152575774 | +0.005371671 | 11 | fixed:1, hybrid:3, learned:12, mix0750:1, source_trust:7 | fixed:3, learned:14, source_trust:7 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.453250186 | 0.462676525 | +0.009426339 | 22 | adaptive:3, hybrid:2, learned:14, source_trust:20 | fixed:3, learned:19, mix0750:1, source_trust:16 |  |
| stump | ok | 16 | psnr_gain | 0.058909355 | 0.079769316 | +0.020859960 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.116350575 | 0.133574687 | +0.017224112 | 6 | fixed:9, learned:9 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| treehill | 00011 | scene | fixed | learned | psnr_gain | 0.294420490 | 0.425618716 | +0.131198225 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.105065691 | 0.172445859 | +0.067380167 |
| stump | 00014 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.171968840 | 0.238778075 | +0.066809235 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| stump | 00000 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.099657978 | 0.157507139 | +0.057849161 |
| treehill | 00015 | scene | fixed | learned | psnr_gain | 0.134147585 | 0.188424747 | +0.054277162 |
| room | 00011 | source_oracle_knn | source_trust | learned | psnr_gain | 1.266590215 | 1.320601400 | +0.054011185 |
| room | 00025 | source_reliability | source_trust | learned | psnr_gain | 0.363678456 | 0.415723932 | +0.052045476 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| stump | 00012 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.098184310 | 0.142417333 | +0.044233024 |
| garden | 00011 | knn | learned | source_trust | psnr_gain | 0.159749647 | 0.196581673 | +0.036832026 |
| room | 00028 | source_reliability | source_trust | learned | psnr_gain | 0.225513237 | 0.261882882 | +0.036369645 |
| room | 00017 | source_reliability | adaptive | source_trust | psnr_gain | 1.096731917 | 1.128950158 | +0.032218241 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00010 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.053699049 | 0.084335065 | +0.030636016 |
| bicycle | 00011 | scene | hybrid | fixed | psnr_gain | -0.035089251 | -0.007346744 | +0.027742508 |
| room | 00007 | source_reliability | learned | source_trust | psnr_gain | 0.237286355 | 0.263807359 | +0.026521005 |
| bicycle | 00016 | scene | hybrid | fixed | psnr_gain | -0.060029153 | -0.033677398 | +0.026351755 |

## v345e

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 9 | 246 | 0.276976720 | 0.284990320 | +0.008013599 | +0.006705092 | 95 |
| candidate_psnr | max | 9 | 246 | 25.417370947 | 25.425384546 | +0.008013599 | +0.006705092 | 95 |
| ssim_gain | max | 9 | 246 | 0.003768700 | 0.003857935 | +0.000089235 | +0.000084854 | 116 |
| candidate_ssim | max | 9 | 246 | 0.840522704 | 0.840611939 | +0.000089235 | +0.000084854 | 116 |
| mse_reduction | max | 9 | 246 | 0.000172621 | 0.000181887 | +0.000009266 | +0.000006935 | 95 |
| candidate_mse | min | 9 | 246 | 0.004433464 | 0.004424198 | +0.000009266 | +0.000006935 | 95 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| fixed_scene_generated_source_summary_unlock | 16 | +0.020859960 | +0.067380167 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |
| knn | 17 | +0.002166590 | +0.036832026 | 1 | learned:17 | learned:16, source_trust:1 |
| pairwise | 5 | +0.023580001 | +0.097477838 | 3 | fixed:2, hybrid:1, mix0250:1, source_trust:1 | fixed:2, learned:1, mix0750:1, source_trust:1 |
| scene | 114 | +0.006597255 | +0.084025668 | 45 | fixed:6, hybrid:37, learned:71 | fixed:13, hybrid:2, learned:94, mix0250:1, mix0750:3, source_trust:1 |
| source_oracle_knn | 8 | +0.010084050 | +0.054011185 | 3 | fixed:1, learned:4, source_trust:3 | fixed:1, learned:5, source_trust:2 |
| source_reliability | 83 | +0.003954236 | +0.052045476 | 28 | adaptive:3, fixed:4, hybrid:1, learned:51, mix0750:1, source_trust:23 | fixed:11, hybrid:1, learned:51, mix0750:2, source_trust:18 |
| target_neighbor_unlock | 3 | +0.000000000 | +0.000000000 | 0 | learned:3 | learned:3 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.582901932 | 0.583337117 | +0.000435186 | 3 | fixed:4, hybrid:1, learned:32 | fixed:6, learned:30, mix0750:1 |  |
| counter | ok | 30 | psnr_gain | 0.426359637 | 0.427055163 | +0.000695526 | 2 | learned:30 | fixed:1, learned:28, mix0750:1 |  |
| flowers | ok | 22 | psnr_gain | 0.092359632 | 0.097795440 | +0.005435808 | 13 | hybrid:13, learned:9 | fixed:1, learned:20, mix0750:1 |  |
| garden | ok | 24 | psnr_gain | 0.147204102 | 0.152575774 | +0.005371671 | 11 | fixed:1, hybrid:3, learned:12, mix0750:1, source_trust:7 | fixed:3, learned:14, source_trust:7 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.453250186 | 0.462676525 | +0.009426339 | 22 | adaptive:3, hybrid:2, learned:14, source_trust:20 | fixed:3, learned:19, mix0750:1, source_trust:16 |  |
| stump | ok | 16 | psnr_gain | 0.058909355 | 0.079769316 | +0.020859960 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118223929 | 0.133574687 | +0.015350757 | 6 | fixed:8, learned:9, mix0250:1 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.105065691 | 0.172445859 | +0.067380167 |
| stump | 00014 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.171968840 | 0.238778075 | +0.066809235 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| stump | 00000 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.099657978 | 0.157507139 | +0.057849161 |
| treehill | 00015 | scene | fixed | learned | psnr_gain | 0.134147585 | 0.188424747 | +0.054277162 |
| room | 00011 | source_oracle_knn | source_trust | learned | psnr_gain | 1.266590215 | 1.320601400 | +0.054011185 |
| room | 00025 | source_reliability | source_trust | learned | psnr_gain | 0.363678456 | 0.415723932 | +0.052045476 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| stump | 00012 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.098184310 | 0.142417333 | +0.044233024 |
| garden | 00011 | knn | learned | source_trust | psnr_gain | 0.159749647 | 0.196581673 | +0.036832026 |
| room | 00028 | source_reliability | source_trust | learned | psnr_gain | 0.225513237 | 0.261882882 | +0.036369645 |
| room | 00017 | source_reliability | adaptive | source_trust | psnr_gain | 1.096731917 | 1.128950158 | +0.032218241 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00010 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.053699049 | 0.084335065 | +0.030636016 |
| bicycle | 00011 | scene | hybrid | fixed | psnr_gain | -0.035089251 | -0.007346744 | +0.027742508 |
| room | 00007 | source_reliability | learned | source_trust | psnr_gain | 0.237286355 | 0.263807359 | +0.026521005 |
| bicycle | 00016 | scene | hybrid | fixed | psnr_gain | -0.060029153 | -0.033677398 | +0.026351755 |
