# Support-Transport Oracle Gap Audit

Read-only diagnostic over metrics already serialized in support_transport_apply_report.json.

## Macro

| method | scenes | views | primary | selected mean | oracle mean | mean headroom | positive views | selected counts | best counts |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| v340d | 6 | 170 | psnr_gain | 0.301510278 | 0.314015968 | +0.012505689 | 65 | adaptive:19, fixed:23, hybrid:30, learned:94, mix0250:4 | adaptive:19, fixed:26, hybrid:1, learned:120, mix0250:1, mix0750:3 |
| v341_source_trust | 6 | 170 | psnr_gain | 0.302505694 | 0.314581083 | +0.012075390 | 68 | adaptive:3, fixed:25, hybrid:28, learned:92, mix0250:4, source_trust:18 | adaptive:1, fixed:29, hybrid:3, learned:115, mix0250:1, mix0750:5, source_trust:16 |
| v342e_fixed_generated_unlock_freeze | 6 | 170 | psnr_gain | 0.302818959 | 0.314581083 | +0.011762124 | 74 | adaptive:19, fixed:9, hybrid:28, learned:92, mix0250:4, source_trust:18 | adaptive:1, fixed:29, hybrid:3, learned:115, mix0250:1, mix0750:5, source_trust:16 |
| v343e_source_oracle_fixed_override | 6 | 170 | psnr_gain | 0.304165626 | 0.314581083 | +0.010415457 | 69 | adaptive:19, fixed:13, hybrid:23, learned:95, source_trust:20 | adaptive:1, fixed:29, hybrid:3, learned:115, mix0250:1, mix0750:5, source_trust:16 |

## v340d

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.301510278 | 0.314015968 | +0.012505689 | +0.010999762 | 65 |
| candidate_psnr | max | 6 | 170 | 26.104417318 | 26.116923007 | +0.012505689 | +0.010999762 | 65 |
| ssim_gain | max | 6 | 170 | 0.003461710 | 0.003595458 | +0.000133749 | +0.000131897 | 81 |
| candidate_ssim | max | 6 | 170 | 0.846938915 | 0.847072663 | +0.000133749 | +0.000131897 | 81 |
| mse_reduction | max | 6 | 170 | 0.000167842 | 0.000180956 | +0.000013114 | +0.000009575 | 65 |
| candidate_mse | min | 6 | 170 | 0.003855106 | 0.003841992 | +0.000013114 | +0.000009575 | 65 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 97 | +0.011514909 | +0.121721897 | 40 | fixed:19, hybrid:26, learned:52 | adaptive:6, fixed:16, learned:71, mix0250:1, mix0750:3 |
| source_oracle_knn | 7 | +0.055380617 | +0.256317118 | 6 | adaptive:2, hybrid:4, learned:1 | adaptive:2, fixed:1, learned:4 |
| source_reliability | 52 | +0.003947892 | +0.037579793 | 15 | adaptive:17, fixed:1, learned:34 | adaptive:11, fixed:6, hybrid:1, learned:34 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.133091322 | +0.013132773 | 20 | hybrid:20, learned:5 | adaptive:4, fixed:5, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.576081269 | 0.584034885 | +0.007953616 | 6 | adaptive:4, fixed:1, learned:32 | adaptive:3, fixed:4, learned:30 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.496007835 | +0.002384674 | 4 | learned:35 | adaptive:2, fixed:1, learned:31, mix0750:1 |  |
| room | ok | 39 | psnr_gain | 0.444247549 | 0.457637859 | +0.013390311 | 20 | adaptive:15, hybrid:10, learned:14 | adaptive:10, fixed:4, learned:24, mix0750:1 |  |
| stump | ok | 16 | psnr_gain | 0.057029761 | 0.079749218 | +0.022719457 | 8 | fixed:16 | fixed:8, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118121383 | 0.133574687 | +0.015453304 | 7 | fixed:6, learned:8, mix0250:4 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| bonsai | 00035 | source_oracle_knn | learned | fixed | psnr_gain | 0.852690731 | 1.109007849 | +0.256317118 |
| room | 00011 | scene | hybrid | learned | psnr_gain | 1.198879503 | 1.320601400 | +0.121721897 |
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| room | 00023 | source_oracle_knn | hybrid | learned | psnr_gain | 0.511351439 | 0.592755082 | +0.081403643 |
| stump | 00014 | scene | fixed | learned | psnr_gain | 0.159542291 | 0.238778075 | +0.079235784 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | scene | fixed | learned | psnr_gain | 0.097256864 | 0.172445859 | +0.075188995 |
| stump | 00000 | scene | fixed | learned | psnr_gain | 0.090731259 | 0.157507139 | +0.066775880 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| stump | 00012 | scene | fixed | learned | psnr_gain | 0.089913240 | 0.142417333 | +0.052504093 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| room | 00025 | scene | hybrid | learned | psnr_gain | 0.370359008 | 0.415723932 | +0.045364924 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00020 | source_reliability | adaptive | learned | psnr_gain | 0.427188690 | 0.464768483 | +0.037579793 |
| room | 00031 | source_reliability | adaptive | learned | psnr_gain | 0.405974683 | 0.443194750 | +0.037220067 |
| room | 00033 | scene | hybrid | fixed | psnr_gain | 0.231861210 | 0.267770773 | +0.035909563 |
| stump | 00010 | scene | fixed | learned | psnr_gain | 0.050573539 | 0.084335065 | +0.033761526 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00001 | scene | fixed | learned | psnr_gain | 0.061309808 | 0.091751286 | +0.030441478 |

## v341_source_trust

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.302505694 | 0.314581083 | +0.012075390 | +0.010484263 | 68 |
| candidate_psnr | max | 6 | 170 | 26.105412733 | 26.117488123 | +0.012075390 | +0.010484263 | 68 |
| ssim_gain | max | 6 | 170 | 0.003469209 | 0.003593406 | +0.000124197 | +0.000119824 | 80 |
| candidate_ssim | max | 6 | 170 | 0.846946414 | 0.847070611 | +0.000124197 | +0.000119824 | 80 |
| mse_reduction | max | 6 | 170 | 0.000168037 | 0.000181003 | +0.000012966 | +0.000009421 | 68 |
| candidate_mse | min | 6 | 170 | 0.003854911 | 0.003841945 | +0.000012966 | +0.000009421 | 68 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 94 | +0.009403587 | +0.084025668 | 36 | fixed:19, hybrid:22, learned:53 | adaptive:1, fixed:17, hybrid:2, learned:69, mix0250:1, mix0750:4 |
| source_oracle_knn | 8 | +0.055960645 | +0.256317118 | 7 | hybrid:5, learned:1, source_trust:2 | fixed:1, learned:5, source_trust:2 |
| source_reliability | 54 | +0.005382290 | +0.052045476 | 21 | adaptive:3, fixed:3, hybrid:1, learned:31, source_trust:16 | fixed:8, hybrid:1, learned:30, mix0750:1, source_trust:14 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.450326866 | 0.462676525 | +0.012349659 | 25 | adaptive:3, hybrid:7, learned:11, source_trust:18 | fixed:3, learned:19, mix0750:1, source_trust:16 |  |
| stump | ok | 16 | psnr_gain | 0.057029761 | 0.079769316 | +0.022739554 | 9 | fixed:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118121383 | 0.133574687 | +0.015453304 | 7 | fixed:6, learned:8, mix0250:4 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| bonsai | 00035 | source_oracle_knn | learned | fixed | psnr_gain | 0.852690731 | 1.109007849 | +0.256317118 |
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| room | 00023 | source_oracle_knn | hybrid | learned | psnr_gain | 0.511351439 | 0.592755082 | +0.081403643 |
| stump | 00014 | scene | fixed | learned | psnr_gain | 0.159542291 | 0.238778075 | +0.079235784 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | scene | fixed | learned | psnr_gain | 0.097256864 | 0.172445859 | +0.075188995 |
| stump | 00000 | scene | fixed | learned | psnr_gain | 0.090731259 | 0.157507139 | +0.066775880 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| room | 00011 | source_oracle_knn | source_trust | learned | psnr_gain | 1.266590215 | 1.320601400 | +0.054011185 |
| stump | 00012 | scene | fixed | learned | psnr_gain | 0.089913240 | 0.142417333 | +0.052504093 |
| room | 00025 | source_reliability | source_trust | learned | psnr_gain | 0.363678456 | 0.415723932 | +0.052045476 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00028 | source_reliability | source_trust | learned | psnr_gain | 0.225513237 | 0.261882882 | +0.036369645 |
| stump | 00010 | scene | fixed | learned | psnr_gain | 0.050573539 | 0.084335065 | +0.033761526 |
| room | 00017 | source_reliability | adaptive | source_trust | psnr_gain | 1.096731917 | 1.128950158 | +0.032218241 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00001 | scene | fixed | learned | psnr_gain | 0.061309808 | 0.091751286 | +0.030441478 |
| bicycle | 00011 | scene | hybrid | fixed | psnr_gain | -0.035089251 | -0.007346744 | +0.027742508 |

## v342e_fixed_generated_unlock_freeze

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.302818959 | 0.314581083 | +0.011762124 | +0.010307360 | 74 |
| candidate_psnr | max | 6 | 170 | 26.105725999 | 26.117488123 | +0.011762124 | +0.010307360 | 74 |
| ssim_gain | max | 6 | 170 | 0.003471775 | 0.003593406 | +0.000121632 | +0.000118375 | 85 |
| candidate_ssim | max | 6 | 170 | 0.846948980 | 0.847070611 | +0.000121632 | +0.000118375 | 85 |
| mse_reduction | max | 6 | 170 | 0.000168453 | 0.000181003 | +0.000012550 | +0.000009186 | 74 |
| candidate_mse | min | 6 | 170 | 0.003854495 | 0.003841945 | +0.000012550 | +0.000009186 | 74 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| fixed_scene_generated_source_summary_unlock | 16 | +0.020859960 | +0.067380167 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 78 | +0.006668004 | +0.084025668 | 27 | fixed:3, hybrid:22, learned:53 | fixed:10, hybrid:2, learned:62, mix0250:1, mix0750:3 |
| source_oracle_knn | 8 | +0.055960645 | +0.256317118 | 7 | hybrid:5, learned:1, source_trust:2 | fixed:1, learned:5, source_trust:2 |
| source_reliability | 54 | +0.005382290 | +0.052045476 | 21 | adaptive:3, fixed:3, hybrid:1, learned:31, source_trust:16 | fixed:8, hybrid:1, learned:30, mix0750:1, source_trust:14 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.450326866 | 0.462676525 | +0.012349659 | 25 | adaptive:3, hybrid:7, learned:11, source_trust:18 | fixed:3, learned:19, mix0750:1, source_trust:16 |  |
| stump | ok | 16 | psnr_gain | 0.058909355 | 0.079769316 | +0.020859960 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118121383 | 0.133574687 | +0.015453304 | 7 | fixed:6, learned:8, mix0250:4 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| bonsai | 00035 | source_oracle_knn | learned | fixed | psnr_gain | 0.852690731 | 1.109007849 | +0.256317118 |
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| room | 00023 | source_oracle_knn | hybrid | learned | psnr_gain | 0.511351439 | 0.592755082 | +0.081403643 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.105065691 | 0.172445859 | +0.067380167 |
| stump | 00014 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.171968840 | 0.238778075 | +0.066809235 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| stump | 00000 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.099657978 | 0.157507139 | +0.057849161 |
| room | 00011 | source_oracle_knn | source_trust | learned | psnr_gain | 1.266590215 | 1.320601400 | +0.054011185 |
| room | 00025 | source_reliability | source_trust | learned | psnr_gain | 0.363678456 | 0.415723932 | +0.052045476 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| stump | 00012 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.098184310 | 0.142417333 | +0.044233024 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00028 | source_reliability | source_trust | learned | psnr_gain | 0.225513237 | 0.261882882 | +0.036369645 |
| room | 00017 | source_reliability | adaptive | source_trust | psnr_gain | 1.096731917 | 1.128950158 | +0.032218241 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00010 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.053699049 | 0.084335065 | +0.030636016 |
| bicycle | 00011 | scene | hybrid | fixed | psnr_gain | -0.035089251 | -0.007346744 | +0.027742508 |
| room | 00007 | source_reliability | learned | source_trust | psnr_gain | 0.237286355 | 0.263807359 | +0.026521005 |

## v343e_source_oracle_fixed_override

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.304165626 | 0.314581083 | +0.010415457 | +0.008316466 | 69 |
| candidate_psnr | max | 6 | 170 | 26.107072666 | 26.117488123 | +0.010415457 | +0.008316466 | 69 |
| ssim_gain | max | 6 | 170 | 0.003493470 | 0.003593406 | +0.000099936 | +0.000091428 | 78 |
| candidate_ssim | max | 6 | 170 | 0.846970675 | 0.847070611 | +0.000099936 | +0.000091428 | 78 |
| mse_reduction | max | 6 | 170 | 0.000168730 | 0.000181003 | +0.000012273 | +0.000008603 | 69 |
| candidate_mse | min | 6 | 170 | 0.003854218 | 0.003841945 | +0.000012273 | +0.000008603 | 69 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| fixed_scene_generated_source_summary_unlock | 16 | +0.020859960 | +0.067380167 | 15 | adaptive:16 | adaptive:1, fixed:7, learned:7, mix0750:1 |
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 2 | +0.010211082 | +0.017349122 | 2 | hybrid:1, source_trust:1 | mix0750:1, source_trust:1 |
| scene | 83 | +0.008577638 | +0.131198225 | 29 | fixed:9, hybrid:21, learned:53 | fixed:13, hybrid:2, learned:65, mix0250:1, mix0750:2 |
| source_oracle_knn | 7 | +0.008147155 | +0.054011185 | 2 | fixed:1, learned:3, source_trust:3 | fixed:1, learned:5, source_trust:1 |
| source_reliability | 54 | +0.005382290 | +0.052045476 | 21 | adaptive:3, fixed:3, hybrid:1, learned:31, source_trust:16 | fixed:8, hybrid:1, learned:30, mix0750:1, source_trust:14 |
| target_neighbor_unlock | 3 | +0.000000000 | +0.000000000 | 0 | learned:3 | learned:3 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.582901932 | 0.583337117 | +0.000435186 | 3 | fixed:4, hybrid:1, learned:32 | fixed:6, learned:30, mix0750:1 |  |
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
| room | 00028 | source_reliability | source_trust | learned | psnr_gain | 0.225513237 | 0.261882882 | +0.036369645 |
| room | 00017 | source_reliability | adaptive | source_trust | psnr_gain | 1.096731917 | 1.128950158 | +0.032218241 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
| stump | 00010 | fixed_scene_generated_source_summary_unlock | adaptive | learned | psnr_gain | 0.053699049 | 0.084335065 | +0.030636016 |
| bicycle | 00011 | scene | hybrid | fixed | psnr_gain | -0.035089251 | -0.007346744 | +0.027742508 |
| room | 00007 | source_reliability | learned | source_trust | psnr_gain | 0.237286355 | 0.263807359 | +0.026521005 |
| bicycle | 00016 | scene | hybrid | fixed | psnr_gain | -0.060029153 | -0.033677398 | +0.026351755 |
| room | 00038 | source_reliability | learned | source_trust | psnr_gain | 0.376399700 | 0.401554072 | +0.025154372 |
