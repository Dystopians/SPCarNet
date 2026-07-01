# Support-Transport Oracle Gap Audit

Read-only diagnostic over metrics already serialized in support_transport_apply_report.json.

## Macro

| method | scenes | views | primary | selected mean | oracle mean | mean headroom | positive views | selected counts | best counts |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| v337diag | 6 | 170 | psnr_gain | 0.301231404 | 0.313737956 | +0.012506552 | 63 | adaptive:14, fixed:25, hybrid:32, learned:95, mix0250:4 | adaptive:10, fixed:31, hybrid:3, learned:120, mix0250:1, mix0750:5 |
| v338b | 6 | 170 | psnr_gain | 0.301231404 | 0.313737956 | +0.012506552 | 63 | adaptive:14, fixed:25, hybrid:32, learned:95, mix0250:4 | adaptive:10, fixed:31, hybrid:3, learned:120, mix0250:1, mix0750:5 |

## v337diag

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.301231404 | 0.313737956 | +0.012506552 | +0.011080465 | 63 |
| candidate_psnr | max | 6 | 170 | 26.104138443 | 26.116644996 | +0.012506552 | +0.011080465 | 63 |
| ssim_gain | max | 6 | 170 | 0.003460387 | 0.003585768 | +0.000125381 | +0.000122215 | 82 |
| candidate_ssim | max | 6 | 170 | 0.846937592 | 0.847062973 | +0.000125381 | +0.000122215 | 82 |
| mse_reduction | max | 6 | 170 | 0.000167804 | 0.000180832 | +0.000013028 | +0.000009509 | 63 |
| candidate_mse | min | 6 | 170 | 0.003855144 | 0.003842116 | +0.000013028 | +0.000009509 | 63 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 104 | +0.014809074 | +0.256317118 | 45 | fixed:19, hybrid:31, learned:54 | adaptive:2, fixed:20, hybrid:2, learned:75, mix0250:1, mix0750:4 |
| source_reliability | 52 | +0.003528396 | +0.037579793 | 14 | adaptive:14, fixed:3, hybrid:1, learned:34 | adaptive:8, fixed:8, hybrid:1, learned:34, mix0750:1 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.442681127 | 0.457637859 | +0.014956732 | 21 | adaptive:14, hybrid:11, learned:14 | adaptive:10, fixed:4, learned:24, mix0750:1 |  |
| stump | ok | 16 | psnr_gain | 0.057029761 | 0.079749218 | +0.022719457 | 8 | fixed:16 | fixed:8, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118121383 | 0.133574687 | +0.015453304 | 7 | fixed:6, learned:8, mix0250:4 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| bonsai | 00035 | scene | learned | fixed | psnr_gain | 0.852690731 | 1.109007849 | +0.256317118 |
| room | 00011 | scene | hybrid | learned | psnr_gain | 1.198879503 | 1.320601400 | +0.121721897 |
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| room | 00023 | scene | hybrid | learned | psnr_gain | 0.511351439 | 0.592755082 | +0.081403643 |
| stump | 00014 | scene | fixed | learned | psnr_gain | 0.159542291 | 0.238778075 | +0.079235784 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | scene | fixed | learned | psnr_gain | 0.097256864 | 0.172445859 | +0.075188995 |
| stump | 00000 | scene | fixed | learned | psnr_gain | 0.090731259 | 0.157507139 | +0.066775880 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| room | 00012 | scene | hybrid | adaptive | psnr_gain | 0.881269117 | 0.942359562 | +0.061090446 |
| stump | 00012 | scene | fixed | learned | psnr_gain | 0.089913240 | 0.142417333 | +0.052504093 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| room | 00025 | scene | hybrid | learned | psnr_gain | 0.370359008 | 0.415723932 | +0.045364924 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00020 | source_reliability | adaptive | learned | psnr_gain | 0.427188690 | 0.464768483 | +0.037579793 |
| room | 00031 | source_reliability | adaptive | learned | psnr_gain | 0.405974683 | 0.443194750 | +0.037220067 |
| room | 00033 | scene | hybrid | fixed | psnr_gain | 0.231861210 | 0.267770773 | +0.035909563 |
| stump | 00010 | scene | fixed | learned | psnr_gain | 0.050573539 | 0.084335065 | +0.033761526 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |

## v338b

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 6 | 170 | 0.301231404 | 0.313737956 | +0.012506552 | +0.011080465 | 63 |
| candidate_psnr | max | 6 | 170 | 26.104138443 | 26.116644996 | +0.012506552 | +0.011080465 | 63 |
| ssim_gain | max | 6 | 170 | 0.003460387 | 0.003585768 | +0.000125381 | +0.000122215 | 82 |
| candidate_ssim | max | 6 | 170 | 0.846937592 | 0.847062973 | +0.000125381 | +0.000122215 | 82 |
| mse_reduction | max | 6 | 170 | 0.000167804 | 0.000180832 | +0.000013028 | +0.000009509 | 63 |
| candidate_mse | min | 6 | 170 | 0.003855144 | 0.003842116 | +0.000013028 | +0.000009509 | 63 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 5 | +0.000000000 | +0.000000000 | 0 | learned:5 | learned:5 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 104 | +0.014809074 | +0.256317118 | 45 | fixed:19, hybrid:31, learned:54 | adaptive:2, fixed:20, hybrid:2, learned:75, mix0250:1, mix0750:4 |
| source_reliability | 52 | +0.003528396 | +0.037579793 | 14 | adaptive:14, fixed:3, hybrid:1, learned:34 | adaptive:8, fixed:8, hybrid:1, learned:34, mix0750:1 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.442681127 | 0.457637859 | +0.014956732 | 21 | adaptive:14, hybrid:11, learned:14 | adaptive:10, fixed:4, learned:24, mix0750:1 |  |
| stump | ok | 16 | psnr_gain | 0.057029761 | 0.079749218 | +0.022719457 | 8 | fixed:16 | fixed:8, learned:7, mix0750:1 |  |
| treehill | ok | 18 | psnr_gain | 0.118121383 | 0.133574687 | +0.015453304 | 7 | fixed:6, learned:8, mix0250:4 | fixed:4, hybrid:1, learned:13 |  |

### Largest Misses

| scene | view | source | selected | best | metric | selected value | best value | headroom |
|---|---|---|---|---|---|---:|---:|---:|
| bonsai | 00035 | scene | learned | fixed | psnr_gain | 0.852690731 | 1.109007849 | +0.256317118 |
| room | 00011 | scene | hybrid | learned | psnr_gain | 1.198879503 | 1.320601400 | +0.121721897 |
| treehill | 00011 | pairwise | mix0250 | learned | psnr_gain | 0.328140877 | 0.425618716 | +0.097477838 |
| treehill | 00016 | scene | fixed | learned | psnr_gain | 0.105828445 | 0.189854114 | +0.084025668 |
| room | 00023 | scene | hybrid | learned | psnr_gain | 0.511351439 | 0.592755082 | +0.081403643 |
| stump | 00014 | scene | fixed | learned | psnr_gain | 0.159542291 | 0.238778075 | +0.079235784 |
| kitchen | 00018 | scene | learned | fixed | psnr_gain | 0.339133044 | 0.416241522 | +0.077108478 |
| stump | 00002 | scene | fixed | learned | psnr_gain | 0.097256864 | 0.172445859 | +0.075188995 |
| stump | 00000 | scene | fixed | learned | psnr_gain | 0.090731259 | 0.157507139 | +0.066775880 |
| bicycle | 00003 | scene | hybrid | learned | psnr_gain | 0.145244036 | 0.206562449 | +0.061318413 |
| room | 00012 | scene | hybrid | adaptive | psnr_gain | 0.881269117 | 0.942359562 | +0.061090446 |
| stump | 00012 | scene | fixed | learned | psnr_gain | 0.089913240 | 0.142417333 | +0.052504093 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| room | 00025 | scene | hybrid | learned | psnr_gain | 0.370359008 | 0.415723932 | +0.045364924 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00020 | source_reliability | adaptive | learned | psnr_gain | 0.427188690 | 0.464768483 | +0.037579793 |
| room | 00031 | source_reliability | adaptive | learned | psnr_gain | 0.405974683 | 0.443194750 | +0.037220067 |
| room | 00033 | scene | hybrid | fixed | psnr_gain | 0.231861210 | 0.267770773 | +0.035909563 |
| stump | 00010 | scene | fixed | learned | psnr_gain | 0.050573539 | 0.084335065 | +0.033761526 |
| treehill | 00017 | scene | fixed | learned | psnr_gain | 0.069650595 | 0.101037730 | +0.031387135 |
