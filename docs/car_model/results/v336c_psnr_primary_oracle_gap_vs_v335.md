# Support-Transport Oracle Gap Audit

Read-only diagnostic over metrics already serialized in support_transport_apply_report.json.

## Macro

| method | scenes | views | primary | selected mean | oracle mean | mean headroom | positive views | selected counts | best counts |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| v335 | 9 | 246 | psnr_gain | 0.274017909 | 0.283612355 | +0.009594446 | 91 | fixed:27, hybrid:56, learned:150, mix0250:8, mix0750:5 | fixed:41, hybrid:7, learned:183, mix0250:4, mix0750:11 |
| v336c | 9 | 246 | psnr_gain | 0.274617423 | 0.284178634 | +0.009561210 | 88 | adaptive:14, fixed:26, hybrid:48, learned:149, mix0250:7, mix0750:2 | adaptive:10, fixed:37, hybrid:5, learned:182, mix0250:3, mix0750:9 |

## v335

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 9 | 246 | 0.274017909 | 0.283612355 | +0.009594446 | +0.008751725 | 91 |
| candidate_psnr | max | 9 | 246 | 25.414412135 | 25.424006581 | +0.009594446 | +0.008751725 | 91 |
| ssim_gain | max | 9 | 246 | 0.003741526 | 0.003847394 | +0.000105868 | +0.000105699 | 122 |
| candidate_ssim | max | 9 | 246 | 0.840495531 | 0.840601399 | +0.000105868 | +0.000105699 | 122 |
| mse_reduction | max | 9 | 246 | 0.000171188 | 0.000181535 | +0.000010347 | +0.000007972 | 91 |
| candidate_mse | min | 9 | 246 | 0.004434897 | 0.004424550 | +0.000010347 | +0.000007972 | 91 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 20 | +0.003197549 | +0.030162277 | 4 | learned:20 | fixed:2, learned:16, mix0250:1, mix0750:1 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 145 | +0.012370320 | +0.256317118 | 67 | fixed:19, hybrid:54, learned:72 | fixed:25, hybrid:4, learned:109, mix0250:1, mix0750:6 |
| source_reliability | 72 | +0.001878031 | +0.035532993 | 16 | fixed:5, hybrid:2, learned:56, mix0250:4, mix0750:5 | fixed:11, hybrid:3, learned:52, mix0250:2, mix0750:4 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| counter | ok | 30 | psnr_gain | 0.426359637 | 0.427055163 | +0.000695526 | 2 | learned:30 | fixed:1, learned:28, mix0750:1 |  |
| flowers | ok | 22 | psnr_gain | 0.092359632 | 0.097795440 | +0.005435808 | 13 | hybrid:13, learned:9 | fixed:1, learned:20, mix0750:1 |  |
| garden | ok | 24 | psnr_gain | 0.145449120 | 0.150329363 | +0.004880243 | 10 | fixed:1, hybrid:3, learned:15, mix0250:3, mix0750:2 | fixed:4, hybrid:2, learned:14, mix0250:2, mix0750:2 |  |
| kitchen | ok | 35 | psnr_gain | 0.493623161 | 0.495882974 | +0.002259814 | 4 | learned:35 | fixed:1, hybrid:1, learned:31, mix0750:2 |  |
| room | ok | 39 | psnr_gain | 0.437285496 | 0.452541352 | +0.015255856 | 24 | fixed:1, hybrid:19, learned:15, mix0250:1, mix0750:3 | fixed:8, hybrid:2, learned:25, mix0250:1, mix0750:3 |  |
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
| stump | 00012 | scene | fixed | learned | psnr_gain | 0.089913240 | 0.142417333 | +0.052504093 |
| bicycle | 00004 | scene | hybrid | learned | psnr_gain | 0.204494401 | 0.252831422 | +0.048337021 |
| treehill | 00015 | pairwise | mix0250 | learned | psnr_gain | 0.149224455 | 0.188424747 | +0.039200292 |
| room | 00017 | scene | hybrid | learned | psnr_gain | 1.082019139 | 1.120058190 | +0.038039051 |
| room | 00031 | scene | hybrid | learned | psnr_gain | 0.405565731 | 0.443194750 | +0.037629019 |
| room | 00033 | scene | hybrid | fixed | psnr_gain | 0.231861210 | 0.267770773 | +0.035909563 |
| room | 00030 | source_reliability | mix0750 | fixed | psnr_gain | 0.185204952 | 0.220737946 | +0.035532993 |
| room | 00009 | scene | hybrid | fixed | psnr_gain | 0.932064548 | 0.965885836 | +0.033821288 |
| stump | 00010 | scene | fixed | learned | psnr_gain | 0.050573539 | 0.084335065 | +0.033761526 |
| room | 00013 | source_reliability | mix0250 | learned | psnr_gain | 0.388008179 | 0.421466756 | +0.033458577 |

## v336c

### Metric Headroom

| metric | dir | scenes | views | selected macro | oracle macro | headroom macro | headroom micro | positive views |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | max | 9 | 246 | 0.274617423 | 0.284178634 | +0.009561210 | +0.008704303 | 88 |
| candidate_psnr | max | 9 | 246 | 25.415011650 | 25.424572860 | +0.009561210 | +0.008704303 | 88 |
| ssim_gain | max | 9 | 246 | 0.003744977 | 0.003850438 | +0.000105461 | +0.000105118 | 118 |
| candidate_ssim | max | 9 | 246 | 0.840498981 | 0.840604442 | +0.000105461 | +0.000105118 | 118 |
| mse_reduction | max | 9 | 246 | 0.000171340 | 0.000181641 | +0.000010301 | +0.000007907 | 88 |
| candidate_mse | min | 9 | 246 | 0.004434745 | 0.004424444 | +0.000010301 | +0.000007907 | 88 |

### Decision Sources

| source | views | mean headroom | max headroom | positive views | output counts | best counts |
|---|---:|---:|---:|---:|---|---|
| knn | 20 | +0.003197549 | +0.030162277 | 4 | learned:20 | fixed:2, learned:16, mix0250:1, mix0750:1 |
| pairwise | 7 | +0.022865537 | +0.097477838 | 4 | fixed:3, mix0250:4 | fixed:3, learned:4 |
| scene | 138 | +0.012380686 | +0.256317118 | 61 | fixed:19, hybrid:47, learned:72 | adaptive:2, fixed:22, hybrid:3, learned:105, mix0250:1, mix0750:5 |
| source_reliability | 79 | +0.002641951 | +0.037579793 | 19 | adaptive:14, fixed:4, hybrid:1, learned:55, mix0250:3, mix0750:2 | adaptive:8, fixed:10, hybrid:2, learned:55, mix0250:1, mix0750:3 |
| target_neighbor_unlock | 2 | +0.000000000 | +0.000000000 | 0 | learned:2 | learned:2 |

### Per Scene

| scene | status | views | primary | selected mean | oracle mean | mean headroom | positive views | output counts | best counts | warnings |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| bicycle | ok | 25 | psnr_gain | 0.119958549 | 0.132245880 | +0.012287332 | 19 | hybrid:20, learned:5 | fixed:8, hybrid:1, learned:15, mix0250:1 |  |
| bonsai | ok | 37 | psnr_gain | 0.575974442 | 0.583337117 | +0.007362675 | 4 | fixed:3, hybrid:1, learned:33 | fixed:6, learned:30, mix0750:1 |  |
| counter | ok | 30 | psnr_gain | 0.426359637 | 0.427055163 | +0.000695526 | 2 | learned:30 | fixed:1, learned:28, mix0750:1 |  |
| flowers | ok | 22 | psnr_gain | 0.092359632 | 0.097795440 | +0.005435808 | 13 | hybrid:13, learned:9 | fixed:1, learned:20, mix0750:1 |  |
| garden | ok | 24 | psnr_gain | 0.145449120 | 0.150329363 | +0.004880243 | 10 | fixed:1, hybrid:3, learned:15, mix0250:3, mix0750:2 | fixed:4, hybrid:2, learned:14, mix0250:2, mix0750:2 |  |
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
