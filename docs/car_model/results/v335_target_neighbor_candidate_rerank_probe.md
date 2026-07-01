# Target-Neighbor Candidate Rerank Probe

Target GT is used only after target-blind candidate selection for analysis.

## Macro

| metric | current | fixed | learned | pure_tnc | oracle | pure_tnc-current | oracle-current |
|---|---:|---:|---:|---:|---:|---:|---:|
| psnr_gain | 0.272793021725 | 0.230035428440 | 0.274551449972 | 0.235473066023 | 0.283612355038 | -0.037319955702 | +0.010819333313 |
| ssim_gain | 0.003738933009 | 0.003414926490 | 0.003670204304 | 0.003419653533 | 0.003790476986 | -0.000319279475 | +0.000051543977 |

## Per Scene

| scene | current PSNR | pure_tnc PSNR | oracle PSNR | pure_tnc-current | oracle-current | TNC/GT match | pure_tnc best counts |
|---|---:|---:|---:|---:|---:|---:|---|
| bicycle | 0.119958548840 | 0.102932934299 | 0.132245880427 | -0.017025614542 | +0.012287331586 | 11/25 | {'fixed': 16, 'mix0750': 1, 'mix0250': 1, 'learned': 7} |
| flowers | 0.092359631686 | 0.078119901015 | 0.097795439781 | -0.014239730671 | +0.005435808095 | 1/22 | {'learned': 2, 'fixed': 19, 'mix0250': 1} |
| garden | 0.145449120172 | 0.134141931210 | 0.150329363496 | -0.011307188962 | +0.004880243324 | 6/24 | {'fixed': 13, 'hybrid': 1, 'learned': 10} |
| stump | 0.057029761393 | 0.058587761943 | 0.079749217953 | +0.001558000550 | +0.022719456560 | 7/16 | {'fixed': 11, 'learned': 2, 'hybrid': 3} |
| counter | 0.426359636889 | 0.346707669971 | 0.427055163312 | -0.079651966918 | +0.000695526423 | 4/30 | {'fixed': 26, 'learned': 3, 'mix0250': 1} |
| treehill | 0.107097397630 | 0.112655552646 | 0.133574686638 | +0.005558155016 | +0.026477289008 | 12/18 | {'learned': 8, 'fixed': 9, 'mix0250': 1} |
| bonsai | 0.575974442276 | 0.482979472629 | 0.583337117465 | -0.092994969647 | +0.007362675188 | 7/37 | {'fixed': 28, 'hybrid': 2, 'learned': 5, 'mix0250': 1, 'mix0750': 1} |
| room | 0.437285496107 | 0.396537426941 | 0.452541351963 | -0.040748069166 | +0.015255855856 | 7/39 | {'fixed': 32, 'learned': 4, 'hybrid': 1, 'mix0750': 2} |
| kitchen | 0.493623160533 | 0.406594943552 | 0.495882974308 | -0.087028216981 | +0.002259813775 | 4/35 | {'fixed': 26, 'learned': 6, 'hybrid': 1, 'mix0750': 1, 'mix0250': 1} |

## Verdict

pure_tnc is useful for measuring candidate-selection headroom. Promote it only if full9 macro improves over current without unacceptable scene regressions.
