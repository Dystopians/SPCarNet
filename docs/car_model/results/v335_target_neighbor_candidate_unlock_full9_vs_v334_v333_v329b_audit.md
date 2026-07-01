# v335 Target-Neighbor Candidate Unlock Full9 Audit

Compared against v334, v333, and v329b under the same full9 scene/root protocol.

## Macro

| metric | v329b | v333 | v334 | v335 | v335-v334 | v335-v333 | v335-v329b |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | 0.272793021725 | 0.274017908934 | +0.001224887209 | +0.001301335580 | +0.001495256455 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | 0.003738933009 | 0.003741526179 | +0.000002593170 | +0.000002617821 | +0.000004865505 |
| target-neighbor rollback count | 0 | 2 | 3 | 3 | +0 | +1 | +3 |
| candidate unlock count | 0 | 0 | 0 | 2 | +2 | +2 | +2 |
| all-axis safe scenes | 9/9 | 9/9 | 9/9 | 9/9 |  |  |  |

## Per Scene

| scene | v334 PSNR | v335 PSNR | delta | v334 SSIM | v335 SSIM | delta | v335 unlocks | v335 rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 0.119958548840 | 0.119958548840 | +0.000000000000 | 0.002988750935 | 0.002988750935 | +0.000000000000 | 0 | 0 |
| flowers | 0.092359631686 | 0.092359631686 | +0.000000000000 | 0.004114340652 | 0.004114340652 | +0.000000000000 | 0 | 0 |
| garden | 0.145449120172 | 0.145449120172 | +0.000000000000 | 0.001919778685 | 0.001919778685 | +0.000000000000 | 0 | 0 |
| stump | 0.057029761393 | 0.057029761393 | +0.000000000000 | 0.001208242029 | 0.001208242029 | +0.000000000000 | 0 | 0 |
| counter | 0.426359636889 | 0.426359636889 | +0.000000000000 | 0.006908347209 | 0.006908347209 | +0.000000000000 | 0 | 0 |
| treehill | 0.107097397630 | 0.118121382508 | +0.011023984878 | 0.001694096459 | 0.001717434989 | +0.000023338530 | 2 | 3 |
| bonsai | 0.575974442276 | 0.575974442276 | +0.000000000000 | 0.005847958294 | 0.005847958294 | +0.000000000000 | 0 | 0 |
| room | 0.437285496107 | 0.437285496107 | +0.000000000000 | 0.005058021117 | 0.005058021117 | +0.000000000000 | 0 | 0 |
| kitchen | 0.493623160533 | 0.493623160533 | +0.000000000000 | 0.003910861697 | 0.003910861697 | +0.000000000000 | 0 | 0 |

## Unlock Promotions

| scene | view | from | to | TNC margin | fixed PSNR gain | learned PSNR gain | selected PSNR gain | fixed SSIM gain | learned SSIM gain | selected SSIM gain |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| treehill | 00000 | fixed | learned | 0.000874153855 | 0.028486481194 | 0.046601732550 | 0.046601732550 | -0.000559389591 | -0.000394821167 | -0.000394821167 |
| treehill | 00010 | fixed | learned | 0.000670378121 | 0.446554320713 | 0.626870797166 | 0.626870797166 | 0.000600218773 | 0.000855743885 | 0.000855743885 |

## Verdict

v335 adds a target-blind fixed-to-learned unlock after v334 rollback. Full9 macro improves over v334/v333/v329b, with no changed scenes except treehill and 9/9 all-axis safe scenes.

Pure target-neighbor candidate ranking was rejected because it harmed full9 macro metrics. v335 is the guarded variant: it only unlocks `fixed -> learned` when learned is substantially more target-neighbor consistent than fixed under a global margin.
