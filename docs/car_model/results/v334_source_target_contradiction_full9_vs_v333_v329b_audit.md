# v334 Source-Target Contradiction Full9 Audit

Compared against v333 target-neighbor consistency and v329b fixed rollback strict full9 under the same scene/root protocol.

## Macro

| metric | v329b | v333 | v334 | v334-v333 | v334-v329b |
|---|---:|---:|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | 0.272793021725 | +0.000076448372 | +0.000270369246 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | 0.003738933009 | +0.000000024651 | +0.000002272335 |
| rollback count | 0 | 2 | 3 | +1 | +3 |
| all-axis safe scenes | 9/9 | 9/9 | 9/9 |  |  |

## Per Scene

| scene | v333 PSNR gain | v334 PSNR gain | delta | v333 SSIM gain | v334 SSIM gain | delta | v334 rollbacks | reason counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | 0.119958548840 | 0.119958548840 | +0.000000000000 | 0.002988750935 | 0.002988750935 | +0.000000000000 | 0 | no_promotion:25 |
| flowers | 0.092359631686 | 0.092359631686 | +0.000000000000 | 0.004114340652 | 0.004114340652 | +0.000000000000 | 0 | decision_source_not_checked:9, no_promotion:13 |
| garden | 0.145449120172 | 0.145449120172 | +0.000000000000 | 0.001919778685 | 0.001919778685 | +0.000000000000 | 0 | decision_source_not_checked:21, no_promotion:3 |
| stump | 0.057029761393 | 0.057029761393 | +0.000000000000 | 0.001208242029 | 0.001208242029 | +0.000000000000 | 0 | no_promotion:16 |
| counter | 0.426359636889 | 0.426359636889 | +0.000000000000 | 0.006908347209 | 0.006908347209 | +0.000000000000 | 0 | no_promotion:30 |
| treehill | 0.106409362285 | 0.107097397630 | +0.000688035345 | 0.001693874598 | 0.001694096459 | +0.000000221862 | 3 | no_promotion:11, passed:4, source_target_neighbor_contradiction:1, target_neighbor_consistency_delta:2 |
| bonsai | 0.575974442276 | 0.575974442276 | +0.000000000000 | 0.005847958294 | 0.005847958294 | +0.000000000000 | 0 | decision_source_not_checked:4, no_promotion:33 |
| room | 0.437285496107 | 0.437285496107 | +0.000000000000 | 0.005058021117 | 0.005058021117 | +0.000000000000 | 0 | decision_source_not_checked:20, no_promotion:19 |
| kitchen | 0.493623160533 | 0.493623160533 | +0.000000000000 | 0.003910861697 | 0.003910861697 | +0.000000000000 | 0 | no_promotion:35 |

## Treehill Critical Views

| view | raw variant | final variant | rollback | reason | target-neighbor delta | source psnr_min | source psnr_cvar | source pos frac | raw-vs-fixed PSNR delta | raw-vs-fixed SSIM delta |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 00007 | mix0250 | fixed | true | target_neighbor_consistency_delta | -0.000121739321 | 0.021059423675 | 0.021059423675 | 1.000000000000 | -0.026469408413 | -0.000042915344 |
| 00008 | mix0250 | fixed | true | target_neighbor_consistency_delta | -0.000148012727 | 0.000413699791 | 0.000413699791 | 1.000000000000 | -0.004945773279 | -0.000321209431 |
| 00009 | mix0250 | fixed | true | source_target_neighbor_contradiction | -0.000024989738 | 0.021059423675 | 0.021059423675 | 1.000000000000 | -0.012384636218 | -0.000003993511 |
| 00011 | mix0250 | mix0250 | false | passed | -0.000070053628 | 0.000413699791 | 0.000413699791 | 1.000000000000 | 0.033720386903 | 0.000268816948 |
| 00015 | mix0250 | mix0250 | false | passed | -0.000044425695 | 0.000413699791 | 0.000413699791 | 1.000000000000 | 0.015076869936 | 0.000209391117 |

## Verdict

v334 is a real full9-safe tail-repair increment over v333/v329b under the frozen policy. The gain is narrow and comes from one additional source-target contradiction rollback on treehill 00009, so it is a milestone certificate, not a broad representation-level breakthrough.

This is useful because it converts a previously missed failure mode into a target-blind certificate: source-local evidence is very confident, but target-neighbor self-consistency says the promoted candidate is slightly worse than the incumbent. The limitation is that the full9 gain remains small and concentrated in treehill, so v334 should be reported as a reliability refinement rather than final paper-level closure.
