# v333 Target-Neighbor Consistency Full9 Audit

Compared against v329b fixed rollback strict full9 under the same scene/root protocol.

## Macro

| metric | v329b | v333 | delta |
|---|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | +0.000193920875 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | +0.000002247684 |
| rollback count | 0 | 2 | +2 |

## Per Scene

| scene | v329b PSNR gain | v333 PSNR gain | delta | v329b SSIM gain | v333 SSIM gain | delta | rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 0.119958548840 | 0.119958548840 | +0.000000000000 | 0.002988750935 | 0.002988750935 | +0.000000000000 | 0 |
| flowers | 0.092359631686 | 0.092359631686 | +0.000000000000 | 0.004114340652 | 0.004114340652 | +0.000000000000 | 0 |
| garden | 0.145449120172 | 0.145449120172 | +0.000000000000 | 0.001919778685 | 0.001919778685 | +0.000000000000 | 0 |
| stump | 0.057029761393 | 0.057029761393 | +0.000000000000 | 0.001208242029 | 0.001208242029 | +0.000000000000 | 0 |
| counter | 0.426359636889 | 0.426359636889 | +0.000000000000 | 0.006908347209 | 0.006908347209 | +0.000000000000 | 0 |
| treehill | 0.104664074413 | 0.106409362285 | +0.001745287872 | 0.001673645443 | 0.001693874598 | +0.000020229154 | 2 |
| bonsai | 0.575974442276 | 0.575974442276 | +0.000000000000 | 0.005847958294 | 0.005847958294 | +0.000000000000 | 0 |
| room | 0.437285496107 | 0.437285496107 | +0.000000000000 | 0.005058021117 | 0.005058021117 | +0.000000000000 | 0 |
| kitchen | 0.493623160533 | 0.493623160533 | +0.000000000000 | 0.003910861697 | 0.003910861697 | +0.000000000000 | 0 |

## Verdict

v333 improves full9 macro PSNR and SSIM over v329b in this replay, with applied target-neighbor rollbacks only on treehill. It is a candidate milestone, but still requires perceptual/qualitative refresh before being claimed as the final paper endpoint.
