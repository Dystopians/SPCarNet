# v339 TNC-Regularized Residual Focus6 Summary

## Macro

| method | PSNR gain | SSIM gain | tnc_reg active scenes | note |
|---|---:|---:|---|---|
| v337diag | 0.301231403771 | 0.003460387180 | {} | diagnostic |
| v339c_incomplete_stack | 0.297524611834 | 0.003441089505 | {'room': 39} | diagnostic |
| v339d_full_stack | 0.301231403771 | 0.003460387180 | {'room': 39} | fair full stack |

## v339d Full-Stack Per Scene

| scene | v339d PSNR | v339d SSIM | selected | generated suppression |
|---|---:|---:|---|---|
| stump | 0.057029761393 | 0.001208242029 | fixed | {'adaptive': 'source_summary_ssim_delta:-6.42120838e-05', 'tnc_reg': 'source_summary_ssim_delta:-5.85615635e-05'} |
| treehill | 0.118121382508 | 0.001717434989 | fixed | {'adaptive': 'source_summary_ssim_delta:-0.000138911334', 'tnc_reg': 'source_summary_ssim_delta:-0.000134403055'} |
| room | 0.442681127076 | 0.005089075137 | hybrid | {} |
| bicycle | 0.119958548840 | 0.002988750935 | hybrid | {'adaptive': 'source_summary_psnr_delta:-0.0152193543', 'tnc_reg': 'source_summary_psnr_delta:-0.0151268617'} |
| bonsai | 0.575974442276 | 0.005847958294 | learned | {'adaptive': 'source_summary_psnr_delta:-0.0454876448', 'tnc_reg': 'source_summary_psnr_delta:-0.0469213021'} |
| kitchen | 0.493623160533 | 0.003910861697 | learned | {'adaptive': 'source_summary_psnr_delta:-0.0761276079', 'tnc_reg': 'source_summary_psnr_delta:-0.0769644151'} |

## Room Probes

| probe | views | selected PSNR | tnc_reg PSNR | suppressed |
|---|---:|---:|---:|---|
| v339b_adaptive_base_smoke3 | 3 | 0.251561514642 | 0.2557358662587899 | {} |
| v339c_adaptive_base_margin_smoke3 | 3 | 0.251561514642 | 0.25617130369391933 | {} |
| v339e_learned_base_full_room | 39 | 0.442681127076 | n/a | {'tnc_reg': 'source_summary_psnr_delta:-0.0135018206'} |

## Verdict

v339 is a real generated-candidate implementation, but it does not improve the full-stack selected output on focus6. In the fair v339d run, macro PSNR/SSIM and oracle headroom are identical to v337diag. The only scene where `tnc_reg` is active is room, and there it is weaker than the already available adaptive/selected behavior.
