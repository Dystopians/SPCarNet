# Final Stage F28 - Bonsai QEM Sparse-Depth Recovery Report

Decision: `FINAL_F28_BONSAI_QEM_SPARSE_DEPTH_PARETO_PASS`.

## Goal

Run a final compact-recovery row that explicitly enables sparse COLMAP depth supervision, instead of only using sparse COLMAP geometry for independent evaluation. This addresses the F11 gap that the final compact rows did not yet contain an explicit sparse-depth-guided recovery variant.

## Run

| field | value |
| --- | --- |
| source compact checkpoint | `outputs/carnet/meshsplatopt/final_stageF22_bonsai_posthoc_qem_baseline/prune50/compact_model` |
| recovery checkpoint | `outputs/carnet/meshsplatopt/final_stageF28_bonsai_qem_sparse_depth/prune50/recovery_model` |
| schedule | `22000->26000` |
| W&B | `07k1ii1d` |
| sparse loss | `--enable_sparse_colmap_depth_loss --lambda_sparse_colmap_depth 0.001` |
| schedule details | start `22000`, warmup `500`, decay `24000->26000`, final multiplier `0.2` |
| sampling | `low_error`, fraction `0.5`, min matches `24` |
| topology freeze | enabled |

The training log confirmed `[SparseCOLMAPDepth] enabled: context initialized.`

## Independent Result

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| QEM50 frozen 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |
| QEM50 + sparse-depth 26k | 44,230 | 11.081614 | 0.243248 | 0.569658 | 0.181698 | 1.779783 | 42.425734 |

## Interpretation

The sparse-depth row is not a PSNR headline over the original QEM50: it gives back only `0.000791 dB` PSNR and `0.000001` SSIM. It is a stronger geometry/perceptual Pareto point, improving LPIPS by `0.000519`, AbsRel by `0.001268`, Depth MAE by `0.014069`, and normal angle by `0.463605` degrees at identical topology.

This is the first final compact-recovery row that explicitly enables sparse COLMAP depth loss and improves the recovered compact representation on the sparse geometry proxies without a meaningful render-quality cost.

