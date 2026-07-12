# Edit-aware ECR eval — toy_parking (delete_faces_box, 711609 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 27.530 | 0.0000 | 31.060 |
| C2_stale | 30.895 | 0.0030 | 32.048 |
| C4_rebuild | 26.597 | 0.0221 | 31.170 |
| C5_ours | 30.859 | 0.0030 | 32.047 |
| ORIG_ecr | nan | nan | 32.049 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +3.365 [+0.212,+8.158]
- ghost_C5_minus_C1: +3.329 [+0.197,+8.144]
- presU_C5_minus_ORIG: -0.002 [-0.004,+0.000]
- ghost_C4_minus_C1: -0.933 [-5.343,+2.114]

Update cost: {"mask_pass_seconds": 15.1194486618042, "local_rebuild_seconds": 18.435994625091553, "bytes_rewritten_renders_depths": 231318383, "bytes_masks": 410336, "n_affected_views": 72, "n_train_views": 72}
