# Edit-aware ECR eval — garden (delete_faces_cylinder, 2037550 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 14.230 | 0.0000 | 24.384 |
| C2_stale | 14.314 | 0.0088 | 26.041 |
| C4_rebuild | 17.318 | 0.0840 | 25.799 |
| C5_ours | 14.288 | 0.0065 | 26.030 |
| ORIG_ecr | nan | nan | 26.050 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +0.085 [+0.065,+0.105]
- ghost_C5_minus_C1: +0.059 [+0.038,+0.081]
- presU_C5_minus_ORIG: -0.020 [-0.034,-0.008]
- ghost_C4_minus_C1: +3.088 [+2.568,+3.616]

Update cost: {"mask_pass_seconds": 37.295748472213745, "local_rebuild_seconds": 70.86895251274109, "bytes_rewritten_renders_depths": 1051515568, "bytes_masks": 1824897, "n_affected_views": 161, "n_train_views": 161}
