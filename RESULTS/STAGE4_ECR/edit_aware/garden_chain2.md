# Edit-aware ECR eval — garden (recolor_faces_cylinder, 22724 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 68.979 | 0.0000 | 18.506 |
| C2_stale | 71.648 | 0.0207 | 18.791 |
| C5_ours | 69.022 | 0.0029 | 18.791 |
| ORIG_ecr | nan | nan | 18.791 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +2.669 [+1.326,+4.085]
- ghost_C5_minus_C1: +0.043 [+0.021,+0.069]
- presU_C5_minus_ORIG: -0.000 [-0.000,-0.000]

Update cost: {"mask_pass_seconds": 47.122085094451904, "local_rebuild_seconds": 74.17058634757996, "bytes_rewritten_renders_depths": 1051494674, "bytes_masks": 1883097, "n_affected_views": 161, "n_train_views": 161}
