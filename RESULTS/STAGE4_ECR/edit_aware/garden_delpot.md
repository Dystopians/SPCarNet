# Edit-aware ECR eval — garden (delete_faces_cylinder, 24952 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 68.609 | 0.0000 | 24.768 |
| C2_stale | 68.672 | 0.0042 | 26.354 |
| C5_ours | 68.648 | 0.0037 | 26.354 |
| ORIG_ecr | nan | nan | 26.354 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +0.063 [+0.028,+0.106]
- ghost_C5_minus_C1: +0.039 [+0.016,+0.069]
- presU_C5_minus_ORIG: -0.000 [-0.000,-0.000]

Update cost: {"mask_pass_seconds": 40.061193227767944, "local_rebuild_seconds": 41.67200446128845, "bytes_rewritten_renders_depths": 368650936, "bytes_masks": 348435, "n_affected_views": 57, "n_train_views": 161}
