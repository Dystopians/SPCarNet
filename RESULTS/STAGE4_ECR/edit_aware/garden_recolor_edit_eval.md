# Edit-aware ECR eval — garden (recolor_faces_cylinder, 2037550 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 14.461 | 0.0000 | 24.365 |
| C2_stale | 16.425 | 0.0346 | 26.057 |
| C5_ours | 14.520 | 0.0034 | 26.046 |
| ORIG_ecr | nan | nan | 26.031 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +1.964 [+1.869,+2.061]
- ghost_C5_minus_C1: +0.058 [+0.048,+0.069]
- presU_C5_minus_ORIG: +0.015 [+0.005,+0.023]

Update cost: {"mask_pass_seconds": 37.6629524230957, "local_rebuild_seconds": 79.30500507354736, "bytes_rewritten_renders_depths": 1036497907, "bytes_masks": 1825264, "n_affected_views": 161, "n_train_views": 161}
