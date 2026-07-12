# Edit-aware ECR eval — toy_parking (delete_faces_box, 711609 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 31.846 | 0.0000 | 35.946 |
| C2_stale | 36.976 | 0.0013 | 36.881 |
| C5_ours | 36.578 | 0.0013 | 36.874 |
| ORIG_ecr | nan | nan | 36.883 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +5.130 [+0.911,+10.173]
- ghost_C5_minus_C1: +4.732 [+0.858,+9.237]
- presU_C5_minus_ORIG: -0.009 [-0.023,+0.001]

Update cost: {"mask_pass_seconds": 15.184749126434326, "local_rebuild_seconds": 15.862394571304321, "bytes_rewritten_renders_depths": 12790131, "bytes_masks": 418761, "n_affected_views": 72, "n_train_views": 72}
