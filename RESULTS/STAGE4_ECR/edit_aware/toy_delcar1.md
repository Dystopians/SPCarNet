# Edit-aware ECR eval — toy_parking (delete_faces_box, 747094 faces)

ghost_psnr_R: similarity of the edit region to the ORIGINAL unedited ECR output — HIGHER = MORE stale-object ghosting. leak_R: mean|. − edited-base| in-region. psnr_U_gt: TRUE-GT PSNR outside the (dilated) edit region.

| method | ghost_psnr_R ↓ | leak_R ↓ | psnr_U (true GT) ↑ |
|---|---|---|---|
| C1_editedbase | 23.506 | 0.0000 | 30.813 |
| C2_stale | 28.079 | 0.0024 | 31.873 |
| C5_ours | 27.746 | 0.0025 | 31.872 |
| ORIG_ecr | nan | nan | 31.874 |

**Paired CIs (10k, seed 0):**
- ghost_C2_minus_C1: +4.573 [+0.150,+11.278]
- ghost_C5_minus_C1: +4.240 [+0.153,+10.286]
- presU_C5_minus_ORIG: -0.002 [-0.004,-0.001]

Update cost: {"mask_pass_seconds": 14.753173112869263, "local_rebuild_seconds": 17.93948221206665, "bytes_rewritten_renders_depths": 228126243, "bytes_masks": 411483, "n_affected_views": 71, "n_train_views": 72}
