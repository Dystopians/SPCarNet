# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/treehill/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `226`
- support expansion added faces: `0`
- candidate faces after expansion: `226`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `226`
- fit samples: `204998`
- selected support mode: `base_carrier`
- selected support added faces: `0`
- selected texture size: `16`
- selected fill mode: `face_mean`
- selected max abs delta RGB: `0.120000`
- max abs delta RGB candidates: `[0.12]`
- policy candidate dominance pruning: `True`
- policy candidates planned before pruning: `1`
- policy candidates planned after pruning: `1`
- policy candidates executed: `1`
- policy candidate early-stop mode: `none`
- policy candidate early-stop skipped: `0`
- surface multiscale prior mode: `local_patch`
- surface multiscale prior selected blend: `0.500000`
- surface multiscale prior blend candidates: `[0.5]`
- surface multiscale prior gate mode: `evidence_consistent`
- surface multiscale prior block sizes: `[2, 4, 8]`
- surface multiscale prior blended bins: `6651`
- surface multiscale prior blended-bin fraction: `0.114958`
- surface multiscale prior mean blend weight: `0.085020`
- surface multiscale prior gate rejected bins: `45310`
- surface multiscale prior empty-bin rejects: `35235`
- surface multiscale prior sign rejects: `44148`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `9274`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `3075`
- view-conditioned basis supported-bin fraction: `0.053149`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `36`
- teacher-distilled basis supported-face fraction: `0.159292`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `72217`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.940852`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `13163`
- local alpha fallback bins: `4971`
- face gain guard enabled: `False`
- face gain guard decision: `skipped_candidate_not_accepted`
- face gain guard allowed faces: `0`
- face gain guard rejected faces: `0`
- face gain guard allowed sample fraction: `0.000000`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.202200`
- policy-val positive-view fraction: `0.916667`
- policy-val CVaR20 view relative gain: `-0.028941`
- policy-val min-view relative gain: `-0.225241`
- policy-val image SSIM gain: `0.000001470`
- policy-val image SSIM positive-view fraction: `0.583333`
- policy-val image SSIM min-view gain: `-0.000080764`
- policy-val image L1 gain: `0.000017347`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000002254`
- policy-val risk gate: `False`
- target written views: `18`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.028941 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.225241 < min_policy_val_min_view_relative_gain -0.000001; ssim_min_view_gain -0.000080764 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000002254 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_gain 0.000001470 < min_policy_val_effective_ssim_gain 0.000010000; effective_ssim_cvar20_view_gain -0.000039359 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000000387 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00769584 | 0.00769584 |
| 0.0020 | 0.001411 | 1.000000 | 0.000880 | 0.000602 | 0.000000149 | 0.583333 | -0.000000060 | 0.000000075 | 0.916667 | -0.000000007 | 0.00769584 | 0.00768498 |
| 0.0039 | 0.002818 | 1.000000 | 0.001757 | 0.001201 | 0.000000348 | 0.750000 | 0.000000000 | 0.000000149 | 0.916667 | -0.000000015 | 0.00769584 | 0.00767416 |
| 0.0078 | 0.005616 | 1.000000 | 0.003505 | 0.002393 | 0.000000656 | 0.833333 | -0.000000060 | 0.000000302 | 0.916667 | -0.000000030 | 0.00769584 | 0.00765262 |
| 0.0156 | 0.011153 | 1.000000 | 0.006930 | 0.004748 | 0.000001281 | 0.833333 | -0.000000119 | 0.000000601 | 0.916667 | -0.000000063 | 0.00769584 | 0.00761001 |
| 0.0312 | 0.021994 | 1.000000 | 0.013397 | 0.009343 | 0.000002479 | 0.833333 | -0.000000298 | 0.000001198 | 0.916667 | -0.000000130 | 0.00769584 | 0.00752657 |
| 0.0625 | 0.042741 | 1.000000 | 0.024944 | 0.018075 | 0.000004565 | 0.833333 | -0.000000715 | 0.000002377 | 0.916667 | -0.000000253 | 0.00769584 | 0.00736691 |
| 0.1250 | 0.080490 | 1.000000 | 0.042492 | 0.033709 | 0.000007868 | 0.833333 | -0.000001669 | 0.000004695 | 0.916667 | -0.000000510 | 0.00769584 | 0.00707640 |
| 0.2500 | 0.141012 | 1.000000 | 0.055397 | 0.032829 | 0.000010649 | 0.833333 | -0.000004768 | 0.000009155 | 0.916667 | -0.000001051 | 0.00769584 | 0.00661064 |
| 0.5000 | 0.202200 | 0.916667 | -0.028941 | -0.225241 | 0.000001470 | 0.583333 | -0.000080764 | 0.000017347 | 0.833333 | -0.000002254 | 0.00769584 | 0.00613974 |
