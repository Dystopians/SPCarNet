# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_structure_shrink_treehill_strict_20260626_0832/treehill/target_evidence_no_gt`
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
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
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
- face gain guard enabled: `True`
- face gain guard decision: `reject_candidate_after_face_gain_guard`
- face gain guard allowed faces: `27`
- face gain guard rejected faces: `199`
- face gain guard allowed sample fraction: `0.392190`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.076816`
- policy-val positive-view fraction: `0.750000`
- policy-val CVaR20 view relative gain: `-0.053640`
- policy-val min-view relative gain: `-0.077837`
- policy-val image SSIM gain: `-0.000009413`
- policy-val image SSIM positive-view fraction: `0.583333`
- policy-val image SSIM min-view gain: `-0.000124395`
- policy-val image L1 gain: `0.000008294`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000009734`
- policy-val risk gate: `False`
- target written views: `18`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.053640 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.077837 < min_policy_val_min_view_relative_gain -0.000001; ssim_gain -0.000009413 < min_policy_val_ssim_mean_gain -0.000000100; ssim_min_view_gain -0.000124395 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000009734 < min_policy_val_l1_min_view_gain -0.000001000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00769584 | 0.00769584 |
| 0.0020 | 0.000569 | 0.916667 | 0.000082 | -0.000001 | 0.000000030 | 0.500000 | -0.000000119 | 0.000000034 | 0.750000 | -0.000000037 | 0.00769584 | 0.00769146 |
| 0.0039 | 0.001136 | 0.916667 | 0.000163 | -0.000004 | 0.000000050 | 0.500000 | -0.000000298 | 0.000000069 | 0.833333 | -0.000000075 | 0.00769584 | 0.00768710 |
| 0.0078 | 0.002263 | 0.916667 | 0.000321 | -0.000017 | 0.000000089 | 0.583333 | -0.000000656 | 0.000000141 | 0.833333 | -0.000000153 | 0.00769584 | 0.00767842 |
| 0.0156 | 0.004492 | 0.916667 | 0.000618 | -0.000067 | 0.000000189 | 0.583333 | -0.000001252 | 0.000000280 | 0.833333 | -0.000000305 | 0.00769584 | 0.00766127 |
| 0.0312 | 0.008850 | 0.916667 | 0.001142 | -0.000267 | 0.000000358 | 0.666667 | -0.000002503 | 0.000000559 | 0.833333 | -0.000000607 | 0.00769584 | 0.00762773 |
| 0.0625 | 0.017160 | 0.916667 | 0.001903 | -0.001064 | 0.000000576 | 0.666667 | -0.000005424 | 0.000001110 | 0.833333 | -0.000001211 | 0.00769584 | 0.00756378 |
| 0.1250 | 0.032160 | 0.916667 | 0.002287 | -0.004250 | 0.000000641 | 0.666667 | -0.000013947 | 0.000002201 | 0.833333 | -0.000002433 | 0.00769584 | 0.00744834 |
| 0.2500 | 0.055682 | 0.916667 | -0.002869 | -0.016983 | -0.000000755 | 0.666667 | -0.000039876 | 0.000004306 | 0.833333 | -0.000004865 | 0.00769584 | 0.00726732 |
| 0.5000 | 0.076816 | 0.750000 | -0.053640 | -0.077837 | -0.000009413 | 0.583333 | -0.000124395 | 0.000008294 | 0.833333 | -0.000009734 | 0.00769584 | 0.00710468 |
