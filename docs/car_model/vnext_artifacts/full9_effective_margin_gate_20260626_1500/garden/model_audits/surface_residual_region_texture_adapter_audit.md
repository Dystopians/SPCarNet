# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/garden/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `319`
- support expansion added faces: `0`
- candidate faces after expansion: `319`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `319`
- fit samples: `201676`
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
- surface multiscale prior blended bins: `8224`
- surface multiscale prior blended-bin fraction: `0.100705`
- surface multiscale prior mean blend weight: `0.088017`
- surface multiscale prior gate rejected bins: `65212`
- surface multiscale prior empty-bin rejects: `52208`
- surface multiscale prior sign rejects: `62947`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `14059`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `3257`
- view-conditioned basis supported-bin fraction: `0.039883`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `53`
- teacher-distilled basis supported-face fraction: `0.166144`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `71583`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.954476`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `19527`
- local alpha fallback bins: `11335`
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
- policy-val relative gain: `0.132269`
- policy-val positive-view fraction: `0.833333`
- policy-val CVaR20 view relative gain: `-0.088840`
- policy-val min-view relative gain: `-0.170154`
- policy-val image SSIM gain: `-0.000009457`
- policy-val image SSIM positive-view fraction: `0.416667`
- policy-val image SSIM min-view gain: `-0.000074267`
- policy-val image L1 gain: `0.000002707`
- policy-val image L1 positive-view fraction: `0.750000`
- policy-val image L1 min-view gain: `-0.000011809`
- policy-val risk gate: `False`
- target written views: `24`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.088840 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.170154 < min_policy_val_min_view_relative_gain -0.000001; ssim_gain -0.000009457 < min_policy_val_ssim_mean_gain -0.000000100; ssim_positive_view_fraction 0.416667 < min_policy_val_ssim_positive_view_fraction 0.550000; ssim_min_view_gain -0.000074267 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000011809 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_gain -0.000009457 < min_policy_val_effective_ssim_gain 0.000010000; effective_ssim_cvar20_view_gain -0.000056684 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000008947 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00366984 | 0.00366984 |
| 0.0020 | 0.000980 | 0.916667 | 0.000346 | -0.000082 | 0.000000104 | 0.666667 | -0.000000119 | 0.000000018 | 0.750000 | -0.000000039 | 0.00366984 | 0.00366624 |
| 0.0039 | 0.001957 | 0.916667 | 0.000690 | -0.000168 | 0.000000209 | 0.750000 | -0.000000298 | 0.000000034 | 0.750000 | -0.000000075 | 0.00366984 | 0.00366266 |
| 0.0078 | 0.003899 | 0.916667 | 0.001365 | -0.000355 | 0.000000412 | 0.833333 | -0.000000596 | 0.000000069 | 0.750000 | -0.000000149 | 0.00366984 | 0.00365553 |
| 0.0156 | 0.007740 | 0.916667 | 0.002677 | -0.000783 | 0.000000805 | 0.916667 | -0.000001252 | 0.000000139 | 0.750000 | -0.000000302 | 0.00366984 | 0.00364143 |
| 0.0312 | 0.015247 | 0.916667 | 0.005138 | -0.001858 | 0.000001550 | 0.833333 | -0.000002682 | 0.000000271 | 0.750000 | -0.000000609 | 0.00366984 | 0.00361388 |
| 0.0625 | 0.029565 | 0.916667 | 0.009411 | -0.004887 | 0.000002841 | 0.833333 | -0.000005603 | 0.000000530 | 0.750000 | -0.000001235 | 0.00366984 | 0.00356134 |
| 0.1250 | 0.055408 | 0.916667 | 0.015363 | -0.014454 | 0.000004520 | 0.833333 | -0.000012279 | 0.000001010 | 0.750000 | -0.000002550 | 0.00366984 | 0.00346650 |
| 0.2500 | 0.095930 | 0.916667 | 0.012973 | -0.047631 | 0.000004396 | 0.583333 | -0.000028789 | 0.000001797 | 0.750000 | -0.000005402 | 0.00366984 | 0.00331779 |
| 0.5000 | 0.132269 | 0.833333 | -0.088840 | -0.170154 | -0.000009457 | 0.416667 | -0.000074267 | 0.000002707 | 0.750000 | -0.000011809 | 0.00366984 | 0.00318443 |
