# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/bonsai/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_teacher_render_visible_region_carriers_v37_visible_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `1110`
- support expansion added faces: `0`
- candidate faces after expansion: `1110`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1110`
- fit samples: `537599`
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
- surface multiscale prior blended bins: `24721`
- surface multiscale prior blended-bin fraction: `0.086997`
- surface multiscale prior mean blend weight: `0.084736`
- surface multiscale prior gate rejected bins: `231424`
- surface multiscale prior empty-bin rejects: `192776`
- surface multiscale prior sign rejects: `227242`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `43177`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `7527`
- view-conditioned basis supported-bin fraction: `0.026489`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `112`
- teacher-distilled basis supported-face fraction: `0.100901`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `192419`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.924704`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `50650`
- local alpha fallback bins: `42458`
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
- policy-val relative gain: `0.194345`
- policy-val positive-view fraction: `0.833333`
- policy-val CVaR20 view relative gain: `-0.002950`
- policy-val min-view relative gain: `-0.076972`
- policy-val image SSIM gain: `0.000065108`
- policy-val image SSIM positive-view fraction: `0.666667`
- policy-val image SSIM min-view gain: `-0.000134230`
- policy-val image L1 gain: `0.000036387`
- policy-val image L1 positive-view fraction: `0.666667`
- policy-val image L1 min-view gain: `-0.000015322`
- policy-val risk gate: `False`
- target written views: `37`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.002950 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.076972 < min_policy_val_min_view_relative_gain -0.000001; ssim_min_view_gain -0.000134230 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000015322 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_cvar20_view_gain -0.000115176 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000007760 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.01079320 | 0.01079320 |
| 0.0020 | 0.001118 | 1.000000 | 0.000692 | 0.000562 | 0.000000884 | 1.000000 | 0.000000179 | 0.000000173 | 0.833333 | -0.000000019 | 0.01079320 | 0.01078113 |
| 0.0039 | 0.002233 | 1.000000 | 0.001379 | 0.001123 | 0.000001763 | 1.000000 | 0.000000298 | 0.000000346 | 0.833333 | -0.000000037 | 0.01079320 | 0.01076909 |
| 0.0078 | 0.004455 | 1.000000 | 0.002737 | 0.002242 | 0.000003517 | 1.000000 | 0.000000596 | 0.000000691 | 0.833333 | -0.000000078 | 0.01079320 | 0.01074511 |
| 0.0156 | 0.008865 | 1.000000 | 0.005387 | 0.004469 | 0.000006959 | 1.000000 | 0.000001192 | 0.000001379 | 0.833333 | -0.000000158 | 0.01079320 | 0.01069752 |
| 0.0312 | 0.017548 | 1.000000 | 0.010424 | 0.008871 | 0.000013575 | 1.000000 | 0.000001907 | 0.000002742 | 0.833333 | -0.000000339 | 0.01079320 | 0.01060380 |
| 0.0625 | 0.034370 | 1.000000 | 0.019449 | 0.017479 | 0.000025888 | 1.000000 | 0.000002444 | 0.000005425 | 0.833333 | -0.000000767 | 0.01079320 | 0.01042223 |
| 0.1250 | 0.065839 | 1.000000 | 0.033304 | 0.030379 | 0.000046665 | 0.916667 | -0.000000894 | 0.000010591 | 0.833333 | -0.000001989 | 0.01079320 | 0.01008258 |
| 0.2500 | 0.120113 | 1.000000 | 0.044838 | 0.025985 | 0.000072842 | 0.833333 | -0.000024319 | 0.000020162 | 0.833333 | -0.000005487 | 0.01079320 | 0.00949679 |
| 0.5000 | 0.194345 | 0.833333 | -0.002950 | -0.076972 | 0.000065108 | 0.666667 | -0.000134230 | 0.000036387 | 0.666667 | -0.000015322 | 0.01079320 | 0.00869559 |
