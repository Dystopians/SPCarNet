# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/room_teacher_surface_evidence_phasej_trainval_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/room/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `1160`
- support expansion added faces: `0`
- candidate faces after expansion: `1160`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1160`
- fit samples: `1023700`
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
- surface multiscale prior blended bins: `17926`
- surface multiscale prior blended-bin fraction: `0.060365`
- surface multiscale prior mean blend weight: `0.090717`
- surface multiscale prior gate rejected bins: `233805`
- surface multiscale prior empty-bin rejects: `195417`
- surface multiscale prior sign rejects: `229037`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `42384`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `18037`
- view-conditioned basis supported-bin fraction: `0.060739`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `245`
- teacher-distilled basis supported-face fraction: `0.211207`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `340299`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.829580`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `67040`
- local alpha fallback bins: `58848`
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
- policy-val relative gain: `0.188480`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.072160`
- policy-val min-view relative gain: `0.063353`
- policy-val image SSIM gain: `0.000035971`
- policy-val image SSIM positive-view fraction: `0.666667`
- policy-val image SSIM min-view gain: `-0.000249982`
- policy-val image L1 gain: `0.000027366`
- policy-val image L1 positive-view fraction: `0.916667`
- policy-val image L1 min-view gain: `-0.000020005`
- policy-val risk gate: `False`
- target written views: `39`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `ssim_min_view_gain -0.000249982 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000020005 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_cvar20_view_gain -0.000203470 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000006014 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00461406 | 0.00461406 |
| 0.0020 | 0.001017 | 1.000000 | 0.000683 | 0.000576 | 0.000001142 | 1.000000 | 0.000000119 | 0.000000137 | 0.916667 | -0.000000058 | 0.00461406 | 0.00460937 |
| 0.0039 | 0.002031 | 1.000000 | 0.001363 | 0.001149 | 0.000002275 | 1.000000 | 0.000000238 | 0.000000275 | 0.916667 | -0.000000117 | 0.00461406 | 0.00460469 |
| 0.0078 | 0.004054 | 1.000000 | 0.002715 | 0.002285 | 0.000004510 | 1.000000 | 0.000000477 | 0.000000548 | 0.916667 | -0.000000237 | 0.00461406 | 0.00459536 |
| 0.0156 | 0.008072 | 1.000000 | 0.005381 | 0.004519 | 0.000008861 | 1.000000 | 0.000000954 | 0.000001093 | 0.916667 | -0.000000481 | 0.00461406 | 0.00457682 |
| 0.0312 | 0.016002 | 1.000000 | 0.010569 | 0.008836 | 0.000017186 | 1.000000 | 0.000001729 | 0.000002167 | 0.916667 | -0.000000970 | 0.00461406 | 0.00454023 |
| 0.0625 | 0.031436 | 1.000000 | 0.020366 | 0.016862 | 0.000032221 | 1.000000 | 0.000002503 | 0.000004260 | 0.916667 | -0.000001974 | 0.00461406 | 0.00446902 |
| 0.1250 | 0.060598 | 1.000000 | 0.037657 | 0.030519 | 0.000055825 | 1.000000 | 0.000000596 | 0.000008257 | 0.916667 | -0.000004100 | 0.00461406 | 0.00433446 |
| 0.2500 | 0.112177 | 1.000000 | 0.064468 | 0.052583 | 0.000078514 | 0.750000 | -0.000024021 | 0.000015549 | 0.916667 | -0.000008842 | 0.00461406 | 0.00409647 |
| 0.5000 | 0.188480 | 1.000000 | 0.072160 | 0.063353 | 0.000035971 | 0.666667 | -0.000249982 | 0.000027366 | 0.916667 | -0.000020005 | 0.00461406 | 0.00374440 |
