# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter_teacher_surface_evidence_phasej_trainval_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/counter/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `1574`
- support expansion added faces: `0`
- candidate faces after expansion: `1574`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1574`
- fit samples: `1463293`
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
- surface multiscale prior blended bins: `24269`
- surface multiscale prior blended-bin fraction: `0.060229`
- surface multiscale prior mean blend weight: `0.087948`
- surface multiscale prior gate rejected bins: `320942`
- surface multiscale prior empty-bin rejects: `267012`
- surface multiscale prior sign rejects: `312609`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `67595`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `20661`
- view-conditioned basis supported-bin fraction: `0.051275`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `266`
- teacher-distilled basis supported-face fraction: `0.168996`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `549308`
- selected alpha: `0.125`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.737122`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `85096`
- local alpha fallback bins: `76904`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `257`
- face gain guard rejected faces: `1317`
- face gain guard allowed sample fraction: `0.639421`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.020429`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.007942`
- policy-val min-view relative gain: `0.003912`
- policy-val image SSIM gain: `0.000050878`
- policy-val image SSIM positive-view fraction: `1.000000`
- policy-val image SSIM min-view gain: `0.000005662`
- policy-val image L1 gain: `0.000006301`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000000440`
- policy-val risk gate: `True`
- target written views: `30`
- target changed fraction: `0.012344`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00558927 | 0.00558927 |
| 0.0020 | 0.000360 | 1.000000 | 0.000165 | 0.000123 | 0.000001018 | 1.000000 | 0.000000119 | 0.000000110 | 1.000000 | 0.000000002 | 0.00558927 | 0.00558726 |
| 0.0039 | 0.000718 | 1.000000 | 0.000328 | 0.000243 | 0.000002022 | 1.000000 | 0.000000298 | 0.000000219 | 1.000000 | 0.000000004 | 0.00558927 | 0.00558526 |
| 0.0078 | 0.001431 | 1.000000 | 0.000651 | 0.000478 | 0.000004033 | 1.000000 | 0.000000656 | 0.000000436 | 1.000000 | 0.000000004 | 0.00558927 | 0.00558127 |
| 0.0156 | 0.002841 | 1.000000 | 0.001280 | 0.000922 | 0.000007952 | 1.000000 | 0.000001371 | 0.000000866 | 1.000000 | 0.000000004 | 0.00558927 | 0.00557339 |
| 0.0312 | 0.005600 | 1.000000 | 0.002475 | 0.001711 | 0.000015462 | 1.000000 | 0.000002742 | 0.000001708 | 0.833333 | -0.000000017 | 0.00558927 | 0.00555797 |
| 0.0625 | 0.010870 | 1.000000 | 0.004608 | 0.002886 | 0.000029107 | 1.000000 | 0.000005126 | 0.000003325 | 0.833333 | -0.000000101 | 0.00558927 | 0.00552851 |
| 0.1250 | 0.020429 | 1.000000 | 0.007942 | 0.003912 | 0.000050878 | 1.000000 | 0.000005662 | 0.000006301 | 0.833333 | -0.000000440 | 0.00558927 | 0.00547509 |
| 0.2500 | 0.035713 | 1.000000 | 0.011128 | 0.001094 | 0.000073507 | 0.833333 | -0.000006855 | 0.000011137 | 0.833333 | -0.000002088 | 0.00558927 | 0.00538966 |
| 0.5000 | 0.051550 | 0.833333 | -0.005778 | -0.023271 | 0.000042761 | 0.750000 | -0.000460744 | 0.000016503 | 0.750000 | -0.000008358 | 0.00558927 | 0.00530114 |
