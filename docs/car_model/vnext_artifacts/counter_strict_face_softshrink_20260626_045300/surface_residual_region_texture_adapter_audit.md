# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter_teacher_surface_evidence_phasej_trainval_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_counter_strict_20260626_045300_counter_strict/counter/target_evidence_no_gt`
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
- selected alpha: `0.25`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.880753`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `85096`
- local alpha fallback bins: `76904`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `235`
- face gain guard rejected faces: `1339`
- face gain guard allowed sample fraction: `0.616322`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.044316`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.018847`
- policy-val min-view relative gain: `0.014913`
- policy-val image SSIM gain: `0.000103652`
- policy-val image SSIM positive-view fraction: `1.000000`
- policy-val image SSIM min-view gain: `0.000011325`
- policy-val image L1 gain: `0.000014212`
- policy-val image L1 positive-view fraction: `1.000000`
- policy-val image L1 min-view gain: `0.000000073`
- policy-val risk gate: `True`
- target written views: `30`
- target changed fraction: `0.011774`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00558927 | 0.00558927 |
| 0.0020 | 0.000430 | 1.000000 | 0.000232 | 0.000187 | 0.000001242 | 1.000000 | 0.000000179 | 0.000000133 | 1.000000 | 0.000000007 | 0.00558927 | 0.00558686 |
| 0.0039 | 0.000859 | 1.000000 | 0.000463 | 0.000374 | 0.000002488 | 1.000000 | 0.000000358 | 0.000000265 | 1.000000 | 0.000000011 | 0.00558927 | 0.00558447 |
| 0.0078 | 0.001713 | 1.000000 | 0.000922 | 0.000742 | 0.000004942 | 1.000000 | 0.000000775 | 0.000000529 | 1.000000 | 0.000000020 | 0.00558927 | 0.00557969 |
| 0.0156 | 0.003405 | 1.000000 | 0.001823 | 0.001466 | 0.000009795 | 1.000000 | 0.000001609 | 0.000001052 | 1.000000 | 0.000000035 | 0.00558927 | 0.00557024 |
| 0.0312 | 0.006724 | 1.000000 | 0.003563 | 0.002859 | 0.000019153 | 1.000000 | 0.000003099 | 0.000002084 | 1.000000 | 0.000000067 | 0.00558927 | 0.00555169 |
| 0.0625 | 0.013104 | 1.000000 | 0.006794 | 0.005422 | 0.000036493 | 1.000000 | 0.000005960 | 0.000004084 | 1.000000 | 0.000000108 | 0.00558927 | 0.00551603 |
| 0.1250 | 0.024841 | 1.000000 | 0.012358 | 0.009664 | 0.000065789 | 1.000000 | 0.000010431 | 0.000007829 | 1.000000 | 0.000000153 | 0.00558927 | 0.00545043 |
| 0.2500 | 0.044316 | 1.000000 | 0.018847 | 0.014913 | 0.000103652 | 1.000000 | 0.000011325 | 0.000014212 | 1.000000 | 0.000000073 | 0.00558927 | 0.00534158 |
| 0.5000 | 0.067744 | 0.916667 | 0.006596 | -0.006435 | 0.000103061 | 0.833333 | -0.000362694 | 0.000022609 | 0.916667 | -0.000000641 | 0.00558927 | 0.00521063 |
