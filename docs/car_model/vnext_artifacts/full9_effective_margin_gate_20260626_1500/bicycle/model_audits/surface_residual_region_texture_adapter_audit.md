# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/bicycle/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `1256`
- support expansion added faces: `0`
- candidate faces after expansion: `1256`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1256`
- fit samples: `318918`
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
- surface multiscale prior blended bins: `38806`
- surface multiscale prior blended-bin fraction: `0.120689`
- surface multiscale prior mean blend weight: `0.082703`
- surface multiscale prior gate rejected bins: `273543`
- surface multiscale prior empty-bin rejects: `211819`
- surface multiscale prior sign rejects: `267477`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `57923`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `1189`
- view-conditioned basis supported-bin fraction: `0.003698`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `41`
- teacher-distilled basis supported-face fraction: `0.032643`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `116399`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.964615`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `48871`
- local alpha fallback bins: `40679`
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
- policy-val relative gain: `0.187942`
- policy-val positive-view fraction: `0.916667`
- policy-val CVaR20 view relative gain: `-0.023694`
- policy-val min-view relative gain: `-0.181883`
- policy-val image SSIM gain: `0.000035246`
- policy-val image SSIM positive-view fraction: `0.750000`
- policy-val image SSIM min-view gain: `-0.000072181`
- policy-val image L1 gain: `0.000030915`
- policy-val image L1 positive-view fraction: `0.916667`
- policy-val image L1 min-view gain: `-0.000017963`
- policy-val risk gate: `False`
- target written views: `25`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.023694 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.181883 < min_policy_val_min_view_relative_gain -0.000001; ssim_min_view_gain -0.000072181 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000017963 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_cvar20_view_gain -0.000041227 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000004572 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00915095 | 0.00915095 |
| 0.0020 | 0.001231 | 1.000000 | 0.000889 | 0.000758 | 0.000000452 | 0.916667 | 0.000000000 | 0.000000140 | 0.916667 | -0.000000030 | 0.00915095 | 0.00913968 |
| 0.0039 | 0.002457 | 1.000000 | 0.001774 | 0.001512 | 0.000000879 | 0.916667 | 0.000000000 | 0.000000281 | 0.916667 | -0.000000060 | 0.00915095 | 0.00912846 |
| 0.0078 | 0.004899 | 1.000000 | 0.003536 | 0.003012 | 0.000001719 | 0.916667 | 0.000000000 | 0.000000561 | 0.916667 | -0.000000119 | 0.00915095 | 0.00910611 |
| 0.0156 | 0.009736 | 1.000000 | 0.007021 | 0.005973 | 0.000003417 | 1.000000 | 0.000000060 | 0.000001121 | 0.916667 | -0.000000246 | 0.00915095 | 0.00906185 |
| 0.0312 | 0.019223 | 1.000000 | 0.013842 | 0.011744 | 0.000006651 | 1.000000 | 0.000000119 | 0.000002232 | 0.916667 | -0.000000507 | 0.00915095 | 0.00897503 |
| 0.0625 | 0.037450 | 1.000000 | 0.026884 | 0.022679 | 0.000012701 | 1.000000 | 0.000000179 | 0.000004423 | 0.916667 | -0.000001095 | 0.00915095 | 0.00880825 |
| 0.1250 | 0.070910 | 1.000000 | 0.050364 | 0.042123 | 0.000022863 | 0.916667 | -0.000001431 | 0.000008665 | 0.916667 | -0.000002533 | 0.00915095 | 0.00850205 |
| 0.2500 | 0.125867 | 1.000000 | 0.070785 | 0.064241 | 0.000035872 | 0.833333 | -0.000014126 | 0.000016664 | 0.916667 | -0.000006415 | 0.00915095 | 0.00799914 |
| 0.5000 | 0.187942 | 0.916667 | -0.023694 | -0.181883 | 0.000035246 | 0.750000 | -0.000072181 | 0.000030915 | 0.916667 | -0.000017963 | 0.00915095 | 0.00743110 |
