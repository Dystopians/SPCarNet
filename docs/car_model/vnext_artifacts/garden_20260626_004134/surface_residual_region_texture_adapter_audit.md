# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1`
- target evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/garden`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json`
- support expansion mode: `target_footprint_residual_debt_ladder`
- support expansion base faces: `319`
- support expansion added faces: `0`
- candidate faces after expansion: `0`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `319`
- fit samples: `201676`
- selected support mode: `base_carrier`
- selected support added faces: `0`
- selected texture size: `8`
- selected fill mode: `nearest_observed`
- selected max abs delta RGB: `0.120000`
- max abs delta RGB candidates: `[0.08, 0.12]`
- policy candidate dominance pruning: `True`
- policy candidates planned before pruning: `48`
- policy candidates planned after pruning: `48`
- policy candidates executed: `48`
- policy candidate early-stop mode: `none`
- policy candidate early-stop skipped: `0`
- surface multiscale prior mode: `local_patch`
- surface multiscale prior selected blend: `0.250000`
- surface multiscale prior blend candidates: `[0.0, 0.25, 0.5]`
- surface multiscale prior gate mode: `evidence_consistent`
- surface multiscale prior block sizes: `[2, 4, 8]`
- surface multiscale prior blended bins: `1782`
- surface multiscale prior blended-bin fraction: `0.087284`
- surface multiscale prior mean blend weight: `0.044298`
- surface multiscale prior gate rejected bins: `13520`
- surface multiscale prior empty-bin rejects: `10758`
- surface multiscale prior sign rejects: `13008`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `5253`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `3238`
- view-conditioned basis supported-bin fraction: `0.158601`
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
- local alpha calibration: `False`
- local alpha mode: `disabled`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `0`
- local alpha uncertainty-shrink policy mode: `n/a`
- local alpha uncertainty-shrink mean: `0.000000`
- local alpha uncertainty-shrink downweighted bins: `0`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `0`
- local alpha fallback bins: `0`
- face gain guard enabled: `False`
- face gain guard decision: `skipped_candidate_not_accepted`
- face gain guard allowed faces: `0`
- face gain guard rejected faces: `0`
- face gain guard allowed sample fraction: `0.000000`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `skipped_candidate_not_accepted`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.138933`
- policy-val positive-view fraction: `0.750000`
- policy-val CVaR20 view relative gain: `-0.155822`
- policy-val min-view relative gain: `-0.248611`
- policy-val image SSIM gain: `-0.000018626`
- policy-val image SSIM positive-view fraction: `0.250000`
- policy-val image SSIM min-view gain: `-0.000094116`
- policy-val image L1 gain: `0.000002394`
- policy-val image L1 positive-view fraction: `0.666667`
- policy-val image L1 min-view gain: `-0.000016468`
- policy-val risk gate: `False`
- target written views: `24`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.155822 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.248611 < min_policy_val_min_view_relative_gain -0.000001; ssim_gain -0.000018626 < min_policy_val_ssim_mean_gain -0.000000100; ssim_positive_view_fraction 0.250000 < min_policy_val_ssim_positive_view_fraction 0.550000; ssim_min_view_gain -0.000094116 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000016468 < min_policy_val_l1_min_view_gain -0.000001000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00366984 | 0.00366984 |
| 0.1250 | 0.062002 | 0.916667 | 0.012557 | -0.025082 | 0.000004202 | 0.750000 | -0.000014126 | 0.000000992 | 0.750000 | -0.000003673 | 0.00366984 | 0.00344230 |
| 0.2500 | 0.105635 | 0.916667 | -0.008216 | -0.074877 | 0.000002429 | 0.666667 | -0.000032127 | 0.000001719 | 0.666667 | -0.000007646 | 0.00366984 | 0.00328218 |
| 0.5000 | 0.138933 | 0.750000 | -0.155822 | -0.248611 | -0.000018626 | 0.250000 | -0.000094116 | 0.000002394 | 0.666667 | -0.000016468 | 0.00366984 | 0.00315998 |
