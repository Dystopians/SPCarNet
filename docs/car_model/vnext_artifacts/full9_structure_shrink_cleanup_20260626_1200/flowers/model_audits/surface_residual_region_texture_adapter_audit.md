# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/flowers/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `342`
- support expansion added faces: `0`
- candidate faces after expansion: `342`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `342`
- fit samples: `130028`
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
- surface multiscale prior blended bins: `12443`
- surface multiscale prior blended-bin fraction: `0.142121`
- surface multiscale prior mean blend weight: `0.085256`
- surface multiscale prior gate rejected bins: `71099`
- surface multiscale prior empty-bin rejects: `54372`
- surface multiscale prior sign rejects: `68810`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `13450`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `1005`
- view-conditioned basis supported-bin fraction: `0.011479`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
- teacher-distilled basis supported faces: `15`
- teacher-distilled basis supported-face fraction: `0.043860`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `47704`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.971365`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `19304`
- local alpha fallback bins: `11112`
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
- policy-val relative gain: `0.125562`
- policy-val positive-view fraction: `0.583333`
- policy-val CVaR20 view relative gain: `-0.224441`
- policy-val min-view relative gain: `-0.278408`
- policy-val image SSIM gain: `-0.000082279`
- policy-val image SSIM positive-view fraction: `0.333333`
- policy-val image SSIM min-view gain: `-0.000208378`
- policy-val image L1 gain: `-0.000000106`
- policy-val image L1 positive-view fraction: `0.500000`
- policy-val image L1 min-view gain: `-0.000023652`
- policy-val risk gate: `False`
- target written views: `22`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.224441 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.278408 < min_policy_val_min_view_relative_gain -0.000001; ssim_gain -0.000082279 < min_policy_val_ssim_mean_gain -0.000000100; ssim_positive_view_fraction 0.333333 < min_policy_val_ssim_positive_view_fraction 0.550000; ssim_min_view_gain -0.000208378 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_gain -0.000000106 < min_policy_val_l1_mean_gain 0.000000000; image_l1_positive_view_fraction 0.500000 < min_policy_val_l1_positive_view_fraction 0.550000; image_l1_min_view_gain -0.000023652 < min_policy_val_l1_min_view_gain -0.000001000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00986263 | 0.00986263 |
| 0.0020 | 0.001275 | 1.000000 | 0.000529 | 0.000260 | 0.000000079 | 0.666667 | -0.000000238 | 0.000000009 | 0.500000 | -0.000000086 | 0.00986263 | 0.00985006 |
| 0.0039 | 0.002543 | 1.000000 | 0.001048 | 0.000509 | 0.000000104 | 0.666667 | -0.000000596 | 0.000000016 | 0.500000 | -0.000000168 | 0.00986263 | 0.00983755 |
| 0.0078 | 0.005062 | 1.000000 | 0.002051 | 0.000977 | 0.000000204 | 0.666667 | -0.000001073 | 0.000000033 | 0.500000 | -0.000000346 | 0.00986263 | 0.00981271 |
| 0.0156 | 0.010025 | 1.000000 | 0.003926 | 0.001787 | 0.000000293 | 0.666667 | -0.000002146 | 0.000000066 | 0.500000 | -0.000000693 | 0.00986263 | 0.00976376 |
| 0.0312 | 0.019656 | 1.000000 | 0.007145 | 0.002908 | 0.000000417 | 0.500000 | -0.000004292 | 0.000000120 | 0.500000 | -0.000001397 | 0.00986263 | 0.00966878 |
| 0.0625 | 0.037737 | 1.000000 | 0.011465 | 0.003151 | -0.000000094 | 0.416667 | -0.000009298 | 0.000000217 | 0.500000 | -0.000002809 | 0.00986263 | 0.00949045 |
| 0.1250 | 0.069175 | 0.916667 | 0.011630 | -0.004356 | -0.000003671 | 0.416667 | -0.000020802 | 0.000000366 | 0.500000 | -0.000005651 | 0.00986263 | 0.00918038 |
| 0.2500 | 0.113158 | 0.833333 | -0.021941 | -0.051345 | -0.000020261 | 0.333333 | -0.000068843 | 0.000000472 | 0.500000 | -0.000011455 | 0.00986263 | 0.00874659 |
| 0.5000 | 0.125562 | 0.583333 | -0.224441 | -0.278408 | -0.000082279 | 0.333333 | -0.000208378 | -0.000000106 | 0.500000 | -0.000023652 | 0.00986263 | 0.00862426 |
