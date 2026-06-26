# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_structure_shrink_bicycle_strict_20260626_1055/bicycle/target_evidence_no_gt`
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
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `41`
- teacher-distilled basis supported-face fraction: `0.032643`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `116399`
- selected alpha: `0.015625`
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
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `49`
- face gain guard rejected faces: `1207`
- face gain guard allowed sample fraction: `0.267388`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.002599`
- policy-val positive-view fraction: `0.750000`
- policy-val CVaR20 view relative gain: `0.000000`
- policy-val min-view relative gain: `0.000000`
- policy-val image SSIM gain: `0.000000571`
- policy-val image SSIM positive-view fraction: `0.583333`
- policy-val image SSIM min-view gain: `-0.000000238`
- policy-val image L1 gain: `0.000000277`
- policy-val image L1 positive-view fraction: `0.583333`
- policy-val image L1 min-view gain: `-0.000000075`
- policy-val risk gate: `True`
- target written views: `25`
- target changed fraction: `0.000174`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00915095 | 0.00915095 |
| 0.0020 | 0.000328 | 0.750000 | 0.000000 | 0.000000 | 0.000000084 | 0.416667 | 0.000000000 | 0.000000034 | 0.583333 | -0.000000015 | 0.00915095 | 0.00914794 |
| 0.0039 | 0.000656 | 0.750000 | 0.000000 | 0.000000 | 0.000000149 | 0.416667 | -0.000000060 | 0.000000069 | 0.583333 | -0.000000015 | 0.00915095 | 0.00914494 |
| 0.0078 | 0.001308 | 0.750000 | 0.000000 | 0.000000 | 0.000000298 | 0.500000 | -0.000000060 | 0.000000137 | 0.583333 | -0.000000045 | 0.00915095 | 0.00913898 |
| 0.0156 | 0.002599 | 0.750000 | 0.000000 | 0.000000 | 0.000000571 | 0.583333 | -0.000000238 | 0.000000277 | 0.583333 | -0.000000075 | 0.00915095 | 0.00912716 |
| 0.0312 | 0.005135 | 0.666667 | -0.000011 | -0.000034 | 0.000001108 | 0.583333 | -0.000000536 | 0.000000551 | 0.583333 | -0.000000134 | 0.00915095 | 0.00910395 |
| 0.0625 | 0.010016 | 0.666667 | -0.000095 | -0.000284 | 0.000002071 | 0.583333 | -0.000001729 | 0.000001098 | 0.583333 | -0.000000261 | 0.00915095 | 0.00905929 |
| 0.1250 | 0.019014 | 0.666667 | -0.000477 | -0.001432 | 0.000003576 | 0.500000 | -0.000006318 | 0.000002161 | 0.583333 | -0.000000536 | 0.00915095 | 0.00897695 |
| 0.2500 | 0.033957 | 0.666667 | -0.002108 | -0.006323 | 0.000005086 | 0.416667 | -0.000022590 | 0.000004208 | 0.583333 | -0.000001132 | 0.00915095 | 0.00884021 |
| 0.5000 | 0.051681 | 0.583333 | -0.010046 | -0.026219 | 0.000002876 | 0.333333 | -0.000076056 | 0.000007985 | 0.583333 | -0.000002630 | 0.00915095 | 0.00867802 |
