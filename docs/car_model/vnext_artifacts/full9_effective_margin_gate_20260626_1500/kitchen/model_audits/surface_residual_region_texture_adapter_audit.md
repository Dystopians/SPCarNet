# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/kitchen/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `1315`
- support expansion added faces: `0`
- candidate faces after expansion: `1315`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1315`
- fit samples: `696342`
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
- surface multiscale prior blended bins: `29952`
- surface multiscale prior blended-bin fraction: `0.088973`
- surface multiscale prior mean blend weight: `0.089746`
- surface multiscale prior gate rejected bins: `269156`
- surface multiscale prior empty-bin rejects: `218778`
- surface multiscale prior sign rejects: `259201`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `49263`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `9669`
- view-conditioned basis supported-bin fraction: `0.028722`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `147`
- teacher-distilled basis supported-face fraction: `0.111787`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `237698`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.924661`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `75336`
- local alpha fallback bins: `67144`
- face gain guard enabled: `True`
- face gain guard decision: `reject_candidate_after_face_gain_guard`
- face gain guard allowed faces: `173`
- face gain guard rejected faces: `1142`
- face gain guard allowed sample fraction: `0.434212`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.047940`
- policy-val positive-view fraction: `0.833333`
- policy-val CVaR20 view relative gain: `-0.006631`
- policy-val min-view relative gain: `-0.021103`
- policy-val image SSIM gain: `-0.000039970`
- policy-val image SSIM positive-view fraction: `0.250000`
- policy-val image SSIM min-view gain: `-0.000174880`
- policy-val image L1 gain: `0.000009977`
- policy-val image L1 positive-view fraction: `0.916667`
- policy-val image L1 min-view gain: `-0.000002163`
- policy-val risk gate: `False`
- target written views: `35`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.006631 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.021103 < min_policy_val_min_view_relative_gain -0.000001; ssim_gain -0.000039970 < min_policy_val_ssim_mean_gain -0.000000100; ssim_positive_view_fraction 0.250000 < min_policy_val_ssim_positive_view_fraction 0.550000; ssim_min_view_gain -0.000174880 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000002163 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_gain -0.000039970 < min_policy_val_effective_ssim_gain 0.000010000; effective_ssim_cvar20_view_gain -0.000142097 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000000105 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00668244 | 0.00668244 |
| 0.0020 | 0.000300 | 1.000000 | 0.000130 | 0.000102 | 0.000000522 | 1.000000 | 0.000000119 | 0.000000052 | 1.000000 | 0.000000007 | 0.00668244 | 0.00668044 |
| 0.0039 | 0.000598 | 1.000000 | 0.000259 | 0.000201 | 0.000000998 | 1.000000 | 0.000000238 | 0.000000103 | 1.000000 | 0.000000011 | 0.00668244 | 0.00667845 |
| 0.0078 | 0.001193 | 1.000000 | 0.000514 | 0.000397 | 0.000002007 | 1.000000 | 0.000000477 | 0.000000205 | 1.000000 | 0.000000022 | 0.00668244 | 0.00667447 |
| 0.0156 | 0.002372 | 1.000000 | 0.001012 | 0.000769 | 0.000003924 | 1.000000 | 0.000000954 | 0.000000409 | 1.000000 | 0.000000045 | 0.00668244 | 0.00666659 |
| 0.0312 | 0.004687 | 1.000000 | 0.001956 | 0.001440 | 0.000007515 | 1.000000 | 0.000001729 | 0.000000811 | 1.000000 | 0.000000086 | 0.00668244 | 0.00665112 |
| 0.0625 | 0.009145 | 1.000000 | 0.003644 | 0.002491 | 0.000013560 | 1.000000 | 0.000002384 | 0.000001594 | 1.000000 | 0.000000164 | 0.00668244 | 0.00662133 |
| 0.1250 | 0.017376 | 1.000000 | 0.006215 | 0.003422 | 0.000021423 | 0.916667 | -0.000002623 | 0.000003097 | 1.000000 | 0.000000285 | 0.00668244 | 0.00656633 |
| 0.2500 | 0.031105 | 1.000000 | 0.008174 | 0.000678 | 0.000020633 | 0.833333 | -0.000038922 | 0.000005793 | 1.000000 | 0.000000410 | 0.00668244 | 0.00647459 |
| 0.5000 | 0.047940 | 0.833333 | -0.006631 | -0.021103 | -0.000039970 | 0.250000 | -0.000174880 | 0.000009977 | 0.916667 | -0.000002163 | 0.00668244 | 0.00636209 |
