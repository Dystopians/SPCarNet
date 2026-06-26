# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/stump/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_structure_shrink_stump_strict_20260626/stump/target_evidence_no_gt`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/carrier.json`
- support expansion mode: `none`
- support expansion base faces: `85`
- support expansion added faces: `0`
- candidate faces after expansion: `85`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `85`
- fit samples: `27077`
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
- surface multiscale prior blended bins: `3136`
- surface multiscale prior blended-bin fraction: `0.144118`
- surface multiscale prior mean blend weight: `0.083022`
- surface multiscale prior gate rejected bins: `17929`
- surface multiscale prior empty-bin rejects: `13657`
- surface multiscale prior sign rejects: `17592`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `3913`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `none`
- view-conditioned basis guard decision: `fallback_to_mean`
- view-conditioned basis supported bins: `243`
- view-conditioned basis supported-bin fraction: `0.011167`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `5`
- teacher-distilled basis supported-face fraction: `0.058824`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `17714`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `4300`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.975659`
- local alpha uncertainty-shrink downweighted bins: `4300`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `5103`
- local alpha fallback bins: `803`
- face gain guard enabled: `True`
- face gain guard decision: `reject_candidate_after_face_gain_guard`
- face gain guard allowed faces: `8`
- face gain guard rejected faces: `77`
- face gain guard allowed sample fraction: `0.448288`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `skipped_candidate_not_accepted`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.020314`
- policy-val positive-view fraction: `0.666667`
- policy-val CVaR20 view relative gain: `-0.172454`
- policy-val min-view relative gain: `-0.344907`
- policy-val image SSIM gain: `0.000000762`
- policy-val image SSIM positive-view fraction: `0.555556`
- policy-val image SSIM min-view gain: `-0.000001967`
- policy-val image L1 gain: `0.000000298`
- policy-val image L1 positive-view fraction: `0.666667`
- policy-val image L1 min-view gain: `-0.000000987`
- policy-val risk gate: `False`
- target written views: `16`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.172454 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.344907 < min_policy_val_min_view_relative_gain -0.000001`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00510390 | 0.00510390 |
| 0.0020 | 0.000105 | 0.777778 | 0.000000 | 0.000000 | -0.000000013 | 0.000000 | -0.000000060 | 0.000000001 | 0.222222 | -0.000000007 | 0.00510390 | 0.00510337 |
| 0.0039 | 0.000209 | 0.777778 | 0.000000 | 0.000000 | -0.000000013 | 0.000000 | -0.000000060 | 0.000000002 | 0.444444 | -0.000000007 | 0.00510390 | 0.00510283 |
| 0.0078 | 0.000418 | 0.777778 | 0.000000 | 0.000000 | -0.000000007 | 0.111111 | -0.000000060 | 0.000000005 | 0.555556 | -0.000000015 | 0.00510390 | 0.00510177 |
| 0.0156 | 0.000833 | 0.777778 | 0.000000 | 0.000000 | 0.000000013 | 0.333333 | -0.000000060 | 0.000000010 | 0.555556 | -0.000000022 | 0.00510390 | 0.00509965 |
| 0.0312 | 0.001654 | 0.666667 | -0.000153 | -0.000307 | 0.000000046 | 0.333333 | -0.000000060 | 0.000000020 | 0.555556 | -0.000000052 | 0.00510390 | 0.00509546 |
| 0.0625 | 0.003257 | 0.666667 | -0.001724 | -0.003447 | 0.000000099 | 0.555556 | -0.000000179 | 0.000000043 | 0.666667 | -0.000000108 | 0.00510390 | 0.00508728 |
| 0.1250 | 0.006309 | 0.666667 | -0.009114 | -0.018227 | 0.000000205 | 0.555556 | -0.000000417 | 0.000000082 | 0.666667 | -0.000000227 | 0.00510390 | 0.00507171 |
| 0.2500 | 0.011797 | 0.666667 | -0.040894 | -0.081788 | 0.000000417 | 0.555556 | -0.000000894 | 0.000000163 | 0.666667 | -0.000000466 | 0.00510390 | 0.00504369 |
| 0.5000 | 0.020314 | 0.666667 | -0.172454 | -0.344907 | 0.000000762 | 0.555556 | -0.000001967 | 0.000000298 | 0.666667 | -0.000000987 | 0.00510390 | 0.00500023 |
