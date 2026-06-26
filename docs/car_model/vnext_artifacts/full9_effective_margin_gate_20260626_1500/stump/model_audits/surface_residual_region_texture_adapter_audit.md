# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/stump/ratio_0200/compact_model`
- fit evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/fit_evidence`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500/stump/target_evidence_no_gt`
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
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `243`
- view-conditioned basis supported-bin fraction: `0.011167`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis guard decision: `keep_teacher_basis`
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
- policy-val relative gain: `0.143583`
- policy-val positive-view fraction: `0.777778`
- policy-val CVaR20 view relative gain: `-0.997521`
- policy-val min-view relative gain: `-1.420808`
- policy-val image SSIM gain: `0.000005305`
- policy-val image SSIM positive-view fraction: `0.777778`
- policy-val image SSIM min-view gain: `-0.000031710`
- policy-val image L1 gain: `0.000002033`
- policy-val image L1 positive-view fraction: `0.777778`
- policy-val image L1 min-view gain: `-0.000004143`
- policy-val risk gate: `False`
- target written views: `16`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.997521 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -1.420808 < min_policy_val_min_view_relative_gain -0.000001; ssim_min_view_gain -0.000031710 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000004143 < min_policy_val_l1_min_view_gain -0.000001000; effective_ssim_gain 0.000005305 < min_policy_val_effective_ssim_gain 0.000010000; effective_ssim_cvar20_view_gain -0.000018597 < min_policy_val_effective_ssim_cvar20_gain 0.000001000; effective_image_l1_cvar20_view_gain -0.000002684 < min_policy_val_effective_l1_cvar20_gain 0.000000000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00510390 | 0.00510390 |
| 0.0020 | 0.000857 | 1.000000 | 0.000369 | 0.000327 | 0.000000026 | 0.555556 | -0.000000179 | 0.000000009 | 0.666667 | -0.000000015 | 0.00510390 | 0.00509953 |
| 0.0039 | 0.001712 | 1.000000 | 0.000736 | 0.000654 | 0.000000066 | 0.777778 | -0.000000238 | 0.000000018 | 0.666667 | -0.000000030 | 0.00510390 | 0.00509516 |
| 0.0078 | 0.003415 | 1.000000 | 0.001467 | 0.001303 | 0.000000086 | 0.777778 | -0.000000477 | 0.000000036 | 0.666667 | -0.000000060 | 0.00510390 | 0.00508647 |
| 0.0156 | 0.006792 | 1.000000 | 0.002916 | 0.002591 | 0.000000232 | 0.777778 | -0.000000894 | 0.000000071 | 0.777778 | -0.000000119 | 0.00510390 | 0.00506924 |
| 0.0312 | 0.013435 | 1.000000 | 0.005759 | 0.005121 | 0.000000457 | 0.777778 | -0.000001729 | 0.000000143 | 0.777778 | -0.000000224 | 0.00510390 | 0.00503533 |
| 0.0625 | 0.026273 | 1.000000 | 0.011225 | 0.009998 | 0.000000881 | 0.777778 | -0.000003457 | 0.000000283 | 0.777778 | -0.000000462 | 0.00510390 | 0.00496981 |
| 0.1250 | 0.050157 | 0.888889 | 0.007044 | -0.004932 | 0.000001682 | 0.777778 | -0.000007093 | 0.000000559 | 0.777778 | -0.000000969 | 0.00510390 | 0.00484791 |
| 0.2500 | 0.090754 | 0.777778 | -0.124591 | -0.243377 | 0.000003152 | 0.777778 | -0.000014663 | 0.000001083 | 0.777778 | -0.000001997 | 0.00510390 | 0.00464070 |
| 0.5000 | 0.143583 | 0.777778 | -0.997521 | -1.420808 | 0.000005305 | 0.777778 | -0.000031710 | 0.000002033 | 0.777778 | -0.000004143 | 0.00510390 | 0.00437107 |
