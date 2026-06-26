# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/room_teacher_surface_evidence_phasej_trainval_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_room_strict_20260626_052500_room_strict/room/target_evidence_no_gt`
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
- view-conditioned basis effective mode: `none`
- view-conditioned basis guard decision: `fallback_to_mean`
- view-conditioned basis supported bins: `18037`
- view-conditioned basis supported-bin fraction: `0.060739`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
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
- local alpha uncertainty-shrink mean: `0.926371`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `67040`
- local alpha fallback bins: `58848`
- face gain guard enabled: `True`
- face gain guard decision: `reject_candidate_after_face_gain_guard`
- face gain guard allowed faces: `216`
- face gain guard rejected faces: `944`
- face gain guard allowed sample fraction: `0.539758`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.056432`
- policy-val positive-view fraction: `0.750000`
- policy-val CVaR20 view relative gain: `-0.014189`
- policy-val min-view relative gain: `-0.031167`
- policy-val image SSIM gain: `0.000040670`
- policy-val image SSIM positive-view fraction: `0.750000`
- policy-val image SSIM min-view gain: `-0.000110984`
- policy-val image L1 gain: `0.000009008`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000007669`
- policy-val risk gate: `False`
- target written views: `39`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.014189 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.031167 < min_policy_val_min_view_relative_gain -0.000001; ssim_min_view_gain -0.000110984 < min_policy_val_ssim_min_view_gain -0.000010000; image_l1_min_view_gain -0.000007669 < min_policy_val_l1_min_view_gain -0.000001000`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00461406 | 0.00461406 |
| 0.0020 | 0.000305 | 0.916667 | 0.000048 | -0.000017 | 0.000000502 | 0.916667 | -0.000000179 | 0.000000047 | 0.833333 | -0.000000024 | 0.00461406 | 0.00461266 |
| 0.0039 | 0.000609 | 0.916667 | 0.000096 | -0.000035 | 0.000000979 | 0.916667 | -0.000000238 | 0.000000093 | 0.833333 | -0.000000050 | 0.00461406 | 0.00461125 |
| 0.0078 | 0.001215 | 0.916667 | 0.000190 | -0.000072 | 0.000001972 | 0.916667 | -0.000000417 | 0.000000187 | 0.833333 | -0.000000101 | 0.00461406 | 0.00460846 |
| 0.0156 | 0.002420 | 0.916667 | 0.000369 | -0.000158 | 0.000003874 | 0.916667 | -0.000000834 | 0.000000372 | 0.833333 | -0.000000203 | 0.00461406 | 0.00460290 |
| 0.0312 | 0.004798 | 0.916667 | 0.000698 | -0.000368 | 0.000007575 | 0.916667 | -0.000001550 | 0.000000738 | 0.833333 | -0.000000408 | 0.00461406 | 0.00459193 |
| 0.0625 | 0.009427 | 0.916667 | 0.001230 | -0.000947 | 0.000014424 | 0.916667 | -0.000003278 | 0.000001448 | 0.833333 | -0.000000838 | 0.00461406 | 0.00457057 |
| 0.1250 | 0.018175 | 0.916667 | 0.001801 | -0.002736 | 0.000026062 | 0.833333 | -0.000006914 | 0.000002795 | 0.833333 | -0.000001747 | 0.00461406 | 0.00453020 |
| 0.2500 | 0.033639 | 0.916667 | 0.000623 | -0.008843 | 0.000041127 | 0.833333 | -0.000016272 | 0.000005223 | 0.833333 | -0.000003638 | 0.00461406 | 0.00445885 |
| 0.5000 | 0.056432 | 0.750000 | -0.014189 | -0.031167 | 0.000040670 | 0.750000 | -0.000110984 | 0.000009008 | 0.833333 | -0.000007669 | 0.00461406 | 0.00435368 |
