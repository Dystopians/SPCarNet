# Surface Residual Region Texture Adapter Audit

- accepted: `False`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1`
- target evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/garden`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `319`
- support expansion added faces: `0`
- candidate faces after expansion: `319`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `319`
- fit samples: `201676`
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
- surface multiscale prior blended bins: `8224`
- surface multiscale prior blended-bin fraction: `0.100705`
- surface multiscale prior mean blend weight: `0.088017`
- surface multiscale prior gate rejected bins: `65212`
- surface multiscale prior empty-bin rejects: `52208`
- surface multiscale prior sign rejects: `62947`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `14059`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `none`
- view-conditioned basis guard decision: `fallback_to_mean`
- view-conditioned basis supported bins: `3257`
- view-conditioned basis supported-bin fraction: `0.039883`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `53`
- teacher-distilled basis supported-face fraction: `0.166144`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `71583`
- selected alpha: `0.0`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.964724`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `19527`
- local alpha fallback bins: `11335`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `58`
- face gain guard rejected faces: `261`
- face gain guard allowed sample fraction: `0.557348`
- bin uncertainty guard enabled: `True`
- bin uncertainty guard decision: `reject_candidate_after_bin_uncertainty_guard`
- bin uncertainty guard allowed bins: `87`
- bin uncertainty guard rejected bins: `6209`
- bin uncertainty guard allowed faces: `13`
- bin uncertainty guard allowed sample fraction: `0.050288`
- policy-val relative gain: `0.002613`
- policy-val positive-view fraction: `0.833333`
- policy-val CVaR20 view relative gain: `-0.000741`
- policy-val min-view relative gain: `-0.002424`
- policy-val image SSIM gain: `0.000001838`
- policy-val image SSIM positive-view fraction: `0.916667`
- policy-val image SSIM min-view gain: `-0.000001550`
- policy-val image L1 gain: `0.000000106`
- policy-val image L1 positive-view fraction: `0.833333`
- policy-val image L1 min-view gain: `-0.000000387`
- policy-val risk gate: `False`
- target written views: `24`
- target changed fraction: `0.000000`
- effective policy: `fallback_noop`
- target coverage gate: `False`
- reject reason: `cvar20_view_relative_gain -0.000741 < min_policy_val_cvar20_relative_gain 0.000000; min_view_relative_gain -0.002424 < min_policy_val_min_view_relative_gain -0.000001`

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00366984 | 0.00366984 |
| 0.0020 | 0.000012 | 0.916667 | -0.000000 | -0.000006 | 0.000000010 | 0.166667 | 0.000000000 | 0.000000001 | 0.333333 | -0.000000002 | 0.00366984 | 0.00366979 |
| 0.0039 | 0.000024 | 0.916667 | -0.000001 | -0.000013 | 0.000000015 | 0.166667 | 0.000000000 | 0.000000002 | 0.416667 | -0.000000002 | 0.00366984 | 0.00366975 |
| 0.0078 | 0.000049 | 0.916667 | -0.000002 | -0.000026 | 0.000000025 | 0.333333 | 0.000000000 | 0.000000003 | 0.500000 | -0.000000004 | 0.00366984 | 0.00366966 |
| 0.0156 | 0.000097 | 0.916667 | -0.000003 | -0.000053 | 0.000000060 | 0.500000 | 0.000000000 | 0.000000005 | 0.750000 | -0.000000007 | 0.00366984 | 0.00366948 |
| 0.0312 | 0.000193 | 0.916667 | -0.000007 | -0.000108 | 0.000000129 | 0.666667 | -0.000000060 | 0.000000008 | 0.750000 | -0.000000019 | 0.00366984 | 0.00366913 |
| 0.0625 | 0.000382 | 0.916667 | -0.000018 | -0.000222 | 0.000000263 | 0.833333 | -0.000000119 | 0.000000015 | 0.833333 | -0.000000043 | 0.00366984 | 0.00366844 |
| 0.1250 | 0.000748 | 0.916667 | -0.000051 | -0.000468 | 0.000000531 | 0.750000 | -0.000000238 | 0.000000030 | 0.833333 | -0.000000086 | 0.00366984 | 0.00366709 |
| 0.2500 | 0.001433 | 0.916667 | -0.000189 | -0.001027 | 0.000000979 | 0.916667 | -0.000000656 | 0.000000058 | 0.833333 | -0.000000177 | 0.00366984 | 0.00366458 |
| 0.5000 | 0.002613 | 0.833333 | -0.000741 | -0.002424 | 0.000001838 | 0.916667 | -0.000001550 | 0.000000106 | 0.833333 | -0.000000387 | 0.00366984 | 0.00366025 |
