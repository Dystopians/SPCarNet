# Surface Residual Region Texture Adapter Audit

- accepted: `True`
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
- selected alpha: `0.0625`
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
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.006009`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.002396`
- policy-val min-view relative gain: `0.000258`
- policy-val image SSIM gain: `0.000001659`
- policy-val image SSIM positive-view fraction: `0.916667`
- policy-val image SSIM min-view gain: `0.000000000`
- policy-val image L1 gain: `0.000000179`
- policy-val image L1 positive-view fraction: `0.750000`
- policy-val image L1 min-view gain: `-0.000000330`
- policy-val risk gate: `True`
- target written views: `24`
- target changed fraction: `0.002080`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.00366984 | 0.00366984 |
| 0.0020 | 0.000199 | 1.000000 | 0.000087 | 0.000021 | 0.000000045 | 0.500000 | 0.000000000 | 0.000000006 | 0.750000 | -0.000000007 | 0.00366984 | 0.00366911 |
| 0.0039 | 0.000397 | 1.000000 | 0.000172 | 0.000041 | 0.000000124 | 0.750000 | 0.000000000 | 0.000000012 | 0.833333 | -0.000000019 | 0.00366984 | 0.00366838 |
| 0.0078 | 0.000791 | 1.000000 | 0.000342 | 0.000080 | 0.000000233 | 0.916667 | 0.000000000 | 0.000000023 | 0.750000 | -0.000000039 | 0.00366984 | 0.00366694 |
| 0.0156 | 0.001570 | 1.000000 | 0.000671 | 0.000146 | 0.000000437 | 0.916667 | 0.000000000 | 0.000000045 | 0.750000 | -0.000000078 | 0.00366984 | 0.00366408 |
| 0.0312 | 0.003095 | 1.000000 | 0.001295 | 0.000237 | 0.000000869 | 0.916667 | 0.000000000 | 0.000000092 | 0.750000 | -0.000000160 | 0.00366984 | 0.00365848 |
| 0.0625 | 0.006009 | 1.000000 | 0.002396 | 0.000258 | 0.000001659 | 0.916667 | 0.000000000 | 0.000000179 | 0.750000 | -0.000000330 | 0.00366984 | 0.00364778 |
| 0.1250 | 0.011293 | 0.916667 | 0.004017 | -0.000349 | 0.000003094 | 0.916667 | -0.000000119 | 0.000000347 | 0.750000 | -0.000000674 | 0.00366984 | 0.00362839 |
| 0.2500 | 0.019687 | 0.916667 | 0.003986 | -0.004159 | 0.000005171 | 0.916667 | -0.000000954 | 0.000000637 | 0.750000 | -0.000001421 | 0.00366984 | 0.00359759 |
| 0.5000 | 0.027777 | 0.833333 | -0.009041 | -0.022159 | 0.000006224 | 0.750000 | -0.000004828 | 0.000001015 | 0.750000 | -0.000003153 | 0.00366984 | 0.00356790 |
