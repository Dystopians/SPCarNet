# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200/bonsai/target_evidence_no_gt`
- region carrier: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_teacher_render_visible_region_carriers_v37_visible_alpha1_policyval_pruned.json`
- support expansion mode: `none`
- support expansion base faces: `1110`
- support expansion added faces: `0`
- candidate faces after expansion: `1110`
- target-support pre-rank enabled: `False`
- target-support pre-rank retained support candidates: `0`
- atlas faces: `1110`
- fit samples: `537599`
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
- surface multiscale prior blended bins: `24721`
- surface multiscale prior blended-bin fraction: `0.086997`
- surface multiscale prior mean blend weight: `0.084736`
- surface multiscale prior gate rejected bins: `231424`
- surface multiscale prior empty-bin rejects: `192776`
- surface multiscale prior sign rejects: `227242`
- surface multiscale prior variance rejects: `0`
- surface multiscale prior cosine rejects: `43177`
- view-conditioned basis mode: `normal_camera_linear`
- view-conditioned basis effective mode: `normal_camera_linear`
- view-conditioned basis guard decision: `keep_view_basis`
- view-conditioned basis supported bins: `7527`
- view-conditioned basis supported-bin fraction: `0.026489`
- view-conditioned basis OOD mode: `diag_z`
- view-conditioned basis OOD max-z: `2.5`
- view-conditioned basis OOD min-std: `0.05`
- teacher-distilled basis mode: `face_uv_patch_mixture_ridge`
- teacher-distilled basis effective mode: `none`
- teacher-distilled basis guard decision: `fallback_to_legacy`
- teacher-distilled basis supported faces: `112`
- teacher-distilled basis supported-face fraction: `0.100901`
- teacher-distilled basis apply mode: `blend`
- teacher-distilled basis blend: `0.5`
- policy-val enabled: `True`
- policy-val samples: `192419`
- selected alpha: `0.25`
- local alpha calibration: `True`
- local alpha mode: `policy_val_bin_uncertainty_shrink`
- local alpha fallback alpha: `0.000000`
- local alpha face count: `0`
- local alpha bin count: `0`
- local alpha bin RGB count: `0`
- local alpha uncertainty-shrink bin count: `8192`
- local alpha uncertainty-shrink policy mode: `keep_with_downweight`
- local alpha uncertainty-shrink mean: `0.924704`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `50650`
- local alpha fallback bins: `42458`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `68`
- face gain guard rejected faces: `1042`
- face gain guard allowed sample fraction: `0.266902`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.014840`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.007120`
- policy-val min-view relative gain: `0.001815`
- policy-val image SSIM gain: `0.000016049`
- policy-val image SSIM positive-view fraction: `0.916667`
- policy-val image SSIM min-view gain: `-0.000000119`
- policy-val image L1 gain: `0.000003021`
- policy-val image L1 positive-view fraction: `1.000000`
- policy-val image L1 min-view gain: `0.000000138`
- policy-val risk gate: `True`
- target written views: `37`
- target changed fraction: `0.001490`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.01079320 | 0.01079320 |
| 0.0020 | 0.000138 | 1.000000 | 0.000084 | 0.000051 | 0.000000154 | 1.000000 | 0.000000060 | 0.000000026 | 1.000000 | 0.000000002 | 0.01079320 | 0.01079171 |
| 0.0039 | 0.000276 | 1.000000 | 0.000167 | 0.000101 | 0.000000318 | 1.000000 | 0.000000119 | 0.000000051 | 1.000000 | 0.000000004 | 0.01079320 | 0.01079022 |
| 0.0078 | 0.000550 | 1.000000 | 0.000333 | 0.000200 | 0.000000666 | 1.000000 | 0.000000298 | 0.000000102 | 1.000000 | 0.000000009 | 0.01079320 | 0.01078726 |
| 0.0156 | 0.001095 | 1.000000 | 0.000659 | 0.000391 | 0.000001316 | 1.000000 | 0.000000596 | 0.000000205 | 1.000000 | 0.000000022 | 0.01079320 | 0.01078138 |
| 0.0312 | 0.002168 | 1.000000 | 0.001289 | 0.000745 | 0.000002598 | 1.000000 | 0.000001073 | 0.000000408 | 1.000000 | 0.000000041 | 0.01079320 | 0.01076980 |
| 0.0625 | 0.004246 | 1.000000 | 0.002464 | 0.001341 | 0.000005027 | 1.000000 | 0.000002086 | 0.000000810 | 1.000000 | 0.000000082 | 0.01079320 | 0.01074736 |
| 0.1250 | 0.008135 | 1.000000 | 0.004472 | 0.002091 | 0.000009368 | 1.000000 | 0.000003278 | 0.000001585 | 1.000000 | 0.000000151 | 0.01079320 | 0.01070539 |
| 0.2500 | 0.014840 | 1.000000 | 0.007120 | 0.001815 | 0.000016049 | 0.916667 | -0.000000119 | 0.000003021 | 1.000000 | 0.000000138 | 0.01079320 | 0.01063302 |
| 0.5000 | 0.024027 | 0.916667 | 0.007305 | -0.004905 | 0.000021577 | 0.916667 | -0.000021100 | 0.000005437 | 0.833333 | -0.000000378 | 0.01079320 | 0.01053387 |
