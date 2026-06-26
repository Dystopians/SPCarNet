# Surface Residual Region Texture Adapter Audit

- accepted: `True`
- source model: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- fit evidence: `outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1`
- target evidence: `/dev/shm/peilincai_spcarnet_vnext_face_softshrink_bonsai_strict_20260626_052500_bonsai_strict/bonsai/target_evidence_no_gt`
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
- local alpha uncertainty-shrink mean: `0.938375`
- local alpha uncertainty-shrink downweighted bins: `8192`
- local alpha uncertainty-shrink upweighted bins: `0`
- local alpha candidate bins: `50650`
- local alpha fallback bins: `42458`
- face gain guard enabled: `True`
- face gain guard decision: `keep_face_gain_guard`
- face gain guard allowed faces: `69`
- face gain guard rejected faces: `1041`
- face gain guard allowed sample fraction: `0.271304`
- bin uncertainty guard enabled: `False`
- bin uncertainty guard decision: `not_requested`
- bin uncertainty guard allowed bins: `0`
- bin uncertainty guard rejected bins: `0`
- bin uncertainty guard allowed faces: `0`
- bin uncertainty guard allowed sample fraction: `0.000000`
- policy-val relative gain: `0.016426`
- policy-val positive-view fraction: `1.000000`
- policy-val CVaR20 view relative gain: `0.008131`
- policy-val min-view relative gain: `0.003620`
- policy-val image SSIM gain: `0.000017370`
- policy-val image SSIM positive-view fraction: `1.000000`
- policy-val image SSIM min-view gain: `0.000001371`
- policy-val image L1 gain: `0.000003422`
- policy-val image L1 positive-view fraction: `0.916667`
- policy-val image L1 min-view gain: `-0.000000138`
- policy-val risk gate: `True`
- target written views: `37`
- target changed fraction: `0.001513`
- effective policy: `accepted_atlas`
- target coverage gate: `True`
- reject reason: ``

## Alpha Rows

| alpha | rel gain | pos view frac | cvar20 view gain | min view gain | ssim gain | ssim pos frac | ssim min gain | image L1 gain | L1 pos frac | L1 min gain | mse before | mse after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.000000000 | 0.000000 | 0.000000000 | 0.01079320 | 0.01079320 |
| 0.0020 | 0.000153 | 1.000000 | 0.000095 | 0.000068 | 0.000000179 | 1.000000 | 0.000000060 | 0.000000029 | 1.000000 | 0.000000002 | 0.01079320 | 0.01079154 |
| 0.0039 | 0.000306 | 1.000000 | 0.000189 | 0.000135 | 0.000000338 | 1.000000 | 0.000000119 | 0.000000057 | 1.000000 | 0.000000004 | 0.01079320 | 0.01078989 |
| 0.0078 | 0.000611 | 1.000000 | 0.000376 | 0.000267 | 0.000000710 | 1.000000 | 0.000000238 | 0.000000115 | 1.000000 | 0.000000007 | 0.01079320 | 0.01078660 |
| 0.0156 | 0.001216 | 1.000000 | 0.000743 | 0.000524 | 0.000001445 | 1.000000 | 0.000000536 | 0.000000231 | 1.000000 | 0.000000016 | 0.01079320 | 0.01078008 |
| 0.0312 | 0.002406 | 1.000000 | 0.001455 | 0.001008 | 0.000002831 | 1.000000 | 0.000000954 | 0.000000461 | 1.000000 | 0.000000030 | 0.01079320 | 0.01076723 |
| 0.0625 | 0.004711 | 1.000000 | 0.002784 | 0.001857 | 0.000005489 | 1.000000 | 0.000001788 | 0.000000915 | 1.000000 | 0.000000047 | 0.01079320 | 0.01074235 |
| 0.1250 | 0.009019 | 1.000000 | 0.005068 | 0.003079 | 0.000010222 | 1.000000 | 0.000002980 | 0.000001794 | 1.000000 | 0.000000043 | 0.01079320 | 0.01069586 |
| 0.2500 | 0.016426 | 1.000000 | 0.008131 | 0.003620 | 0.000017370 | 1.000000 | 0.000001371 | 0.000003422 | 0.916667 | -0.000000138 | 0.01079320 | 0.01061591 |
| 0.5000 | 0.026463 | 0.916667 | 0.008131 | -0.001966 | 0.000022888 | 0.833333 | -0.000019789 | 0.000006152 | 0.916667 | -0.000000958 | 0.01079320 | 0.01050757 |
