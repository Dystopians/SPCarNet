# Teacher Surface Evidence Cache Augmentation

- base evidence dir: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/train_visible_bary_base/kitchen`
- teacher render dir: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model/train/ours_26000_phasej_trainval_gate/renders`
- parent source: `npz:rgb_render`
- processed views: `46`
- skipped views: `0`
- mean active fraction: `0.343699`
- mean target L1: `0.008766`
- selection mode: `better_masked_residual`
- mask target: `True`

Fields written:

- `teacher_residual_rgb`
- `teacher_residual_l1`
- `teacher_residual_rgb_raw`
- `teacher_better_mask`
- `teacher_gain_l1`
- `teacher_parent_delta_l1`

Top-support rebuild:

- rows: `8192`
- csv: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence/top_residual_supports.csv`
- parent csv: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence/top_residual_supports_parent.csv`
