# 2026-05-12 Subagent Paper-Loop Continuation Report

## Status

Final status for this continuation: `NOT COMPLETE`.

This continuation did complete a real implementation/evidence milestone:

- train/eval code now supports deployable SPCarNet `visible_only` rescoring;
- Phase-S face-local SH1 now supports validation shrink, all-train fold
  consistency logging, and patch-certified local growth;
- W&B-logged medium/long validations were run for the remaining Phase-S v1
  scenes and the new `bicycle` patch-certified follow-up;
- the existing Stage ELA12 clean-best collector was rerun with W&B to verify the
  selected-clean audit status;
- metrics, qualitative output paths, commands, and known weaknesses are
  documented.

It did not complete the paper-level goal. The remaining blocker is scientific,
not only engineering: the representation-level edits are too low-amplitude and
do not robustly fix `bicycle`, `counter`, or `treehill`; the clean-best
MeshSplatting reconciliation is positive on the existing five-scene selected
audit but still does not cover the full nine-scene Mip-NeRF360 setting.

## Code Changes

Changed files:

- `scripts/car_model/rescore_spcarnet_multihypothesis.py`
- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

Main method changes:

- SPCarNet adds `visible_only` and `visible_rag_sym` deployable selectors.
- The rescoring script now fails nonzero when a requested selector has no
  eligible evidence fields, preventing silent NaN wins.
- Phase-S face-local SH1 supports train-only validation shrink.
- Phase-S records all-train fold-consistency diagnostics with explicit wording
  that this is not independent cross-fit.
- Phase-S adds patch-certified growth around accepted seed faces, including a
  centroid-neighbor mode for disconnected but spatially adjacent evidence.

## Completed Experiments

### SPCarNet selector

Output:

- `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_visible_rescored.json`

Key validation metrics, lower is better:

| selector | recon | hidden | free | visible |
|---|---:|---:|---:|---:|
| K=1/first contained candidate | 0.06786 | 0.10013 | 0.03643 | 0.06246 |
| K=8 `rag_sym` | 0.06700 | 0.09971 | 0.03546 | 0.06294 |
| K=8 `visible_only` | 0.06259 | 0.09425 | 0.03217 | 0.05592 |
| K=8 `visible_rag_sym` | 0.06426 | 0.09630 | 0.03353 | 0.05950 |
| K=8 oracle | 0.06132 | 0.09357 | 0.03114 | 0.05670 |

Conclusion: `visible_only` is the strongest current deployable selector on this
evidence. `rag_sym` remains an ablation, not the headline selector.

### Phase-S gaincert v1

Single-gate root:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512`

Strict multifold root:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512`

Strict train-val results:

| scene | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | report-only test dPSNR / dSSIM / dLPIPS |
|---|---:|---:|---:|---:|---|
| garden | true | +0.000520 | +0.000017 | -0.000082 | +0.000063 / +0.000001 / -0.000001 |
| flowers | true | +0.000030 | +0.000000 | +0.000000 | +0.001677 / +0.000158 / -0.000305 |
| bicycle | false | +0.000143 | +0.000001 | +0.000023 | +0.000374 / +0.000035 / -0.000115 |
| bonsai | true | +0.000156 | -0.000001 | +0.000020 | +0.000715 / +0.000016 / -0.000047 |
| kitchen | true | +0.000072 | +0.000000 | -0.000001 | +0.000084 / +0.000000 / -0.000001 |
| room | true | +0.000051 | +0.000000 | -0.000000 | +0.000046 / +0.000000 / +0.000000 |
| stump | true | +0.000001 | -0.000000 | -0.000000 | +0.000000 / -0.000000 / +0.000000 |

`counter` and `treehill` are blocked at single-gate rejection. `bonsai` is
accepted by the configured tolerance gate, but is not an all-axis clean win.

Qualitative output:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/qualitative_gallery/gallery.html`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/qualitative_gallery/selected_views.json`

### Phase-S ablations/follow-ups

v3 low-strength face-shrink root:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v3_faceshrink_lowstrength_cached_dense16_20260512`

Result: rejects `bicycle`, `counter`, and `treehill` at the single train-val
gate.

Topology-neighbor patch-cert root:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_patchcert_v4_cached_dense16_20260512`

Result: rejects `bicycle` at the single train-val gate.

Centroid patch-cert v2 root:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_patchcert_v4_centroid_v2_cached_dense16_20260512`
- strict JSON: `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_patchcert_v4_centroid_v2_cached_dense16_20260512/bicycle/multifold_trainval_gate.json`

Result: single-gate accepts `bicycle`, expanding from `7` seed faces to `48`
patch-certified faces, but strict four-offset rejects:

| offset | dPSNR | dSSIM | dLPIPS |
|---:|---:|---:|---:|
| 0 | +0.000038 | +0.000015 | -0.000028 |
| 1 | +0.000029 | -0.000000 | +0.000002 |
| 2 | -0.000017 | -0.000000 | -0.000000 |
| 3 | -0.000212 | -0.000079 | +0.000117 |
| mean | -0.000041 | -0.000016 | +0.000023 |

Conclusion: patch growth is a real representation attempt, but not a successful
paper method yet.

### Stage ELA12 clean-best audit rerun

Command:

```bash
WANDB_MODE=online PYTHONUNBUFFERED=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_collect_stageela12_fair_baseline_audit.py \
  --wandb \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group cleanbest_protocol_reconcile_20260512 \
  --wandb_name collect_stageela12_fair_baseline_audit_20260512
```

W&B run: `rmpikjz2`.

Report:

- `docs/car_model/stageELA12_fair_baseline_audit_report.md`
- `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/fair_baseline_audit.json`
- `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/qualitative_gallery/gallery.html`

Result: the selected-clean audit is ready and remains `5/5` strict full-pass on
the currently complete artifact set (`bonsai`, `courtyard`, `room`, `counter`,
and `parking_phone_tiny`). The report also states the limitation explicitly:
this is not the full nine-scene Mip-NeRF360 benchmark mean.

## Representative Commands

SPCarNet visible rescoring:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/rescore_spcarnet_multihypothesis.py \
  --variants visible_only visible_rag_sym \
  --wandb_project mesh-splatting-ecsr
```

Phase-S strict gate pattern:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene room \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/room/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/room/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_gaincert_v1_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_gaincert_v1_20260512_multifold \
  --offsets 0,1,2,3 \
  --iteration 26000 \
  --gpu 7 \
  --policy_holdout_fraction 0.25 \
  --calib_sampler uniform \
  --calib_max_views 32 \
  --calib_stride 1 \
  --alpha_feature_mode confidence_magnitude_edge \
  --alpha_default 0.0 \
  --gate_min_psnr_gain 0.0 \
  --gate_max_ssim_regression 0.00005 \
  --gate_max_lpips_regression 0.00015 \
  --wandb_project mesh-splatting-ecsr
```

## Review

Strengths:

- The implementation moved beyond parameter-only scanning.
- The `visible_only` selector is a clear object-side improvement.
- Phase-S now has train-only certificates and multi-offset validation across
  the completed positive scenes.
- Negative evidence is explicit rather than hidden by fallback wording.

Weaknesses:

- Phase-S image gains are extremely small and often visually invisible.
- `bicycle`, `counter`, and `treehill` are not solved.
- `bonsai` should not be presented as an all-axis strict win.
- Face-local SH1 increases vertices/attributes, so rate-distortion reporting
  must include more than triangle count.
- The clean-best baseline envelope is positive on the existing five-scene
  Stage ELA12 audit, but full nine-scene Mip-NeRF360 clean-best closure is still
  not available.

## Exact Next Step

Do not continue with local gain/patch scans first. The next command should
collect or render the missing full9 same-protocol clean-best rows, then compare
the current method under exactly the same scene/iteration/eval policy:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_collect_stageela12_fair_baseline_audit.py \
  --out-dir outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit \
  --report docs/car_model/stageELA12_fair_baseline_audit_report.md \
  --wandb \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group cleanbest_protocol_reconcile_20260512
```

If the collector still reports only five scenes, the next GPU work is to
generate the missing clean-best/method rows rather than tuning Phase-S.
