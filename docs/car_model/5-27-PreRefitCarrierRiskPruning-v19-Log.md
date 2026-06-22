# 5-27 Pre-Refit Carrier Risk Pruning v19 Log

## Context

The v18 medium bonsai run completed all 8 pipeline commands under:

`/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v18_20260622_bonsai_medium`

It was a clean negative result:

- profile: `field_region_render_risk_strict_v18`
- contract: `field_region_render_risk_strict_v18_objective_aware_bad_region_alpha_risk`
- candidate-owned refit decision: rejected
- rejection reasons: `balanced_delta_below_0`, `render_region_tail_cvar_below_-2e-05`
- train-val delta: PSNR `+0.0004844666`, SSIM `-0.0000111461`, LPIPS `+0.0000191480`, balanced `-0.0001214147`
- report-only test delta vs Phase-J fallback: PSNR `+0.0004978180`, SSIM `-0.0000063181`, LPIPS `+0.0000104606`, balanced `+0.0001622438`
- selector result: no accepted candidate, `phasej_fallback`

The v18 ELA objective-aware region-risk path was enabled, but it acted after refit/materialization and did not prevent unstable carriers from entering the local training stage. The final filter/selector had no usable candidate.

## v19 Method Change

v19 moves objective feedback earlier: before candidate-owned refit, it uses the train-only render-region objective to remove whole carrier face sets that are empirically harmful across their own changed regions.

Fixed profile:

`field_region_render_risk_strict_v19`

Contract:

`field_region_render_risk_strict_v19_objective_aware_pre_refit_carrier_risk_pruning`

The policy is scene-agnostic and pre-registered:

- a row is evaluable only if `crop_changed` is not false and `metrics_skipped_equal_crop` is false;
- a bad row has `core_balanced_delta < -0.001`, or the auxiliary pair `delta_core_ssim < -0.001` and `delta_core_lpips > 0.001`;
- a carrier is prune-eligible if bad rows exceed `0.5` of evaluable rows and at least one bad row exists;
- pruning is capped at `0.5` of the input face set, so the method cannot silently delete the whole candidate space;
- the exact decision is written to `candidate_owned_refit_plans/{scene}/pre_refit_risk_prune_report.json`.

This is a real pipeline change rather than a selector-threshold tweak: it changes which faces are allowed to participate in the candidate-owned refit itself.

## Implementation Validation

Compilation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  utils/evidence_lumigraph_adapter.py
```

Dry-run:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v19 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v19 \
  --pipeline_label dryrun_field_region_render_risk_strict_v19 \
  --wandb_mode online \
  --dry_run
```

Dry-run result:

- commands: `8`
- fixed profile: true
- override policy: forbidden
- manifest includes `pre_refit_risk_prune=true`
- candidate-owned refit command records the future prune-report path

Probe on v18 bonsai artifacts:

- input carrier count: `8`
- input face count: `411`
- prune-eligible bad carrier count: `3`
- removed carriers: `3`
- removed faces: `192`
- remaining faces: `219`
- removed face fraction: `0.4671532847`
- missing carrier rows: `0`
- unknown carrier rows: `0`

The probe shows the policy is strong enough to remove the worst carrier sets but stays under the fixed 50% deletion cap.

## Next Experiment

Run v19 medium bonsai with W&B enabled:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v19 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v19_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v19_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Decision gate:

- if v19 candidate-owned refit is still rejected, inspect `pre_refit_risk_prune_report.json`, refit render-region objective, and filter summary to decide whether the policy is over-pruning or whether the underlying candidate residual field remains too unstable;
- if v19 produces accepted candidates, expand to at least one outdoor scene before claiming progress.
