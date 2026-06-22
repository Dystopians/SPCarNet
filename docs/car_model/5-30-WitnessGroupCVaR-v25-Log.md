# 5-30 Witness-Group CVaR v25 Log

Status: `MEDIUM_VALIDATED_NEGATIVE_ON_BONSAI`

Date: 2026-06-22

## Method Interface

v25 adds a train-only witness-group CVaR regression objective inside the face-local residual fitter. It groups fitting samples by train view and group type:

- `full_view`: all fitting samples in the train view.
- `region_view`: core/context render-region samples in the train view.
- `bystander_view`: non-core/context bystander samples in the train view.

For each group, the fitter computes:

```text
regression = relu((after_mse - before_mse) / before_mse - margin)
witness_loss = mean(top_tail_fraction(regression_groups))
```

The loss is added to the residual fitting objective as:

```text
render_region_total += witness_constraint_weight * witness_loss
```

Default behavior is unchanged when `witness_constraint_weight=0`.

## Exposed Arguments

Fitter:

- `--witness_constraint_weight`
- `--witness_constraint_tail_fraction`
- `--witness_constraint_min_samples`
- `--witness_constraint_margin`
- `--witness_constraint_include_full_view` / `--no-witness_constraint_include_full_view`
- `--witness_constraint_include_region_view` / `--no-witness_constraint_include_region_view`
- `--witness_constraint_include_bystander_view` / `--no-witness_constraint_include_bystander_view`

PhaseK/autovisual expose the same fields with the `delta_` prefix.

## Fixed v25 Profile

`field_region_render_risk_strict_v25` inherits v24 and sets:

- `delta_witness_constraint_weight=0.25`
- `delta_witness_constraint_tail_fraction=0.25`
- `delta_witness_constraint_min_samples=16`
- `delta_witness_constraint_margin=0.0`
- all witness group types enabled

Contract id:

```text
field_region_render_risk_strict_v25_train_objective_witness_group_cvar
```

## Validation

Commands run:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Result: passed.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py --help | rg "witness_constraint"
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py --help | rg "delta_witness_constraint"
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py --help | rg "field_region_render_risk_strict_v25|delta_witness_constraint"
```

Result: all expected arguments/profile entries appeared.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v25 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v25_witness_worker \
  --pipeline_label dryrun_field_region_render_risk_strict_v25_witness_worker \
  --wandb_mode online \
  --dry_run \
  --force
```

Result: passed, wrote 8 dry-run commands.

Manifest checks:

- `profile_contract_id=field_region_render_risk_strict_v25_train_objective_witness_group_cvar`
- `fixed_profile=true`
- resolved witness weight/tail/min/margin: `0.25 / 0.25 / 16 / 0.0`
- plan command contains all `--delta_witness_constraint_*` flags.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v25 \
  --scenes bonsai \
  --delta_witness_constraint_weight 0.1 \
  --dry_run
```

Result: failed as expected with `profile field_region_render_risk_strict_v25 is fixed; remove profile-field overrides: delta_witness_constraint_weight`.

## Caveats

This worker did not run long GPU experiments. v25 is a real train-objective implementation and has only been smoke/dry-run validated here. Medium/full W&B validation is still required before claiming metric improvement over v24 or MeshSplatting.

## 2026-06-22 Medium Validation Attempts

Primary run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v25_witness_20260622_bonsai_medium
```

Initial GPU2 run failed during surface-evidence construction because the GPU
had only about 650 MB free after resident jobs and the renderer needed another
694 MB allocation. This is a resource failure, not a method result:

```text
torch.OutOfMemoryError: CUDA out of memory
no evidence views rendered for split=train
skipped_failures=255
```

The same run was retried after hardlinking the already-complete, same-scene,
same-baseline v24 surface evidence cache into the v25 root. The fitter then
completed and wrote a valid v25 audit:

```text
audit:
  accepted: true
  policy_pass: true
  selected_faces: 5790
  accepted_faces: 169
  render_region_objective.witness_groups: 19
  render_region_objective.witness_group_counts:
    full_view: 5
    region_view: 5
    bystander_view: 9
  render_region_objective.witness_sample_counts:
    full_view: 85666
    region_view: 2375
    bystander_view: 83291
  witness_constraint_weight: 0.25
  witness_constraint_tail_fraction: 0.25
  witness_constraint_min_samples: 16
  final_witness_constraint_loss: 0.0
```

That GPU2 retry still failed later in render evidence maps because the
post-fit render process had insufficient remaining GPU memory. To avoid mixing
resource failure with method evidence, a clean GPU4 retry was started with a
fresh output root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v25_witness_20260622_bonsai_medium_gpu4
```

GPU4 retry command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v25 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v25_witness_20260622_bonsai_medium_gpu4 \
  --pipeline_label field_region_render_risk_strict_v25_witness_20260622_bonsai_medium_gpu4 \
  --wandb_mode online \
  --force
```

Status at this log update: GPU4 run has confirmed evidence-cache reuse and the
actual command includes all `--witness_constraint_*` flags. It has progressed
past fitting and into render evidence maps for:

```text
ours_26000_field_region_render_risk_strict_v25_witness_20260622_bonsai_medium_gpu4_plan_base
```

No metric claim should be made until this run finishes render, trainval gate,
and selector.

Current interpretation:

- v25 is a genuine train-objective method change, not a parameter scan.
- It directly targets the v24 bottleneck: mean local gains hiding tail-view
  regressions.
- The first full-fit audit proves that the witness groups are populated and
  active, but the method is not yet validated by fair metrics because the first
  GPU2 run was resource-blocked after fitting.

## 2026-06-22 GPU4 Medium Result

Final root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v25_witness_20260622_bonsai_medium_gpu4
```

The GPU4 run completed the substantive plan, filter, candidate-owned refit,
and selector work with W&B online logging. The top-level pipeline process then
failed while writing the final command manifest because an output metadata
field containing a comma-separated face-id allowlist was treated as a filesystem
path. The method artifacts and selector decisions were already written; the
manifest writer has been hardened to skip obvious non-path metadata strings.

Plan-generation decision:

```text
accepted: false
selected_label: phasej_guarded_adaptedge
decision_reasons: balanced_delta_below_0
trainval_balanced_delta: -0.00028133392333984375
trainval_delta:
  LPIPS: +0.000021338462829589844
  PSNR:  +0.0004482269287109375
  SSIM:  -0.000015139579772949219
report-only test delta:
  LPIPS: +0.000010818243026733398
  PSNR:  +0.0005474090576171875
  SSIM:  -0.000007510185241699219
```

Candidate-owned refit decision:

```text
accepted: false
selected_label: phasej_guarded_adaptedge
decision_reasons:
  - balanced_delta_below_0
  - render_region_tail_cvar_below_-2e-05
trainval_balanced_delta: -0.0004203915596008301
trainval_delta:
  LPIPS: +0.00003679096698760986
  PSNR:  +0.000659942626953125
  SSIM:  -0.00001722574234008789
report-only test delta:
  LPIPS: +0.00013685226440429688
  PSNR:  -0.0026493072509765625
  SSIM:  -0.00007390975952148438
```

Selector strict replay:

```text
trial: strictfull_s1
accepted: false
decision_reasons: balanced_delta_below_0
trainval_balanced_delta: -0.00012302398681640625
trainval_delta:
  LPIPS: +0.000006496906280517578
  PSNR:  +0.0000629425048828125
  SSIM:  -0.0000028014183044433594
report-only test delta:
  LPIPS: +0.000008732080459594727
  PSNR:  +0.00006866455078125
  SSIM:  -0.0000025033950805664062
```

Outer selector:

```text
accepted: false
selected_trial: phasej_fallback
candidate_count: 20
selected_trainval_balanced_delta: 0.0
effective_report_only_test_delta:
  LPIPS: 0.0
  PSNR:  0.0
  SSIM:  0.0
selection_uses_test: false
```

Interpretation:

- v25 proves that a witness-group CVaR term can be wired into the real fitter
  and can populate full-view, region-view, and bystander-view groups.
- It does not solve the performance bottleneck. Full-frame LPIPS/SSIM
  regressions remain large enough to make trainval balanced negative.
- The candidate-owned path is worse than the broad plan because its local ROI
  repair still has unsafe tail behavior.
- This result should be archived as a negative but useful method audit. The
  next serious method should change the representation or rendering path, not
  simply add another threshold around the same face-local SH1 residual.
