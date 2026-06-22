# 2026-05-22 AutoVisual Render-Region Filtered v1

Status: `COMPLETE_AS_INTERFACE_AND_SAFETY_FIX`, `NOT_COMPLETE_SCIENTIFICALLY`.

This stage converted the AutoVisual FaceLocal route from a raw candidate
generator into a fixed train-render verified policy, then fixed a serious
certification mismatch in the coupled selector. The final v1d validation is
legal and reproducible, but it still does not produce a non-noise visual or
metric gain over the Phase-J fallback on `flowers` or `bonsai`.

## Motivation

The Render-Verified Carrier Materialization v3 run was safer than previous
carrier rows, but the accepted gains were still nearly invisible. The failure
was not simply missing evaluation plumbing: raw face-local carriers could pass
local proxy checks while producing no meaningful global render change.

This update therefore made AutoVisual stricter and more auditable:

1. Build a stronger scene-agnostic visual candidate plan.
2. Render raw train-side candidate regions.
3. Filter whole carriers before selector/materialization using train-render
   region evidence.
4. Let the coupled selector consume only the filtered plan.
5. Require a non-noise train-val selector threshold before promotion.
6. Preserve strict PatchCert certification during selector replay.

No held-out test metrics are used for selection.

## Implemented Interfaces

Updated:

```text
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py
scripts/car_model/ecsr_run_facelocal_coupled_selector.py
```

Key changes:

- Added `visual_medium` as the default fixed profile.
- `visual_medium` uses `delta_strength=0.30` and
  `delta_max_abs_rgb=0.028`.
- Added default stages: `plan,filter,selector`.
- The plan stage now emits a raw-base train render-region objective.
- The filter stage calls `ecsr_filter_facelocal_plan_by_render_region.py`.
- The selector stage consumes the filtered plan by default.
- Added selector non-noise thresholds:
  - `selector_min_trainval_psnr_gain=2e-5`
  - `selector_min_trainval_balanced_delta=5e-5`
  - `selector_tail_min_trainval_balanced_delta=5e-5`
- Fixed selector path logging for absolute experiment roots outside the repo.
- Fixed negative float CLI forwarding by using `--arg=value` syntax.
- Added a soft render-region fallback: if a carrier cannot be matched to a
  prebuilt render-region carrier and `--no-drop_unmapped` is used, proxy-positive
  unmapped carriers pass through instead of being incorrectly rejected by NaN
  region stats.
- Added certification-aware strict selector replay. Strict PatchCert carrier
  plans now default to `strictfull_s1`, replaying the complete certified plan
  with scale `1.0`, no face-id subset, and no alpha refit. The selector no
  longer silently runs invalid subset/scale/alpha trials on strict certified
  plans.

Validation:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py
```

Dry-run manifest:

```text
/tmp/ecsr_selector_strictsafe_dry_20260522/flowers/facelocal_coupled_selector.log
```

The dry-run confirms the strict path only emits `strictfull_s1`, keeps
`--delta_facelocal_materialize_plan_scale 1.0`, and does not pass
`--delta_facelocal_materialize_plan_face_ids` or alpha JSON.

## Experiment Timeline

### v1b hard render-region filter

Root:

```text
/data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai
```

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --scenes flowers,bonsai \
  --gpu 1 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai \
  --pipeline_label autovisual_visual_medium_filter_v1b_20260522 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_mode online \
  --continue_on_error \
  --force
```

Result:

| scene | plan accepted | filter kept rows | filter kept carriers | final selected | effective test dPSNR | effective test dSSIM | effective test dLPIPS |
|---|---:|---:|---:|---|---:|---:|---:|
| flowers | true | 0/3 | 0/1 | phasej fallback | +0.000000 | +0.000000 | +0.000000 |
| bonsai | true | 0/4 | 0/1 | phasej fallback | +0.000000 | +0.000000 | +0.000000 |

Diagnosis: the hard render-region filter over-rejected because plan carrier
IDs/faces did not overlap the prebuilt render-region carriers. The result was a
safe no-op, not an improvement.

### v1c soft unmapped-carrier fallback

Root:

```text
/data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1c_soft_20260522_flowers_bonsai
```

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --stages filter,selector \
  --scenes flowers,bonsai \
  --gpu 1 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1c_soft_20260522_flowers_bonsai \
  --plan_template /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai/candidate_plans/{scene}/facelocal_visual_candidate_plan.json \
  --evidence_root /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai/surface_evidence \
  --pipeline_label autovisual_visual_medium_filter_v1c_soft_20260522 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_mode online \
  --continue_on_error \
  --force \
  --no-filter_drop_unmapped
```

Result:

| scene | filter kept rows | filter kept carriers | selector outcome |
|---|---:|---:|---|
| flowers | 3/3 | 1/1 | failed during strict materialization |
| bonsai | 4/4 | 1/1 | failed during strict materialization |

Diagnosis: soft filtering correctly rescued proxy-positive unmapped carriers,
but the existing coupled selector then attempted old-style subset/scale/alpha
replay on strict PatchCert plans. The materializer rejected this correctly:
strict certified carriers cannot be row-sliced, rescaled without a render-trust
certificate, or alpha-refit without breaking the certificate.

### v1d strict-safe selector

Root:

```text
/data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1d_strictsafe_20260522_flowers_bonsai
```

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --stages filter,selector \
  --scenes flowers,bonsai \
  --gpu 7 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1d_strictsafe_20260522_flowers_bonsai \
  --plan_template /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai/candidate_plans/{scene}/facelocal_visual_candidate_plan.json \
  --evidence_root /data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1b_20260522_flowers_bonsai/surface_evidence \
  --pipeline_label autovisual_visual_medium_filter_v1d_strictsafe_20260522 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_mode online \
  --continue_on_error \
  --force \
  --no-filter_drop_unmapped
```

Result summary:

```text
/data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1d_strictsafe_20260522_flowers_bonsai/pipeline_summary.md
/data/peilincai/spcarnet_runs/autovisual_visual_medium_filter_v1d_strictsafe_20260522_flowers_bonsai/selector/coupled_selector_summary.md
```

| scene | strict replay | inner gate | selector pass | selected | train-val dPSNR | train-val dSSIM | train-val dLPIPS | train-val balanced | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| flowers | strictfull_s1 | true | false | phasej fallback | +0.000005722 | +0.000000000 | +0.000000209 | +0.000001550 | +0.000000000 | +0.000000000 | +0.000000000 |
| bonsai | strictfull_s1 | true | false | phasej fallback | +0.000007629 | +0.000000060 | -0.000000089 | +0.000010610 | +0.000000000 | +0.000000000 | +0.000000045 |

Selector rejection reasons:

| scene | selector reasons | tail notes |
|---|---|---|
| flowers | PSNR gain below `2e-5`; balanced below `5e-5` | PSNR/balanced tail thresholds failed; negative fractions too high |
| bonsai | PSNR gain below `2e-5`; balanced below `5e-5` | PSNR/balanced tail thresholds failed |

The final effective held-out delta is zero on both scenes because the selector
correctly falls back to Phase-J:

| scene | effective test dPSNR | effective test dSSIM | effective test dLPIPS |
|---|---:|---:|---:|
| flowers | +0.000000000 | +0.000000000 | +0.000000000 |
| bonsai | +0.000000000 | +0.000000000 | +0.000000000 |

## Conclusion

This stage fixes a real reliability flaw: strict PatchCert plans are now handled
by a certification-preserving selector path instead of invalid subset/scale/alpha
materialization. It also documents that hard render-region filtering was too
brittle and that soft proxy-positive fallback is necessary when carrier IDs do
not match render-region carriers.

Scientifically, v1d is still not enough. The full certified replay is almost a
no-op in image space: the inner gate accepts only `1e-6` to `1e-5` train-val
deltas, and the outer selector rejects both scenes. This is a safety/plumbing
milestone, not a paper-level improvement.

The next method step should not be another threshold sweep. The evidence points
to insufficient effect size from the current face-local residual representation.
The next serious attempt should change representation capacity or geometry
coupling while preserving the strict certificate semantics introduced here.

## 2026-06-20 Follow-Up

The next attempt is tracked here:

```text
docs/car_model/5-22-RepresentationField-Pivot-Log.md
```

It introduces `field_smoke` and `field_medium` profiles with shared residual
field and field-basis PatchCert replay. The first smoke validates that the new
representation path is non-empty and strict-selector compatible, but its
image-level gains remain noise-scale. The required next gate is a W&B-online
`field_medium` run on `flowers,bonsai`.
