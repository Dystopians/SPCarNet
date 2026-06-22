# 2026-06-21 Full-Frame Visible Carrier Preselection v16 Log

Status: `REAL_METHOD_AUDIT_CHANGE`, `REJECT_AS_PERFORMANCE_METHOD`,
`NOT_COMPLETE_SCIENTIFICALLY`.

This checkpoint tests whether the render-region candidate pipeline can stop
promoting locally strong but full-frame-invisible residual repairs. It is a
useful evidence-hygiene milestone, but it does not produce a paper-level
improvement over the MeshSplatting-derived Phase-J baseline.

## Implemented Method Change

v16 extends the train/eval pipeline with full-frame-visible carrier
preselection:

- candidate render-region construction records per-region full-frame support:
  `frame_pixels`, `bbox_pixels`, `visible_frame_fraction`,
  `bbox_frame_fraction`, `residual_mass`, and `residual_mass_fraction`;
- carrier ranking can use train-only full-frame residual support through
  `--frame_aware_ranking`;
- frame-aware carrier filtering exposes:
  `--min_frame_support_fraction`, `--min_residual_mass_fraction`, and
  `--max_carriers`;
- aggregate subset selection can prefer full-frame visibility through
  `--prefer_full_frame_visibility`;
- fixed profile:
  `field_region_render_risk_strict_v16`;
- fixed contract:
  `field_region_render_risk_strict_v16_full_frame_visible_residual_carrier_preselection`.

Files changed:

```text
scripts/car_model/ecsr_build_candidate_plan_render_regions.py
scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py
scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_build_candidate_plan_render_regions.py \
  scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py \
  scripts/car_model/ecsr_apply_render_cvar_aggregate_subset.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Dry-run validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v16 \
  --stages filter,selector \
  --scenes bonsai \
  --gpu -1 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_dryrun \
  --pipeline_label field_region_render_risk_strict_v16_20260621_dryrun \
  --dry_run \
  --wandb_mode offline
```

The dry-run manifest confirmed:

```text
frame_aware_ranking: true
min_frame_support_fraction: 1.0e-05
min_residual_mass_fraction: 1.0e-07
max_carriers: 8
aggregate_subset_prefer_full_frame_visibility: true
```

## Builder Replay Evidence

Builder replay on existing v11 raw plans:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_builder_replay
```

Results:

| scene | selected carriers | regions | rejected frame-aware carriers | reading |
|---|---:|---:|---:|---|
| flowers | 4 | 26 | 0 | frame-aware records are populated |
| bonsai | 8 | 57 | 11 | broad carrier pool is capped by top-8 full-frame support |

This replay proves the frame-aware builder path is active. It does not prove
metric improvement.

## W&B-Online Medium Validation

Run root:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium
```

Main command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v16 \
  --stages filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium \
  --wandb_mode online \
  --force
```

The earlier GPU3 attempt failed during bonsai render-evidence evaluation due to
CUDA OOM. The GPU5 rerun completed all seven stages:

```text
candidate_regions: 0
candidate_region_eval: 0
candidate_owned_refit: 0
candidate_region_eval_refit: 0
filter: 0
aggregate_subset: 0
selector: 0
```

An earlier plan-stage rerun also required repairing a truncated cached bonsai
PNG under:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders/00085.png
```

The cache was regenerated with `meshsplatopt_render_evidence_maps.py`; all 292
train PNGs then loaded successfully.

## Plan-Stage Result

Decision paths:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/plan_generation/decisions/flowers_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/plan_generation/decisions/bonsai_decision.json
```

| scene | accepted | main rejection | test dPSNR | test dSSIM | test dLPIPS | reading |
|---|---:|---|---:|---:|---:|---|
| flowers | false | render-region changed fraction below 0.05 | +0.000001907 | -0.000000179 | +0.000000417 | weak/no-op-scale |
| bonsai | false | PSNR gain and balanced delta below 0 | -0.000185013 | -0.000001371 | -0.000003397 | LPIPS-only tiny gain with PSNR/SSIM loss |

## Candidate-Owned Refit Result

Decision paths:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/candidate_owned_refit/decisions/flowers_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/candidate_owned_refit/decisions/bonsai_decision.json
```

| scene | accepted | rejection reasons | train-val balanced | test balanced | train dPSNR | train dSSIM | train dLPIPS | test dPSNR | test dSSIM | test dLPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | false | render-region tail CVaR below `-2e-05` | +0.000051737 | +0.000013471 | +0.000036240 | +0.000000358 | -0.000000417 | +0.000005722 | +0.000000000 | -0.000000387 |
| bonsai | false | PSNR below 0; balanced below 0; render-region tail CVaR below `-2e-05` | -0.000244498 | -0.000055015 | -0.000164032 | -0.000001907 | +0.000002116 | -0.000179291 | -0.000001252 | -0.000007465 |

The non-noop audits passed for both candidates, so the failures are not caused
by empty application. They are real metric/tail failures.

## Render-Region and Aggregate Evidence

Flowers filter/aggregate:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/filtered_candidate_plans/flowers/filter_summary.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/filtered_candidate_plans/flowers/aggregate_subset_summary.md
```

Key flowers numbers:

```text
filter input rows / carriers: 10 / 1
filter kept rows / carriers: 10 / 1
render-region mean balanced: +0.017368886
render-region tail balanced: -0.002410293
full-frame visibility adjusted delta: +0.000000869
full-frame changed fraction: 0.000090788
aggregate selected rows / carriers: 0 / 0
aggregate reason: selected_carriers_below_1
```

Bonsai filter/aggregate:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/filtered_candidate_plans/bonsai/filter_summary.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/filtered_candidate_plans/bonsai/aggregate_subset_summary.md
```

Key bonsai numbers:

```text
filter input rows / carriers: 138 / 14
filter kept rows / carriers: 39 / 4
aggregate hard-pool carriers: 4
aggregate selected rows / carriers: 0 / 0
final aggregate reason: full_frame_changed_pixel_fraction_below_2e-05
best full-frame visibility values: about 3.7411e-05 to 4.3668e-05
best full-frame changed fractions: about 9.6102e-05 to 1.33720e-04
```

Bonsai still has the recurring failure pattern: large ROI mean gains coexist
with severe negative tails and weak full-frame stability.

## Final Selector Result

Selector paths:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/selector/coupled_selector_summary.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v16_20260621_flowers_bonsai_medium/selector/bonsai/coupled_selector_decision.json
```

Final selector:

```text
scenes with plan candidates: 0
accepted scenes: 0
bonsai selected trial: phasej_fallback
bonsai rejection: no_plan_candidates
effective held-out deltas after fallback: zero
```

## Interpretation

v16 should be retained as an audit and safety layer, not as the main scientific
claim.

What it fixes:

- candidate generation now records whether a residual repair has full-frame
  visibility support;
- aggregate selection can prefer candidates with visible full-frame effect;
- the pipeline no longer lets locally attractive ROI rows silently become a
  promoted full-frame claim.

What it does not fix:

- local render-region gains are still too diluted to move full-frame metrics;
- bonsai has strong mean ROI gains but unstable negative tails;
- flowers has tiny positive full-frame metrics, but fails tail safety and is
  visually too subtle;
- selector correctly falls back to Phase-J, so v16 has no effective promoted
  gain over the baseline.

## Next Required Move

The next work should be v17, not another carrier-threshold sweep. The method
needs a real risk-adaptive materialization mechanism:

- calibrate residual strength per carrier/bin using train-only holdout
  evidence;
- shrink or zero carriers whose holdout tail is negative even if their mean ROI
  is positive;
- preserve full-frame-visible support while preventing the high-tail-risk
  carriers from entering Phase-K;
- validate on `flowers,bonsai` first, then expand only if the train-val
  non-noise gate passes without held-out-test selection.

Until v17 or a similar upstream representation/materialization change produces
nonzero stable train-val gains, this line remains:

```text
NOT_COMPLETE_SCIENTIFICALLY
NO_PAPER_LEVEL_BASELINE_DOMINATION
```
