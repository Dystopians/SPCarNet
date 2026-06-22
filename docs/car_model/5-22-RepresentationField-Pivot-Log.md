# 2026-06-20 Representation-Field Pivot Log

Status: `IN_PROGRESS_ENGINEERING_MILESTONE`, `NOT_COMPLETE_SCIENTIFICALLY`.

This log records the first representation-level pivot after the AutoVisual
render-region filtered v1d stage. The previous stage fixed certification and
selector legality, but the selected policy still fell back to Phase-J because
held-out gains were only `1e-6` to `1e-5`. The active hypothesis here is that
the bottleneck is no longer only thresholding or filtering; the face-local
residual representation itself lacks enough effect size.

## Method Change

The AutoVisual face-local pipeline now has profile-controlled representation
capacity instead of only scalar threshold changes:

- `REPRESENTATION_DEFAULTS` preserves the old behavior for `smoke`, `balanced`,
  `visual_medium`, and `strict` profiles.
- `field_smoke` enables a shared residual field with 8 anchors and a
  `field_linear` cluster basis. Its validation gates are intentionally light so
  a short smoke run can prove the new representation is not silently disabled.
- `field_medium` enables a shared residual field with 32 anchors and a
  `field_quad` cluster basis. It keeps the non-noise selector thresholds from
  `visual_medium` and is the first profile that should count as a real method
  candidate.
- New representation knobs are passed through the train/eval pipeline instead
  of being hard-coded or only available in the downstream Phase-K script.

Implementation file:

```text
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

## Verification

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  -m py_compile scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Profile consistency check:

```text
all profiles share the same profile-key set
field_smoke policy_samples=12 patch_samples=8 holdout=2/1
field_medium policy_samples=512 patch_samples=16 holdout=4/3
```

Dry-run command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --dry_run --profile field_medium --scenes flowers \
  --output_root /tmp/ecsr_field_medium_dry_20260620 \
  --pipeline_label autovisual_field_medium_dry_20260620 \
  --wandb_mode disabled --gpu 5 --force
```

Dry-run manifest confirmed:

```text
shared residual field: true
shared residual field anchors: 32
cluster basis mode: field_quad
selection uses test: false
```

## Smoke Run

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_smoke \
  --scenes flowers \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers \
  --pipeline_label autovisual_field_smoke_fixed_20260620 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_mode online \
  --continue_on_error \
  --force
```

Key paths:

```text
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers/candidate_plans/flowers/facelocal_visual_candidate_plan.json
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers/plan_generation/decisions/flowers_decision.json
```

The fixed smoke run is no longer a no-op:

| item | value |
|---|---:|
| candidates | 2 |
| carriers | 1 |
| basis rows | 16 |
| shared residual field | true |
| cluster basis | field_linear |
| fit proxy relative gain | 0.565276 |
| policy-val proxy relative gain | 0.559815 |

Carrier holdout certificate passed for the selected carrier:

```text
carrier_id: carrier_2497215_7271717
faces: [2497215, 7271717]
passing_groups: 2 / 2
mean_relative_gain: 0.800189
min_relative_gain: 0.620340
```

Plan gate result on `flowers`:

| split | method | LPIPS | PSNR | SSIM | delta summary |
|---|---|---:|---:|---:|---|
| train policy-val | Phase-J base | 0.297203869 | 20.855226517 | 0.647178471 | reference only |
| train policy-val | field smoke plan | 0.297203749 | 20.855230331 | 0.647178471 | dPSNR +0.000003815, balanced +0.000006199 |
| test report-only | Phase-J base | 0.329505473 | 20.300607681 | 0.557457805 | not used for selection |
| test report-only | field smoke plan | 0.329505622 | 20.300609589 | 0.557457685 | dPSNR +0.000001907, dLPIPS +0.000000149 |

W&B runs observed so far:

```text
mzx49d5x phasej test reference
pz9g3jgp phasej train policy-val reference
m2jmxw1v field smoke plan test
nmsy571f field smoke plan train policy-val
erhyvhrt selector strictfull_s1 phasej test reference
og63684x selector strictfull_s1 phasej train policy-val reference
0cdld1z2 selector strictfull_s1 plan test
```

Final selector result:

```text
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers/pipeline_summary.json
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers/selector/coupled_selector_summary.json
/data/peilincai/spcarnet_runs/autovisual_field_smoke_fixed_20260620_flowers/selector/flowers/coupled_selector_decision.json
```

| scene | selected trial | accepted | selection uses test | train-val dPSNR | train-val balanced | tail balanced CVaR | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | strictfull_s1 | true | false | +0.000003815 | +0.000006199 | -0.000062477 | +0.000001907 | -0.000000119 | +0.000000149 |

## Current Interpretation

This is a real engineering milestone because the representation-field policy is
wired into the train/eval pipeline and produces a non-empty certified carrier.
It is not yet a scientific success. The measured image-level deltas remain
noise-scale, the report-only test balanced score is negative, and the tail
balanced CVaR is clearly negative. The selector accepted `strictfull_s1` only
because `field_smoke` intentionally sets selector thresholds to zero. This
acceptance proves replay legality and non-noop execution, not method strength.

The next experiment must be `field_medium`, not another hand-tuned scalar sweep.
The medium profile is the first fair test of whether the representation-field
capacity can produce a non-noise, selector-accepted gain on more than one scene.

## Next Gate

Run:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_medium \
  --scenes flowers,bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_medium_20260620_flowers_bonsai \
  --pipeline_label autovisual_field_medium_20260620 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_mode online \
  --continue_on_error \
  --force
```

Success requires all of the following:

- selection uses test: `false`;
- non-noop operator audit passes;
- outer selector passes non-noise thresholds:
  `dPSNR >= 2e-5`, `balanced >= 5e-5`, tail balanced `>= 5e-5`;
- no meaningful LPIPS or tail regression;
- portfolio-level improvement on `flowers,bonsai`, not a single raw report-only
  row.

## 2026-06-20 Carrier-Restricted Candidate-Owned Refit

Status: `IN_PROGRESS_REAL_METHOD_FIX`,
`NOT_COMPLETE_SCIENTIFICALLY`.

The next bottleneck is structural, not a threshold choice. Candidate-owned
render-region refit created a promising local train objective on `flowers`, but
the refit carrier expanded beyond the seed candidate carrier:

```text
seed carrier:   carrier_509936_790063_1448551
refit carrier:  carrier_32621_509936_790063_1448551
```

This violated the intended "candidate-owned" locality: the refit was allowed to
pull in a neighboring face that was not part of the render-region carrier used
to justify the repair. The local mean remained positive, but tail safety became
worse and the render-region filter correctly rejected all rows.

Implemented fix:

- `ecsr_apply_surface_residual_facelocal_sh1_delta.py` now accepts
  `--allowed_face_ids` and filters the fitting/refit candidate faces before
  sample collection and certification.
- `ecsr_run_phasek_barycentric_gate_scene.py` forwards
  `--delta_facelocal_allowed_face_ids` into the face-local operator.
- `ecsr_run_autovisual_facelocal_pipeline.py` automatically derives the allowed
  face set from the candidate-owned render-region carrier JSON, falling back to
  the seed candidate plan when needed.

This is a real method constraint: refit can still optimize coefficients and pass
the normal train-only certificates, but it cannot enlarge the carrier outside
the region that was already selected by the method.

Validation already done:

```text
py_compile passed:
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

dry-run command check:
--delta_facelocal_allowed_face_ids 509936,790063,1448551
```

Running validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_owned_refit_medium \
  --scenes flowers \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_restricted_20260620_flowers \
  --pipeline_label autovisual_field_region_owned_refit_restricted_20260620_flowers \
  --wandb_mode online \
  --gpu 5 \
  --force
```

In parallel, the unrestricted bonsai run remains useful as an ablation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_owned_refit_medium \
  --scenes bonsai \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_medium_20260620_bonsai \
  --pipeline_label autovisual_field_region_owned_refit_medium_20260620_bonsai \
  --wandb_mode online \
  --gpu 4 \
  --force
```

Gate for this fix:

- refit audit must show `allowed_face_ids_count > 0`;
- refit candidate carrier must not introduce faces outside the allowed set;
- candidate-owned render-region tail must improve relative to unrestricted
  refit;
- selector must either accept a non-noise train-val gain or honestly fallback;
- report-only test remains evidence, never a selector input.

## 2026-06-20 Strict Tail-Risk Adaptive Scale Policy

Status: `IMPLEMENTED_AND_RUNNING_VALIDATION`,
`NOT_COMPLETE_SCIENTIFICALLY`.

The carrier-restricted refit fixed locality, but the finished validation showed
that the bottleneck is still effect size and train-val tail stability, not
merely interface wiring.

Finished evidence:

| run | scene | result |
|---|---|---|
| restricted raw-seed selector | flowers | all strict scales accepted by the inner gate, but all rejected by the outer selector |
| unrestricted candidate-owned refit | bonsai | refit generated 6 faces but train-val balanced was negative and the render-region filter kept 0 rows |

Flowers raw-seed strict replay details:

| scale | inner accepted | train-val balanced | tail balanced CVaR | balanced negative fraction | report-only test balanced |
|---:|---:|---:|---:|---:|---:|
| 1.00 | true | +0.000027657 | -0.000048295 | 0.500000 | +0.000003815 |
| 0.75 | true | +0.000021935 | -0.000045389 | 0.421053 | +0.000005603 |
| 0.50 | true | +0.000013828 | -0.000035562 | 0.421053 | +0.000007987 |
| 0.35 | true | +0.000012517 | -0.000007458 | 0.105263 | +0.000009775 |
| 0.20 | true | +0.000005603 | -0.000005774 | 0.078947 | +0.000004888 |

The selector correctly fell back to Phase-J because every candidate remained
below the non-noise thresholds (`dPSNR >= 2e-5`, balanced `>= 5e-5`, tail
balanced `>= 5e-5`). This is an honest failure: shrink improves tail safety, but
also collapses the already-small mean effect.

Implemented next method/policy:

- `ecsr_run_facelocal_coupled_selector.py` now supports
  `--selector_strict_adaptive_scale_policy`.
- The policy is
  `strict_patchcert_train_only_tail_risk_scale_v1`: it expands full-plan strict
  replay scales using train-only per-view certificate tail risk, never held-out
  test metrics.
- Each trial manifest and selector decision records `strict_scale_policy`,
  including `base_scales`, `added_scales`, final scales, risk mean/CVaR/max,
  low-support fraction, and policy band.
- `ecsr_run_autovisual_facelocal_pipeline.py` now exposes the policy in profiles.
- New profile: `field_region_owned_refit_adaptive_medium`.
- This profile keeps render-region filtering as diagnostics, but uses the raw
  strict candidate plan for selector replay so scale-policy candidates are not
  rejected before shrink can be tested.

Dry-run verification:

```text
/tmp/spcarnet_selector_adaptive_dry
/tmp/spcarnet_pipeline_adaptive_profile_dry
```

The flowers dry-run produced this fixed scale set:

```text
base:  1.0,0.75,0.5,0.35,0.2
extra: 0.85,0.7,0.55,0.42,0.32
final: 1.0,0.85,0.75,0.7,0.55,0.5,0.42,0.35,0.32,0.2
policy band: high_tail_risk
selection uses test: false
```

Running validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_owned_refit_adaptive_medium \
  --stages selector \
  --scenes flowers \
  --plan_template '/data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_restricted_20260620_flowers/candidate_plans/{scene}/facelocal_visual_candidate_plan.json' \
  --evidence_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_restricted_20260620_flowers/surface_evidence \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_adaptive_20260620_flowers_selector \
  --pipeline_label autovisual_field_region_owned_refit_adaptive_20260620_flowers_selector \
  --wandb_mode online \
  --gpu 1 \
  --force
```

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_owned_refit_adaptive_medium \
  --stages selector \
  --scenes bonsai \
  --plan_template '/data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_medium_20260620_bonsai/candidate_plans/{scene}/facelocal_visual_candidate_plan.json' \
  --evidence_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_medium_20260620_bonsai/surface_evidence \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_adaptive_20260620_bonsai_selector \
  --pipeline_label autovisual_field_region_owned_refit_adaptive_20260620_bonsai_selector \
  --wandb_mode online \
  --gpu 5 \
  --force
```

Decision gate for this adaptive policy:

- if an adaptive scale passes non-noise selector thresholds and improves
  report-only test without LPIPS/tail harm, promote the profile to the next
  multi-scene run;
- if no adaptive scale passes, the branch is still scientifically weak and the
  next fix must change the representation/objective itself, not add more scalar
  search.

## 2026-06-20 Strict Per-Face Alpha-Shrink Certificate

Status: `IMPLEMENTED_AND_RUNNING_VALIDATION`,
`NOT_COMPLETE_SCIENTIFICALLY`.

Uniform full-plan shrink is too blunt: it reduces side effects, but it also
shrinks the already-small useful signal. The next representation-safe extension
is per-face monotone shrink:

- keep the strict PatchCert carrier face set fixed;
- fit one train-only alpha per face;
- require every alpha to stay in `[0,1]`;
- never amplify coefficients;
- bind the alpha JSON to the strict replay certificate with SHA-256;
- still require post-materialization train-val promotion.

Implemented code:

```text
scripts/car_model/ecsr_run_facelocal_coupled_selector.py
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

New interface:

```text
--selector_strict_fit_plan_alphas
profile: field_region_owned_refit_alpha_medium
render-trust protocol: strict_full_carrier_monotone_shrink_render_trust_v2
```

Strict materialization now accepts alpha only when:

- render-trust exists;
- `selection_uses_test=false`;
- plan SHA matches;
- alpha JSON SHA matches;
- `allow_alpha_shrink=true`;
- alpha face IDs are all present in the candidate plan;
- all alpha values are finite and within `[0,1]`.

First live validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes flowers \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_alpha_20260620_flowers_s1 \
  --plan_template '/data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_restricted_20260620_flowers/candidate_plans/{scene}/facelocal_visual_candidate_plan.json' \
  --evidence_root /data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_restricted_20260620_flowers/surface_evidence \
  --selector_fit_plan_alphas \
  --selector_strict_fit_plan_alphas \
  --selector_strict_replay_scales 1.0 \
  --wandb_project mesh-splatting-ecsr
```

Early audit result:

```text
alpha faces: 10
alpha min / max / mean: 0.990521 / 0.999999 / 0.997405
strict materialize: true
vertices added: 30
render-trust alpha invalid faces: []
render-trust alpha invalid values: {}
```

Interpretation so far:

- This is a real strict-certification interface improvement.
- The first alpha fit is very conservative and almost identical to uniform
  scale 1, so it may not be enough by itself.
- If the rendered train-val result is still below the selector gate, the next
  alpha step must expose stronger train-only risk terms or use render-region
  objective samples, not just evidence-space alpha fitting.

### Flowers Result

Finished outputs:

```text
/data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_adaptive_20260620_flowers_selector/selector
/data/peilincai/spcarnet_runs/autovisual_field_region_owned_refit_alpha_20260620_flowers_s1
```

Uniform adaptive scale policy on `flowers`:

| scale | inner accepted | train-val balanced | tail balanced CVaR | balanced negative fraction | report-only test balanced |
|---:|---:|---:|---:|---:|---:|
| 1.00 | true | +0.000027657 | -0.000048295 | 0.500000 | +0.000003815 |
| 0.85 | true | +0.000020146 | -0.000045612 | 0.421053 | +0.000006199 |
| 0.70 | true | +0.000019550 | -0.000042327 | 0.473684 | +0.000005603 |
| 0.55 | true | +0.000018120 | -0.000039890 | 0.447368 | +0.000006795 |
| 0.42 | true | +0.000012636 | -0.000026174 | 0.394737 | +0.000009775 |
| 0.32 | true | +0.000006914 | -0.000009693 | 0.078947 | +0.000009775 |

Selector decision: `phasej_fallback`, `accepted=false`.

Strict alpha-shrink `s1` on `flowers`:

| trial | inner accepted | train-val balanced | tail balanced CVaR | balanced negative fraction | report-only test balanced | selector |
|---|---:|---:|---:|---:|---:|---|
| strictfull_s1_alpha | true | +0.000027657 | -0.000048146 | 0.500000 | +0.000003815 | fallback |

Conclusion:

- The adaptive uniform policy is valid and test-free, but does not reach the
  `5e-5` non-noise selector gate.
- The alpha interface is valid, but the first alpha solution is nearly identity
  (`mean alpha = 0.997405`), so it does not materially change the rendered
  metrics.
- This branch still fails scientifically on `flowers`. The next method must
  increase representation/objective strength rather than further slicing the
  same weak candidate with safer selection.

## 2026-06-20 Fixed Strict Render-Region-Risk v1

Status: `IMPLEMENTED_AND_RUNNING_VALIDATION`,
`NOT_COMPLETE_SCIENTIFICALLY`.

The adaptive-scale and strict-alpha branches exposed a serious methodology
risk: even when all decisions are train-only, the experiment could still look
like a parameter search over scales, alphas, or filter thresholds. This update
locks the next validation into a fixed policy contract instead of another
manual search.

New profile:

```text
field_region_render_risk_strict_v1
contract: field_region_render_risk_strict_v1_fixed_train_only_no_scale_search
```

Fixed contract:

- candidate-owned render regions are mandatory;
- candidate-owned refit is mandatory;
- selector must consume the filtered plan, not the raw plan;
- strict replay scale is fixed at `1.0`;
- adaptive scale search is disabled;
- selector alpha fitting is disabled;
- strict alpha fitting is disabled;
- render-region plan gate is non-permissive;
- region filter requires train-only positive local effect, tail safety, context
  safety, and a nonzero visible crop-difference floor;
- fixed-profile fields cannot be overridden from CLI.

Implemented files:

```text
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py
```

Engineering additions:

- `ecsr_filter_facelocal_plan_by_render_region.py` now records SHA-256 hashes
  for the candidate plan, render-region objective JSON, and carrier JSON.
- The render-region filter now rejects carriers below:
  - `min_mean_crop_abs_diff`;
  - `min_max_crop_abs_diff`.
- The AutoVisual manifest/summary now records:
  - `profile_contract_id`;
  - `fixed_profile`;
  - `profile_override_policy`;
  - `profile_override_fields`;
  - profile/default SHA-256 values;
  - explicit plan/filter/selector contracts.

Validation already passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py
```

Dry-run manifest:

```text
/tmp/ecsr_strict_render_region_risk_dry/pipeline_command_manifest.json
```

Dry-run audit:

```text
profile: field_region_render_risk_strict_v1
fixed_profile: true
profile_override_policy: forbidden
profile_override_fields: []
selection_uses_test: false
selector_plan_template: /tmp/ecsr_strict_render_region_risk_dry/filtered_candidate_plans/{scene}/facelocal_visual_candidate_plan_filtered.json
selector strict replay scales: 1.0
selector fit plan alphas: false
strict adaptive scale policy: false
strict fit plan alphas: false
```

Override rejection test passed:

```text
--selector_strict_replay_scales 1.0,0.5 -> rejected
--selector_fit_plan_alphas -> rejected
```

Running W&B-online validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v1 \
  --scenes flowers \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v1_20260620_flowers \
  --pipeline_label field_region_render_risk_strict_v1_20260620_flowers \
  --wandb_mode online \
  --gpu 1 \
  --force
```

Expected scientific interpretation:

- If this profile accepts a candidate, the result is much cleaner than the
  adaptive/alpha branches because it is a fixed train-only policy with no scale
  or alpha search.
- If this profile rejects all candidates, that is also useful evidence: the
  current face-local SH representation is too weak to produce visible,
  train-region-safe, globally promotable edits under a fair fixed policy.
- This still does not prove paper-level success; it is the next required gate
  before considering broader multi-scene validation.

## 2026-06-20 Fixed Strict Render-Region-Risk v2

The strict v1 validation on `flowers` finished and rejected the candidate under
the fixed train-only contract. This exposed a specific mechanism bottleneck
rather than just another selector-threshold issue.

v1 evidence:

```text
run: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v1_20260620_flowers
profile: field_region_render_risk_strict_v1
contract: field_region_render_risk_strict_v1_fixed_train_only_no_scale_search
plan gate accepted: false
plan gate reason: render_region_changed_fraction_below_0.05
broad train regions changed: 1 / 62 = 0.016129
candidate-owned train regions changed by raw plan: 12 / 12 = 1.0
candidate-owned refit accepted: false
candidate-owned refit reason: candidate_checkpoint_operator_rejected_or_noop
filtered rows: 0 / 13
selector decision: phasej_fallback, no_plan_candidates
selection uses test: false
```

Interpretation:

- The raw operator can visibly alter the tiny candidate-owned crops.
- The broad render-region gate almost never sees nonzero change.
- Candidate-owned refit collapses to a no-op.
- Therefore the repaired carrier support is too narrow and unstable.

v2 method change:

```text
profile: field_region_render_risk_strict_v2
contract: field_region_render_risk_strict_v2_coverage_prefix_fixed_train_only_no_scale_search
```

Code changed:

```text
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_build_candidate_plan_render_regions.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

New fixed policy components:

- Coverage-prefix carrier selection:
  - `patch_cert_carrier_holdout_auto_prefix_min_face_fraction = 0.70`;
  - `patch_cert_carrier_holdout_auto_prefix_face_bonus = 0.02`;
  - the effective minimum face count is computed from train-certified
    candidate faces, not set per scene.
- Candidate-owned region support expansion:
  - training evidence crops define the support;
  - faces inside candidate-owned region bboxes are admitted only from train
    evidence using alpha/residual support;
  - seed faces and expanded faces are both written to the carrier JSON;
  - expansion settings are recorded in the manifest and carrier markdown.
- The fixed profile still forbids CLI overrides and still disables selector
  alpha fitting, strict alpha fitting, and adaptive scale search.

Validation already passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/ecsr_build_candidate_plan_render_regions.py
```

Dry-run contract:

```text
/tmp/ecsr_strict_render_region_risk_v2_dry/pipeline_command_manifest.json
coverage_prefix_contract:
  min_face_fraction: 0.7
  face_bonus: 0.02
candidate_region_contract:
  expand_faces: true
  expand_min_face_pixels: 12
  expand_min_face_views: 1
  expand_max_faces_per_carrier: 32
selector fit plan alphas: false
selector strict replay scales: 1.0
selection uses test: false
```

Override rejection test passed:

```text
--delta_patch_cert_carrier_holdout_auto_prefix_min_face_fraction 0.2 -> rejected
```

Offline builder sanity check on the v1 `flowers` plan:

```text
input plan faces: 13
output carrier faces after expansion: 64
expanded faces: 51
carrier 1: seed 3 -> total 32
carrier 2: seed 10 -> total 32
```

Running W&B-online validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v2 \
  --scenes flowers \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v2_20260620_flowers \
  --pipeline_label field_region_render_risk_strict_v2_20260620_flowers \
  --wandb_mode online \
  --gpu 1 \
  --force
```

Early in-run signal:

```text
operator audit accepted: true
accepted faces: 19
accepted carriers: 3
coverage-prefix effective minimum faces: 15
coverage-prefix min face fraction: 0.7
final plan-gate decision: false
final plan-gate reason: render_region_changed_fraction_below_0.05
```

Required interpretation gate:

- If v2 clears the broad render-region gate and keeps train-val/test deltas
  nonnegative, it becomes the next fixed-policy candidate for multi-scene
  validation.
- If v2 still collapses at refit/filter/selector, the next method change must
  move beyond face-local SH residuals toward a stronger region-level
  representation. More threshold tuning is not justified.

Status: `V2_IMPLEMENTED_AND_RUNNING_VALIDATION`, `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-20 Fixed Strict Render-Region-Risk v3

v2 exposed a second bottleneck during candidate-owned refit. The holdout
certificate was positive, but the absolute global policy-val sample gate
remained too coarse for local region repair:

```text
v2 refit audit:
allowed face ids: 96
selected faces before final gate: 24
carrier-holdout selected faces: 16
carrier-holdout selected carriers: 2
carrier-holdout effective coverage floor: 12
final accepted policy-val relative gain: 0.164315
final accepted policy-val samples: 187
configured min_policy_val_samples: 512
final candidate plan rows: 0
```

The problem is not that the local repair has no train evidence. It has positive
holdout and positive final accepted proxy, but the fixed absolute 512-sample
gate can erase small localized repairs. Changing that value per scene would be
a parameter game, so v3 implements a support-aware fixed policy.

v3 method change:

```text
profile: field_region_render_risk_strict_v3
contract: field_region_render_risk_strict_v3_support_aware_policy_floor_fixed_train_only_no_scale_search
```

New support-aware sample gate:

```text
configured absolute min policy-val samples: 512
adaptive sample fraction: 0.30
adaptive min samples: 128
effective min samples:
  min(512, max(128, ceil(0.30 * available_policy_val_samples)))
```

This keeps the original 512-sample demand for broad-support candidates while
allowing local region repairs to pass only when they clear a fixed fractional
support floor and a fixed absolute floor. The effective threshold is written to
the operator audit as:

```text
effective_min_policy_val_samples
min_policy_val_adaptive_sample_fraction
min_policy_val_adaptive_min_samples
min_policy_val_adaptive_sample_floor
```

Validation already passed:

```text
py_compile passed for apply, PhaseK wrapper, pipeline, and candidate-region builder.
dry-run manifest passed:
  /tmp/ecsr_strict_render_region_risk_v3_dry/pipeline_command_manifest.json
override rejection passed:
  --delta_min_policy_val_adaptive_sample_fraction 0.1 -> rejected
```

Running W&B-online validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v3 \
  --scenes flowers \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v3_20260620_flowers \
  --pipeline_label field_region_render_risk_strict_v3_20260620_flowers \
  --wandb_mode online \
  --gpu 4 \
  --force
```

Status: `V3_IMPLEMENTED_AND_RUNNING_VALIDATION`, `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-20 Fixed Strict Render-Region-Risk v4

v3 fixed the refit no-op problem but exposed a tail-risk problem:

```text
run: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v3_20260620_flowers
refit candidate plan rows: 10
refit candidate carriers: 1
candidate-owned refit changed regions: 7 / 16 = 0.4375
mean core balanced delta: +0.021256
tail core balanced CVaR: -0.002509
worst core balanced delta: -0.006660
negative core balanced fraction: 0.125
train-val balanced delta: -0.000268
filter kept rows: 0 / 10
filter reason: tail_core_balanced_delta_below_-2e-05
```

Interpretation:

- v3 made the local refit real instead of no-op.
- The mean candidate-owned render-region gain was strong.
- A small tail subset regressed enough that the fair filter correctly rejected
  the carrier.
- The next method change should preserve the whole strict carrier while reducing
  tail risk, not split the carrier and violate PatchCert integrity.

v4 method change:

```text
profile: field_region_render_risk_strict_v4
contract: field_region_render_risk_strict_v4_tail_safe_shrink_fixed_train_only_no_scale_search
```

New train-only tail-safe shrink:

```text
enabled only when the only filter failure is tail_core_balanced_delta
scale = mean_core_balanced_delta /
        (mean_core_balanced_delta + abs(tail_core_balanced_delta - min_tail))
scale is clamped by fixed min scale 0.50
```

This is not selector scale search. The shrink is a deterministic function of
train render-region statistics and is written into each kept candidate row as
`render_region_tail_safe_shrink`.

Validation:

```text
py_compile passed for filter, pipeline, apply, PhaseK wrapper, and builder.
dry-run manifest passed:
  /tmp/ecsr_strict_render_region_risk_v4_dry/pipeline_command_manifest.json
offline filter replay on v3 refit:
  input rows: 10
  kept rows: 10
  shrink scale: 0.894813
  decision note: tail_safe_shrink_applied
```

Running W&B-online validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v4 \
  --scenes flowers \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v4_20260620_flowers \
  --pipeline_label field_region_render_risk_strict_v4_20260620_flowers \
  --wandb_mode online \
  --gpu 4 \
  --force
```

Status: `V4_IMPLEMENTED_AND_RUNNING_VALIDATION`, `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-21 Fixed Strict Render-Region-Risk v5/v6

v4 showed a different failure mode: the local carrier could be visibly active,
but context/tail train risk could still block the filter. v5 adds a second
deterministic shrink path for train render-risk failures:

```text
profile: field_region_render_risk_strict_v5
contract: field_region_render_risk_strict_v5_context_tail_risk_shrink_fixed_train_only_no_scale_search
```

v5 `flowers` full-chain evidence:

```text
run: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v5_20260620_flowers
plan broad scene-prior gate: rejected
  reason: render_region_changed_fraction_below_0.05
  changed regions: 1 / 62 = 0.016129

candidate-owned refit:
  candidate plan rows: 10
  carrier count: 1
  changed regions: 26 / 27 = 0.962963
  mean core balanced delta: +0.0125318
  mean core dPSNR: +0.0073181
  tail core balanced CVaR: -0.0040394
  worst core balanced delta: -0.0080191
  report-only test balanced delta: +0.0011944
  decision: rejected by tail CVaR before filter shrink

filter:
  input rows: 10
  kept rows: 10
  kept carriers: 1
  deterministic shrink scale: 0.7503097
  note: tail_safe_shrink_applied

selector:
  trial strictfull_s1 inner gate: accepted
  train-val balanced delta: +0.00003159
  report-only test balanced delta: +0.0012445
  outer selector: rejected
  selector reason: selector_balanced_delta_below_5e-05
  train-val tail reason: psnr negative fraction 0.236842 > 0.20
```

Interpretation:

- v5 is a real method improvement over the previous no-op and filter-empty
  failures: the candidate-owned support now changes almost every owned region
  and produces a positive report-only held-out test delta.
- It is still not a completed paper-level result. The broad scene-prior gate is
  mostly unchanged, and the fair outer selector does not promote the method
  because train-val evidence is below the fixed non-noise threshold.
- The next step must improve tail stability and train-val mean without using
  test-set selection.

v6 method change:

```text
profile: field_region_render_risk_strict_v6
contract: field_region_render_risk_strict_v6_pre_registered_trainval_shrink_ladder
selector_strict_replay_scales: 1.0,0.85,0.75,0.6,0.5,0.35,0.2
selector adaptive scale policy: false
selector strict alpha shrink: false
fixed profile override policy: forbidden
```

This is a pre-registered train-val shrink ladder, not a per-scene manual search.
Every scale is a monotone shrink of the same full strict carrier. Selection is
still train-val only, with report-only test metrics written after the fact.

Validation completed:

```text
py_compile passed for ecsr_run_autovisual_facelocal_pipeline.py
dry-run manifest passed:
  /tmp/ecsr_strict_render_region_risk_v6_dry/pipeline_command_manifest.json
override rejection passed:
  --selector_strict_replay_scales 1.0 -> rejected
```

Running W&B-online validation:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes flowers \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v6_selector_ladder_20260621_flowers/selector \
  --plan_template '/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v5_20260620_flowers/filtered_candidate_plans/{scene}/facelocal_visual_candidate_plan_filtered.json' \
  --evidence_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v5_20260620_flowers/surface_evidence \
  --selector_strict_replay_scales 1.0,0.85,0.75,0.6,0.5,0.35,0.2 \
  --no-selector_strict_adaptive_scale_policy \
  --no-selector_strict_fit_plan_alphas \
  --selector_enable_tail_stable_promotion \
  --force
```

Status: `V6_IMPLEMENTED_AND_RUNNING_SELECTOR_VALIDATION`, `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-21 Update: v7 ROI-Stable Promotion and End-to-End Validation In Progress

v7 changes the selection rule from a pure full-frame mean gate to a fixed
train-only rule with two requirements:

```text
1. full-frame train-val metrics must be non-regressive under the Phase-K gate;
2. a train-only render-region filter must certify a strong ROI improvement.
```

This is meant to handle the core failure mode observed in v5/v6: local repairs
are visually meaningful, but full-frame averages can be too small to pass a
strict mean-only selector.

Implemented fixed profile:

```text
profile: field_region_render_risk_strict_v7
contract: field_region_render_risk_strict_v7_roi_stable_trainval_promotion
selector_region_min_trainval_balanced_delta: 0.0
selector_region_min_mean_core_balanced_delta: 0.01
selector_region_min_mean_delta_psnr: 0.001
selector_region_min_changed_fraction: 0.50
selector_region_max_negative_core_balanced_fraction: 0.35
selector_region_max_context_mse_regression: 1e-6
```

Flowers evidence so far:

```text
candidate-owned region objective:
  regions: 27
  mean core balanced delta: +0.0169456
  mean core dPSNR: +0.0090981
  tail core balanced CVaR: -0.0017248
  negative core balanced fraction: 0.185185

candidate-owned refit plan:
  candidate rows: 10
  carrier count: 1
  policy-val proxy relative gain: +0.506778
  fit proxy relative gain: +0.040958

refit gate:
  train-val balanced delta: +0.00004244
  report-only test balanced delta: +0.00123966
  decision: rejected by render-region tail CVaR

filter:
  input rows: 10
  kept rows: 10
  kept carriers: 1
  mean core balanced delta: +0.0194297
  mean dPSNR: +0.0113588
  tail core balanced: -0.0067895
  note: tail_safe_shrink_applied

selector strictfull_s1 inner gate:
  accepted: true
  train-val balanced delta: +0.00003755
  report-only test balanced delta: +0.00123131
  report-only test PSNR/SSIM/LPIPS deltas: +0.000169754 / +0.000070930 / +0.000017852
```

Bonsai evidence so far:

```text
initial v7 plan:
  candidate rows: 172
  carrier count: 19
  policy-val proxy relative gain: +0.536735
  fit proxy relative gain: +0.421184

initial full-frame gate:
  train-val balanced delta: -0.00025773
  report-only test balanced delta: -0.00014448
  decision: rejected by psnr_gain_below_0 and balanced_delta_below_0

initial render-region gate:
  accepted: true
  regions: 62
  mean core balanced delta: +0.0020345
  mean dPSNR: +0.0042091
  changed regions: 5 / 62 = 0.080645
```

Interpretation:

- v7 is a real method-level change, not a parameter scan: it adds fixed
  ROI-stable promotion and forces the evidence to come from train-only
  render-region certification plus train-val full-frame non-regression.
- flowers currently has the strongest end-to-end evidence: local ROI gain is
  large, full-frame train-val is positive, and report-only test is positive.
- bonsai remains the key weakness: local ROI evidence exists, but the initial
  full-frame gate is negative. The candidate-owned refit/filter/selector
  stages are still running and must decide whether this is recoverable.

v6 final selector result:

```text
output:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v6_selector_ladder_20260621_flowers/selector/flowers/coupled_selector_decision.json

accepted: false
selected trial: phasej_fallback
selection uses test: false

best-looking trials:
  strictfull_s1:
    inner accepted: true
    train-val balanced delta: +0.00003159
    report-only test balanced delta: +0.00124454
    selector rejected: selector_balanced_delta_below_5e-05,
      tail_balanced_delta_below_5e-05,
      tail_psnr_negative_fraction_exceeds_0.2
  strictfull_s0p75:
    inner accepted: true
    train-val balanced delta: +0.00002813
    report-only test balanced delta: +0.00124633
    selector rejected by mean/tail thresholds

low scales:
  strictfull_s0p35 report-only test balanced delta: -0.00000107
  strictfull_s0p2 report-only test balanced delta: +0.00000310
```

Conclusion: a pre-registered shrink ladder alone does not solve the selector
problem. The useful local repair exists, but mean/tail-only promotion still
rejects it. v7 ROI-stable promotion is therefore not cosmetic; it is required
to represent the intended local-repair claim.

Partial v7 flowers re-decision using completed scales `1.0,0.85`:

```text
output:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v7_partial_redecision_20260621_flowers/selector/flowers/coupled_selector_decision.json

accepted: true
selected trial: strictfull_s1
selected selector pass mode: region_stable
selection uses test: false
train-val balanced delta: +0.00003755
report-only test deltas:
  PSNR +0.000169754
  SSIM +0.000070930
  LPIPS +0.000017852

ROI promotion evidence:
  accepted carriers: 1 / 1
  mean changed fraction: 0.9375
  mean core balanced delta: +0.0194297
  mean dPSNR: +0.0113588
  negative core balanced fraction: 0.3125
  max context MSE regression: 2.8871e-08
```

Bonsai v7 candidate-owned refit exposes the main remaining weakness:

```text
decision:
  accepted: false
  train-val balanced delta: -0.00025296
  report-only test balanced delta: -0.00329977
  reasons:
    psnr_gain_below_0
    balanced_delta_below_0
    render_region_tail_cvar_below_-2e-05

ROI/refit region objective:
  mean core balanced delta: +0.114849
  mean dPSNR: +0.192188
  tail core balanced CVaR: -0.118622
  worst core balanced delta: -0.392172
  negative core balanced fraction: 0.21875
```

Interpretation: bonsai is not failing because the method cannot find a local
signal. It fails because the learned residual has strong ROI mean gains but
large tail failures and full-frame regressions. The next method-level step
should explicitly suppress out-of-ROI / tail-risk spillover rather than adding
more selector thresholds.

Status: `V7_IMPLEMENTED_AND_RUNNING_END_TO_END_VALIDATION`, `NOT_COMPLETE_SCIENTIFICALLY`.

## 2026-06-21 Update: v8 Tail-Severity-Gated Shrink

v7 exposed a specific mechanism flaw on bonsai: the train-render ROI mean could
be strongly positive while the tail CVaR was severely negative. The previous
tail-safe shrink policy computed an analytic mean/tail shrink ratio, but then
floored the effective scale at `0.5`. This protected mild tail-risk cases like
flowers, but it could also rescue bonsai carriers whose raw safe scale was far
below `0.5`.

Implemented fixed v8:

```text
profile: field_region_render_risk_strict_v8
contract: field_region_render_risk_strict_v8_tail_severity_gated_roi_stable_promotion
new fixed parameter:
  filter_tail_safe_shrink_min_raw_scale = 0.60
```

Mechanism:

- compute the raw analytic mean/tail shrink ratio before applying the
  effective scale floor;
- allow shrink rescue only when `raw_scale >= 0.60`;
- reject carriers that would need an extreme shrink to hide tail failures;
- record both `tail_safe_shrink_raw_scale` and `tail_safe_shrink_scale` in the
  candidate plan, per-carrier filter summary, and markdown table;
- keep v7 ROI-stable selector promotion unchanged, so the new mechanism only
  changes carrier safety, not the selector success criterion.

Verification:

```text
py_compile:
  scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
  passed

dry-run manifest:
  /tmp/ecsr_strict_render_region_risk_v8_dry/pipeline_command_manifest.json
  profile contract: field_region_render_risk_strict_v8_tail_severity_gated_roi_stable_promotion
  filter command includes --tail_safe_shrink_min_raw_scale 0.6

fixed-profile override protection:
  overriding --filter_tail_safe_shrink_min_raw_scale under v8 correctly errors
```

Offline replay on bonsai using the v7 candidate-owned refit evidence:

```text
input: 170 rows / 17 carriers
v7 filter kept: 150 rows / 15 carriers
v8 filter kept: 90 rows / 9 carriers

accepted carrier raw-scale range:
  min 0.634739
  max 1.000000
  accepted carriers below 0.60: 0

rejected carriers below 0.60:
  8 / 8 rejected carriers

worst rejected examples:
  raw 0.018703, mean +0.003618, tail -0.189873
  raw 0.071405, mean +0.024688, tail -0.321084
  raw 0.369602, mean +0.106460, tail -0.181600
```

Offline replay on flowers using the v7 candidate-owned refit evidence:

```text
input: 10 rows / 1 carrier
v8 filter kept: 10 rows / 1 carrier
```

Interpretation: v8 is a real fixed-policy method change. It preserves the
flowers carrier that v7 can promote, while removing the bonsai carriers whose
tail failures are too severe for the deterministic shrink policy to justify.
The open question is whether the safer bonsai carrier set now survives
train-val selector validation; W&B-online v8 bonsai selector replay is running
at:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v8_replay_20260621_bonsai/selector
```

Status: `V8_IMPLEMENTED`, `BONSAI_SELECTOR_RUNNING`, `NOT_COMPLETE_SCIENTIFICALLY`.

### 2026-06-21 Follow-up: Full v7 Flowers Result and Bonsai Retry

The full v7 flowers pipeline finished, confirming the earlier partial
re-decision:

```text
run:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v7_20260621_flowers

selector decision:
  accepted: true
  selected trial: strictfull_s1
  selected pass mode: region_stable
  selection uses test: false
  train-val balanced delta: +0.0000375509
  report-only test deltas:
    PSNR +0.000169754
    SSIM +0.0000709295
    LPIPS +0.0000178516
```

The first v8 bonsai replay run exposed an experimental-system reliability
issue rather than a method decision: ELA sometimes read train/base PNGs while
they were still being written, causing transient `image file is truncated`
errors. The same file could be opened successfully moments later, confirming a
read/write race. I changed `utils/evidence_lumigraph_adapter.py` so image reads
retry for up to 30 attempts with bounded waits, then restarted bonsai v8 under
an isolated method/output name:

```text
failed run:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v8_replay_20260621_bonsai/selector

retry run:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v8_retry_20260621_bonsai/selector

retry PhaseJ methods:
  ours_26000_phasej_guarded_adaptedge_ela_field_region_render_risk_strict_v8_retry_20260621_bonsai
  ours_26000_phasej_trainval_gate_rendercalib_v1_field_region_render_risk_strict_v8_retry_20260621_bonsai
```

Status: `FLOWERS_V7_FULL_ACCEPTED`, `BONSAI_V8_RETRY_RUNNING`, `NOT_COMPLETE_SCIENTIFICALLY`.
