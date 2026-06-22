# 5-28 Risk-Aware Carrier Shrink v20-v23 Log

Date: 2026-06-22

Status: active validation, not paper-complete.

## Why This Exists

v18 and v19 exposed the same core problem from two sides:

- post-refit alpha risk suppression was too late to stop unstable local
  coefficients from being trained;
- hard pre-refit carrier pruning was too destructive because it removed the
  exact local evidence that sometimes produced a tiny PSNR gain.

The current repair path therefore keeps candidate carriers available, but makes
high-risk carrier-owned faces enter the facelocal refit with lower coefficient
authority. This is a method-level change in the train/eval pipeline, not a
threshold-only selector change.

## v20 Method

Fixed profile:

```text
profile: field_region_render_risk_strict_v20
contract: field_region_render_risk_strict_v20_train_render_risk_pre_refit_carrier_shrink
```

Core mechanism:

- read train-only render-region objective rows before candidate-owned refit;
- map objective rows back to carrier face sets;
- select bad carriers using a fixed objective-bad rule;
- write a per-face alpha JSON before refit;
- forward that alpha JSON into the facelocal materialization script;
- multiply selected face-local residual coefficients by the per-face scale
  before the candidate checkpoint is evaluated.

This changed the actual candidate-owned refit path through:

```text
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
```

Validation command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

Medium run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v20 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Evidence paths:

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium/candidate_owned_refit_plans/bonsai/pre_refit_risk_shrink_report.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium/candidate_owned_refit_plans/bonsai/pre_refit_risk_shrink_materialize_alpha.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium/candidate_owned_refit/bonsai/model/surface_residual_facelocal_sh1_delta_audit.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v20_trainrisk_20260622_bonsai_medium/candidate_owned_refit/decisions/bonsai_decision.json
```

Observed v20 candidate-owned evidence:

```text
pre-refit shrink status: applied
carrier_count: 8
shrunk_carrier_count: 3
shrunk_face_count in final filtered plan scope: 34
alpha JSON face count reaching materialization: 192
facelocal face_risk_scale.enabled: true
facelocal matched scaled faces: 144
facelocal affected coefficient rows: 432
scale range: 0.55 to 0.85, mean 0.725
```

v20 train-val comparison against Phase-J fallback:

```text
Phase-J train-val: LPIPS 0.2431543916, PSNR 30.0707855225, SSIM 0.9085068107
v20 candidate:    LPIPS 0.2431887388, PSNR 30.0711555481, SSIM 0.9084812999
delta:            LPIPS +0.0000343472, PSNR +0.0003700256, SSIM -0.0000255108
```

v20 test comparison against Phase-J fallback:

```text
Phase-J test: LPIPS 0.2512249947, PSNR 29.2167377472, SSIM 0.8995919228
v20 test:    LPIPS 0.2512435019, PSNR 29.2173900604, SSIM 0.8995746970
delta:       LPIPS +0.0000185072, PSNR +0.0006523132, SSIM -0.0000172257
```

v20 is a real pipeline connection, but it is not a successful scientific
result. The candidate-owned decision rejects it:

```text
accepted: false
decision reasons: balanced_delta_below_0, render_region_tail_cvar_below_-2e-05
render-region mean core balanced delta: +0.2384702469
render-region mean PSNR delta: +0.2950335247
render-region mean SSIM delta: +0.0022347373
render-region mean LPIPS delta: +0.0050629012
render-region tail CVaR: -0.0996214043
negative core-balanced fraction: 0.1707317073
worst core-balanced delta: -0.3696248531
```

Interpretation:

v20 solved the previous interface/no-op weakness, but it did not solve local
tail collapse. The changed-region crop mean is positive, while a minority of
regions still collapses enough for the strict train-only gate to reject the
candidate. This should not be described as a paper-level win.

Final selector result:

```text
selector decision:
  accepted: false
  selected_trial: phasej_fallback
  selected_trainval_balanced_delta: 0.0

strictfull_s1 trial:
  accepted: false
  decision reasons: balanced_delta_below_0
  trainval balanced delta: -0.0000489354
  trainval delta: LPIPS +0.0000051409, PSNR +0.0000896454, SSIM -0.0000017881
  report-only test delta: LPIPS +0.0000027418, PSNR +0.0000762939, SSIM -0.0000011325
```

## v21 Method

Fixed profile:

```text
profile: field_region_render_risk_strict_v21
contract: field_region_render_risk_strict_v21_severity_aware_pre_refit_carrier_risk_shrink
```

v21 keeps the v20 mechanism and makes the shrink policy severity-aware. Carrier
scale is no longer only tied to bad-row fraction. It is driven by the maximum
severity from:

- bad-region fraction;
- worst core-balanced region regression;
- tail CVaR regression.

Fixed profile settings:

```text
candidate_region_pre_refit_risk_shrink_min_scale: 0.10
candidate_region_pre_refit_risk_shrink_severity_aware: true
candidate_region_pre_refit_risk_shrink_severity_select_min: 0.5
candidate_region_pre_refit_risk_shrink_severity_balanced_span: 0.05
candidate_region_pre_refit_risk_shrink_tail_fraction: 0.25
```

This is still a fixed policy, not a per-scene parameter scan.

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

git diff --check -- scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v21 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v21_severity \
  --pipeline_label dryrun_field_region_render_risk_strict_v21_severity \
  --wandb_mode online \
  --dry_run \
  --force
```

Dry-run result:

```text
commands: 8
profile: field_region_render_risk_strict_v21
severity-aware pre-refit shrink: enabled
min scale: 0.10
```

Active medium run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v21 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 1 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v21_severity_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v21_severity_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Current live status:

```text
run root: /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v21_severity_20260622_bonsai_medium
GPU: 1
status: running; plan_generation reached trainval ELA, no candidate-owned refit decision yet.
artifact caveat: this run was launched before the v22 alpha-artifact split, so
  its alpha JSON may still follow the old materialization path semantics.
```

## v22 Method

Fixed profile:

```text
profile: field_region_render_risk_strict_v22
contract: field_region_render_risk_strict_v22_region_local_bad_row_suppression
```

v22 keeps v21 severity-aware carrier shrink and adds a more local train-only
suppression rule. Instead of lowering all faces in a risky carrier equally, it
uses the train render-region objective rows to identify severe bad rows, maps
those rows back to the carrier expansion metadata, and suppresses only the
high-pixel faces visible in those bad rows. The goal is to preserve the
positive witness regions that v20 already found while neutralizing the rows
that made the tail CVaR fail.

Fixed profile settings:

```text
candidate_region_pre_refit_risk_local_suppression: true
candidate_region_pre_refit_risk_local_suppression_scale: 0.02
candidate_region_pre_refit_risk_local_min_bad_balanced: 0.02
candidate_region_pre_refit_risk_local_positive_margin: 0.02
candidate_region_pre_refit_risk_local_min_face_pixels: 12
candidate_region_pre_refit_risk_local_max_faces_per_bad_row: 16
```

Reproducibility repair added with v22:

```text
pre-refit alpha consumed by refit:
  candidate_owned_refit_plans/{scene}/pre_refit_risk_scale_refit.json
selector materialization alpha:
  candidate_owned_refit_plans/{scene}/selector_materialize_alpha.json
legacy path avoided for new v22 runs:
  candidate_owned_refit_plans/{scene}/pre_refit_risk_shrink_materialize_alpha.json
```

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

git diff --check -- scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v22 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v22_local_suppression \
  --pipeline_label dryrun_field_region_render_risk_strict_v22_local_suppression \
  --wandb_mode online \
  --dry_run \
  --force
```

Dry-run result:

```text
commands: 8
profile: field_region_render_risk_strict_v22
local suppression: enabled
pre_refit_risk_shrink_alpha_json: pre_refit_risk_scale_refit.json
selector_risk_shrink_alpha_json: selector_materialize_alpha.json
```

Medium run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v22 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v22_local_suppression_20260622_bonsai_medium \
  --pipeline_label field_region_render_risk_strict_v22_local_suppression_20260622_bonsai_medium \
  --wandb_mode online \
  --force
```

Execution note:

```text
first attempt: GPU3, failed in surface evidence cache with CUDA OOM because
  the card had only about 660 MiB free despite low util.
second attempt: GPU5, reached plan facelocal fitting.
status: aborted intentionally after v23 review fixes, because v22 is a flawed
  ablation and GPU5 was needed for the audited v23 method.
```

Review caveat:

v22 should be treated as an ablation, not the final method. A code review found
three trust issues:

- refit alpha and selector alpha were separated, but both stages still wrote
  the same shrink report path;
- local suppression could fall back to arbitrary carrier faces when expansion
  metadata did not match the bad row;
- positive-witness preservation almost never triggered for normal view-labeled
  objective rows.

These are not direct test-set leakage, but they are serious reproducibility and
method-definition weaknesses.

## v23 Method

Fixed profile:

```text
profile: field_region_render_risk_strict_v23
contract: field_region_render_risk_strict_v23_audited_metadata_local_suppression
```

v23 keeps the v22 scientific intent but makes it auditable:

- refit report:
  `candidate_owned_refit_plans/{scene}/pre_refit_risk_shrink_report_refit.json`
- selector report:
  `candidate_owned_refit_plans/{scene}/pre_refit_risk_shrink_report_selector.json`
- refit alpha:
  `candidate_owned_refit_plans/{scene}/pre_refit_risk_scale_refit.json`
- selector alpha:
  `candidate_owned_refit_plans/{scene}/selector_materialize_alpha.json`
- manifest now records SHA-256 hashes for output files that exist when the
  manifest is written;
- risk-shrink refit is not allowed to silently reuse a stale refit plan without
  `--force`;
- after candidate-owned refit, the pipeline asserts that the facelocal audit
  actually enabled `face_risk_scale`, matched scaled faces, and affected
  coefficient rows;
- local bad-row suppression no longer falls back to arbitrary carrier faces.
  A bad row must have a view and must match face expansion view/pixel metadata;
  otherwise it is explicitly recorded as skipped;
- positive witness views can now protect faces for ordinary bad rows, while
  severe bad rows can override that protection.

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py

git diff --check -- \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  docs/car_model/5-28-RiskAwareCarrierShrink-v20-v21-Log.md

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v23 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v23_audited_local_suppression \
  --pipeline_label dryrun_field_region_render_risk_strict_v23_audited_local_suppression \
  --wandb_mode online \
  --dry_run \
  --force
```

Dry-run result:

```text
commands: 8
profile: field_region_render_risk_strict_v23
contract: field_region_render_risk_strict_v23_audited_metadata_local_suppression
report split: refit and selector paths present in manifest
output artifact hashing: output_path_sha256s field present in manifest
```

Medium-run result:

```text
v21 root:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v21_severity_20260622_bonsai_medium
v23 root:
  /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v23_audited_local_suppression_20260622_bonsai_medium
```

v21 candidate-owned refit decision:

```text
accepted: false
selected_label: phasej_guarded_adaptedge
decision reasons: balanced_delta_below_0, render_region_tail_cvar_below_-2e-05
train-val balanced delta: -0.0006714463
train-val delta: LPIPS +0.0000314862, PSNR +0.0003433228, SSIM -0.0000192523
report-only test balanced delta: +0.0001194477
report-only test delta: LPIPS +0.0000156164, PSNR +0.0006427765, SSIM -0.0000105500
```

v21 candidate-owned render-region objective:

```text
regions: 41
changed regions: 41 / 41
mean core balanced delta: +0.2351670927
mean core PSNR delta: +0.2911424869
mean core SSIM delta: +0.0020019895
mean core LPIPS delta: +0.0048007593
tail core balanced CVaR: -0.0996001417
negative core-balanced fraction: 0.1951219512
worst core-balanced delta: -0.3704202175
wins: core balanced 33, core PSNR 41, core SSIM 25, core LPIPS 17
```

v23 plan-generation decision:

```text
accepted: false
selected_label: phasej_guarded_adaptedge
decision reason: balanced_delta_below_0
train-val balanced delta: -0.0002019405
train-val delta: LPIPS +0.0000158548, PSNR +0.0003833771, SSIM -0.0000134110
report-only test balanced delta: +0.0001702309
report-only test delta: LPIPS +0.0000078082, PSNR +0.0004444122, SSIM -0.0000059009
```

v23 plan-generation render-region objective:

```text
regions: 62
changed regions: 5 / 62
mean core balanced delta: +0.0020345228
mean core PSNR delta: +0.0042090877
mean core SSIM delta: -0.0000049155
mean core LPIPS delta: +0.0001038128
tail core balanced CVaR: 0.0
negative core-balanced fraction: 0.0
worst core-balanced delta: 0.0
wins: core balanced 5, core PSNR 5, core SSIM 3, core LPIPS 0
```

Interpretation:

v23 made the artifact path and local-suppression policy auditable, but it did
not solve the scientific problem. Both v21 and v23 show the same pattern:
PSNR can be nudged upward by a tiny amount, while LPIPS and SSIM regress. v21
also proves that large positive mean crop gains are not enough because the
tail contains severe bad regions. Therefore the next step should not be
another profile-level threshold tweak or multi-scene packaging pass.

## Current Decision Gate And Pivot

The next milestone is not another cosmetic README update. The method must pass
this gate before it should be expanded:

```text
candidate-owned refit decision accepted, or selector promotes a candidate;
train-val balanced delta >= 0;
test metrics do not regress beyond the strict gate;
render-region tail CVaR >= -2e-05;
negative core-balanced region fraction stays below the registered threshold;
qualitative crops show visible local improvement instead of barely measurable noise.
```

The v21/v23 evidence fails this gate. The next method must move from post-fit
risk scaling into the fitting objective itself. The planned v24 direction is:

- add a train-only bystander / zero-delta witness objective so samples outside
  the target region explicitly penalize residual motion;
- keep the existing core reconstruction and view-tail CVaR terms, but record
  bystander sample counts and final loss in the facelocal audit;
- pass the new objective through PhaseK and the autovisual fixed profile as a
  preregistered profile, not a per-scene parameter scan;
- rerun bonsai medium with W&B online, then only expand to more scenes if the
  train-val decision and render-region tail both pass.
