# Phase-S Fold-Aware PatchCert Continuation Log

Date: 2026-05-14

This log records the continuation after the compact-stratified PatchCert v6
milestone. The immediate goal is to turn the remaining weakness into a fixed
method improvement, not a parameter scan:

```text
v6: compact-stratified PatchCert gate
v7: seed-face all-train fold consistency before materialization
v8: seed-face fold consistency plus aggregate patch-carrier fold consistency
v8.1: neighbor-admission fold consistency plus post-shrink patch-carrier fold consistency
v8.2: strict carrier integrity fix for plan replay, post-shrink gain, and whole-patch budgeting
```

Held-out test renders remain report-only. Selection and promotion use
train-only evidence, train-val renders, and fixed gates.

## Why v8 Exists

The v6 compact override improved coverage from `1 / 5` to `2 / 5`
(`bicycle`, `flowers`), but `flowers` is not a clean all-diagnostic win: it is
accepted because the edited carrier is compact and bounded by component, tail,
and stratified train-val risk.

The v7 crossfold run adds a stricter seed-face certificate:

```text
--delta_crossfold_gain_certificate_folds 4
--delta_crossfold_min_passing_folds 4
--delta_crossfold_min_fold_relative_gain 0
--delta_crossfold_min_fold_samples 4
```

Subagent review found a remaining method gap: PatchCert patch growth could
still add neighboring faces whose patch carrier was only certified on the
single policy-val split. v8 fixes that by certifying the grown patch itself on
all train-only folds before it can be accepted.

## Implementation

Changed files:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

New fixed interface:

```text
--delta_patch_cert_crossfold_folds
--delta_patch_cert_crossfold_min_passing_folds
--delta_patch_cert_crossfold_min_fold_relative_gain
--delta_patch_cert_crossfold_min_fold_samples
```

The runner forwards these to the face-local SH operator as:

```text
--patch_cert_crossfold_folds
--patch_cert_crossfold_min_passing_folds
--patch_cert_crossfold_min_fold_relative_gain
--patch_cert_crossfold_min_fold_samples
```

The operator now builds a train-only fold cache and evaluates every grown
PatchCert carrier with `evaluate_proxy_for_faces(...)` on each fold. A patch
carrier is accepted only if it reaches the required number of eligible passing
folds. After review, this interface was tightened further: v8.1 can require
each neighbor to pass the same fold certificate before it enters the patch, and
it recomputes the patch-fold certificate after patch shrink before
materialization. When the new certificate is disabled, historical PatchCert
behavior is preserved: a failed grown patch can still fall back to its already
certified seed face.

This is still explicitly an all-train fold consistency certificate, not a true
cross-fit estimator.

## Validation Already Run

Code-level checks after the implementation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  docs/car_model/5-14-PhaseS-CompactStratified-Gate-Log.md \
  docs/car_model/5-14-SPCarNet-Method-Modules-And-Evidence-Log.md
```

Both checks passed.

## Running Experiments

### v6 Four-Offset Audit

Purpose: post-hoc strict audit for the v6 accepted scenes.

Output root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/phase_s_patchcert_v6_compactstrat_gate_20260514
```

Expected final JSON:

```text
.../bicycle/multifold_trainval_gate.json
.../flowers/multifold_trainval_gate.json
```

W&B group:

```text
phase_s_patchcert_v6_compactstrat_multifold_20260514
```

Status at log creation: running, no final JSON yet.

### v7 Seed-Fold PatchCert

Purpose: stricter seed-face fold consistency before PatchCert materialization.

Output roots:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v7_crossfold_compactstrat_20260514_bicycle
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v7_crossfold_compactstrat_20260514_flowers
```

Expected final decisions:

```text
.../decisions/bicycle_decision.json
.../decisions/flowers_decision.json
```

W&B group:

```text
phase_s_patchcert_v7_crossfold_compactstrat_20260514
```

Status at log creation:

- `bicycle`: operator completed as no-op/fallback (`accepted=false`,
  `accepted_faces=0`), render/eval still running.
- `flowers`: operator materialized a compact edit (`accepted=true`,
  `accepted_faces=36`, `vertices_added=108`), render/eval still running.

### v8 Patch-Fold PatchCert

Purpose: test the new method interface where the grown patch carrier itself must
pass all train-only folds.

First launched scene:

```text
flowers
```

Output root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v8_patchfold_compactstrat_20260514_flowers
```

Log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v8_patchfold_compactstrat_20260514_logs/flowers.out
```

W&B group:

```text
phase_s_patchcert_v8_patchfold_compactstrat_20260514
```

Additional fixed args beyond v7:

```text
--delta_patch_cert_crossfold_folds 4
--delta_patch_cert_crossfold_min_passing_folds 4
--delta_patch_cert_crossfold_min_fold_relative_gain 0
--delta_patch_cert_crossfold_min_fold_samples 4
```

Status at log creation: running.

### v8.1 Neighbor-Fold PatchCert

Purpose: close the scientific loophole found in review. v8 only certified the
aggregate patch carrier; a weak neighbor could still be masked by a strong seed.
v8.1 requires neighbor admission to be fold-aware and records a post-shrink
materialized patch certificate.

First launched scene:

```text
flowers
```

Output root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v81_neighborpatchfold_compactstrat_20260514_flowers
```

Log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v81_neighborpatchfold_compactstrat_20260514_logs/flowers.out
```

W&B group:

```text
phase_s_patchcert_v81_neighborpatchfold_compactstrat_20260514
```

Additional fixed arg beyond v8:

```text
--delta_patch_cert_neighbor_crossfold
```

Status at update: running.

### v8.1 Interim Audit

The v8.1 `flowers` operator completed and confirms that the intended new gate
was active:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v81_neighborpatchfold_compactstrat_20260514_flowers/flowers/model/surface_residual_facelocal_sh1_delta_audit.json
```

Observed audit facts:

```text
accepted=true
accepted_faces=18
vertices_added=54
patch_neighbor_crossfold=true
rejected_neighbor_crossfold=43
accepted_post_shrink_patch_crossfold=3
```

The v8.1 `bicycle` fixed-policy run was also launched under:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v81_neighborpatchfold_compactstrat_20260514_bicycle
```

Its operator audit already shows a safe no-op fallback:

```text
accepted=false
accepted_faces=0
no_op_copy=true
```

Final train-val gate decisions and held-out report-only deltas are still
pending for both scenes.

### v8.2 Strict Carrier Integrity Fix

A second method review found three remaining loopholes that are small in code
but important for paper credibility:

1. Candidate-plan export could serialize all face candidates rather than only
   final certified accepted faces.
2. Plan materialization could replay rows without rechecking that they carried
   explicit certification metadata.
3. A patch could pass before shrink and then be accepted without a hard
   post-shrink policy-val gain check; a global face budget could also slice a
   certified patch carrier after growth.

Implemented fixes in:

```text
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
```

New behavior:

- `--materialize_allow_uncertified_plan` defaults to `false`; plan replay now
  rejects rows without `policy_pass=true`, without `final_certified_face=true`,
  or with failed PatchCert metadata.
- `candidate_plan_out` writes `final_certified_accepted_faces_only`, not the
  broader pre-gate candidate list.
- every patch must still pass `patch_cert_min_policy_val_samples` and
  `patch_cert_min_relative_gain` after shrink;
- `max_faces_to_apply` is enforced at whole-patch granularity, so a certified
  carrier is accepted or rejected as a unit;
- `--patch_cert_neighbor_crossfold` now raises if
  `--patch_cert_crossfold_folds <= 1`, avoiding silent inert configuration.

Validation after the fix:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
```

Both checks passed.

### v8.2 Fixed-Policy Launch

At `2026-05-14 14:57 PDT`, the strict-carrier v8.2 protocol was launched on
the two scenes that currently matter most for the paper-facing claim:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v82_strictcarrier_20260514_flowers
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v82_strictcarrier_20260514_bicycle
```

W&B group:

```text
phase_s_patchcert_v82_strictcarrier_20260514
```

The launch uses the same fixed Phase-J reference methods as v6/v7/v8/v8.1 and
adds the hard v8.2 carrier-integrity checks:

```text
--delta_patch_cert_crossfold_folds 4
--delta_patch_cert_crossfold_min_passing_folds 4
--delta_patch_cert_neighbor_crossfold
--delta_patch_cert_shrink
--gate_compact_enable
```

Interim process evidence:

```text
flowers: active operator runner under phase_s_patchcert_v82_strictcarrier_20260514_flowers
bicycle: active operator runner under phase_s_patchcert_v82_strictcarrier_20260514_bicycle
```

The v8.2 row is still not claimable until both scene-level decision JSON files,
held-out report-only metrics, train-val gate metrics, and qualitative panels are
materialized.

### v8.3 Strict Preset and Plan-Replay Closure

A follow-up review found that v8.2's direct fitting path was strong enough for
the launched flowers/bicycle jobs, but the generic plan-replay interface could
still undermine the paper wording:

- `materialize_plan_limit` or `materialize_plan_face_ids` could slice a
  certified patch carrier into row subsets;
- `materialize_plan_scale` or per-face alpha JSON could alter coefficients
  after certification;
- replay checked row metadata but did not require a complete patch certificate
  or verify that every face in `patch_certificate.faces` was present in the
  replayed rows;
- `force_apply` could be combined with candidate-plan export and produce
  misleading rows.

The code now adds a named strict preset:

```text
--strict_patchcert_carrier
--delta_strict_patchcert_carrier
```

Strict mode requires patch growth, patch-fold certification, neighbor fold
admission, patch shrink, and non-inert fold thresholds.  Certified plan replay
now rejects limit/face-id slicing, coefficient scaling, alpha replay, non-final
export policies, missing PatchCert metadata, missing crossfold/post-shrink
certificates, and split patch carriers.  Legacy plan replay remains possible
only with the explicitly named ablation escape hatch:

```text
--materialize_allow_uncertified_plan
--delta_facelocal_materialize_allow_uncertified_plan
```

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  --source_model /tmp/no_model --evidence_dir /tmp/no_evidence \
  --output_model /tmp/no_out --strict_patchcert_carrier \
  --patch_cert_rings 1 --patch_cert_crossfold_folds 4 \
  --patch_cert_crossfold_min_passing_folds 4 \
  --patch_cert_neighbor_crossfold --materialize_plan_in /tmp/no_plan \
  --materialize_plan_scale 0.5
```

The second command fails before reading a checkpoint with the expected strict
replay rejection: a scale change would alter certified coefficients.

The already running v8.2 jobs should therefore be treated as direct-path
fold-aware PatchCert evidence.  The next paper-facing row should use a new v8.3
label with `--delta_strict_patchcert_carrier`, so the audit files explicitly
record the strict preset.

### v8.3 Fixed-Policy Launch

At `2026-05-14 15:12 PDT`, the strict-preset row was launched on the same two
gatekeeper scenes:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v83_strictpreset_20260514_flowers
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v83_strictpreset_20260514_bicycle
```

W&B group:

```text
phase_s_patchcert_v83_strictpreset_20260514
```

Both launches forward:

```text
--delta_strict_patchcert_carrier
--delta_patch_cert_crossfold_folds 4
--delta_patch_cert_crossfold_min_passing_folds 4
--delta_patch_cert_neighbor_crossfold
--delta_patch_cert_shrink
--gate_compact_enable
```

The expected evidence files are:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v83_strictpreset_20260514_flowers/decisions/flowers_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v83_strictpreset_20260514_bicycle/decisions/bicycle_decision.json
```

These are still pending.

### v7/v8 Final Gate Results

The first completed fold-aware ablations are negative:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v7_crossfold_compactstrat_20260514_summary/summary_2scene.md
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v8_patchfold_compactstrat_20260514_summary/summary_flowers.md
```

Results:

| row | scenes | accepted | note |
|---|---:|---:|---|
| v7 seed-fold PatchCert | 2 | 0 | `bicycle` operator no-op; `flowers` rejected by balanced/tail/LPIPS gate |
| v8 aggregate patch-fold PatchCert | 1 | 0 | `flowers` same final gate rejection as v7 |

`flowers` v7/v8 report-only held-out deltas are near numerical zero:

```text
dPSNR=+0.0000038
dSSIM=-0.0000003
dLPIPS=-0.0000005
```

This is a useful ablation result, not a success row.  It shows that fold-aware
carrier certification improves method hygiene but does not by itself create a
visible or gate-stable gain.

Qualitative ablation panels:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v7_crossfold_compactstrat_20260514_qualitative/patchcert_qualitative_contact_sheet.png
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v8_patchfold_compactstrat_20260514_qualitative/patchcert_qualitative_contact_sheet.png
```

### v8.3 Replay Hardening Follow-Up

A second strict-replay audit found one remaining high-severity carrier split
case: carrier completeness was checked before row-level certification, so a
plan could contain all patch rows, fail one row later, and materialize only the
surviving subset.  The strict path now raises if any row-level certification
failure occurs after plan parsing.

Additional hardening:

- `materialize_plan_scale` must be finite and exactly `1.0` in strict replay;
- strict replay requires `plan_meta.strict_patchcert_carrier=true`;
- duplicate face rows are rejected;
- rows in the same patch must agree on the exact `patch_certificate.faces` set.

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  --source_model /tmp/no_model --evidence_dir /tmp/no_evidence \
  --output_model /tmp/no_out --strict_patchcert_carrier \
  --patch_cert_rings 1 --patch_cert_crossfold_folds 4 \
  --patch_cert_crossfold_min_passing_folds 4 \
  --patch_cert_neighbor_crossfold --materialize_plan_in /tmp/no_plan \
  --materialize_plan_scale nan
```

The second command fails before checkpoint loading with the expected strict
scale rejection.

### v8.4 Final Strict-Validator Launch

Because v8.3 was launched before the row-level replay hardening above landed,
a final same-protocol v8.4 row was launched from the hardened code:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v84_strictvalidator_20260514_flowers
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v84_strictvalidator_20260514_bicycle
```

W&B group:

```text
phase_s_patchcert_v84_strictvalidator_20260514
```

This row is the intended paper-facing strict-carrier evidence row if it passes
the same train-val gate and held-out report-only checks.  v8.2/v8.3 remain
useful ablations and debugging evidence, but they should not supersede v8.4.

## Honest Read

v6 remains the latest completed Phase-S evidence row. v7/v8/v8.1/v8.2 are not
claimable until their train-val gate decisions, report-only held-out metrics,
and qualitative outputs are complete. v8.3 is the first code path whose strict
carrier preset also covers plan replay, but it is not a result row until it is
run under the same fixed protocol.

If v8.1 accepts `flowers` with comparable held-out gains, the method story
becomes stronger: the positive compact outdoor result is no longer only a
compact gate override; the patch neighbors and final materialized carrier are
also supported by train-fold certificates. If v8.1 rejects or collapses the
gain, the correct conclusion is that v6 was too permissive for a paper-facing
representation edit and should remain an ablation or diagnostic rather than the
endpoint. If v8.2 changes v8.1 materially, v8.2 supersedes v8.1 because it
closes the plan/materialization and whole-patch integrity loopholes.
