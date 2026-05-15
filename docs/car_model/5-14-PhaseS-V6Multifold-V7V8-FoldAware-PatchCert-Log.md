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

Interim v8.4 operator audits:

```text
flowers: accepted=true, accepted_faces=18, vertices_added=54, strict_patchcert_carrier=true
bicycle: accepted=false, accepted_faces=0, vertices_added=0, strict_patchcert_carrier=true
```

The flowers carrier is the same compact three-patch / eighteen-face structure
seen in v8.1-v8.3, now produced from the hardened strict-validator code.  The
bicycle result remains a safe no-op, which is scientifically useful but does
not improve coverage.  Final decision files are still pending.

Additional live evidence at `2026-05-14 15:36 PDT`:

```text
flowers base held-out test:
  LPIPS=0.3947871029, PSNR=19.6687068939, SSIM=0.5116778612
flowers Phase-J held-out reference:
  LPIPS=0.3295054734, PSNR=20.3006076813, SSIM=0.5574578047
flowers Phase-J train-val reference:
  LPIPS=0.2972038686, PSNR=20.8552265167, SSIM=0.6471784711

bicycle base held-out test:
  LPIPS=0.3322745562, PSNR=23.2934818268, SSIM=0.6596511602
bicycle Phase-J held-out reference:
  LPIPS=0.2660875022, PSNR=24.0215435028, SSIM=0.7023565769
```

This interim table is deliberately not a promotion result.  It records that the
surface-attached strict edit alone is far below the Phase-J appearance-adapted
reference on held-out RGB.  The runner is still computing the candidate
appearance-adapted train-val/test row and the final train-val gate.  Until that
decision lands, v8.4 is best interpreted as a strict carrier-integrity
experiment, not a solved paper endpoint.

Final v8.4 summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v84_strictvalidator_20260514_summary/summary_2scene.md
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v84_strictvalidator_20260514_qualitative/patchcert_qualitative_contact_sheet.png
```

Result:

| scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | Phase-J fallback | false | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| flowers | Phase-J fallback | false | +0.000042 | -0.000013 | +0.000004 | +0.000000 | +0.000000 | -0.000000 |

v8.4 therefore closes as a negative but useful integrity row: the hardened
strict carrier replay works, but fixed train-val promotion accepts `0 / 2`
scenes.

### v9 Patch-Cluster Shared-Basis Carrier

Subagent method-gap review found that v7/v8/v8.4 mostly tightened certificate
integrity while leaving the direct edit capacity almost unchanged. The next
real method change is therefore a representation-level carrier constraint
rather than another threshold scan.

Implemented change:

- the face-local residual-SH operator now supports
  `--patch_cert_cluster_basis`;
- when enabled, each accepted multi-face PatchCert carrier is refit with one
  shared three-corner residual-SH basis copied across the faces in that carrier;
- the fit uses train-only residual samples and compares its MSE against the
  independent face-local fit on the same samples;
- if the shared basis regresses beyond the fixed
  `--patch_cert_cluster_basis_max_fit_mse_regression` bound, the carrier is
  restored to its pre-refit coefficients and rejected under strict mode;
- policy-val, shrink, and patch crossfold certificates are evaluated after the
  shared basis is materialized.

Important wording constraint: this is a shared corner-slot SH carrier basis,
not a continuous geometric patch basis. It should be reported as a stronger
representation prior over a certified carrier, not as mesh topology repair.

Validation before scene launch:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
```

Additional review:

- a static subagent review found no correctness or backward-compatibility bug
  in the diff;
- a runner forwarding smoke confirmed that
  `--delta_patch_cert_cluster_basis*` is forwarded to the materializer as
  `--patch_cert_cluster_basis*`;
- a synthetic CPU unit smoke confirmed that the shared-basis fitter can improve
  a two-face synthetic carrier and writes non-zero shared coefficients.

Scene pilots launched at `2026-05-14 15:48 PDT`:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v9_clusterbasis_20260514_flowers
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v9_clusterbasis_20260514_bicycle
```

W&B group:

```text
phase_s_patchcert_v9_clusterbasis_20260514
```

Decision: `NOT COMPLETE`. v9 is the first post-v8 attempt that changes the
carrier representation itself. It still needs full runner decisions,
train-val/held-out metrics, and qualitative panels before it can be compared
against v6/v8.4/Phase-J.

Early v9 operator audit on `flowers` found an important failure mode:

```text
accepted=true, accepted_faces=5, vertices_added=15,
accepted_cluster_basis=0, rejected_cluster_basis=6,
accepted_patches=0, mean_patch_size=1.0
```

This means the strict shared-basis multi-face carriers were all rejected by the
fit-regression bound and the run fell back to single-face certificates.  The
v9 row must therefore not be described as a shared-basis success unless a later
decision shows accepted shared carriers.  This failure motivates v10.

Final v9 summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v9_clusterbasis_20260514_summary/summary_2scene.md
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v9_clusterbasis_20260514_qualitative/patchcert_qualitative_contact_sheet.png
```

Result: fixed train-val promotion accepts `0 / 2` scenes.  `bicycle` is a
candidate no-op, and `flowers` is rejected by the balanced/tail/LPIPS gate with
numerical-zero held-out change.  This closes v9 as a negative representation
ablation.

### v6 Multifold Train-Val Gate Follow-Up

The four-offset train-val fairness check for the last positive v6 row now has a
two-scene summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/phase_s_patchcert_v6_compactstrat_gate_20260514/summary/summary_2scene.md
```

Result:

| scene | accepted | reason | report test dPSNR | report test dSSIM | report test dLPIPS |
|---|---:|---|---:|---:|---:|
| bicycle | false | offset2 PSNR gain below 0 | +0.000387192 | +0.000035524 | -0.000115275 |
| flowers | true | all offsets pass | +0.001676559 | +0.000158310 | -0.000304669 |

Mean effective held-out delta after falling back on rejected scenes:

```text
dPSNR=+0.000838280
dSSIM=+0.000079155
dLPIPS=-0.000152335
```

This is useful evidence that v6 was not purely a single-offset artifact, but it
also confirms that the effect size is small and not uniformly accepted.

### v10 Scaled Shared-Basis Carrier

To avoid the v9 collapse from over-tying all faces to identical coefficients, a
second low-rank carrier parameterization was implemented:

- `--patch_cert_cluster_basis_mode shared` preserves v9 behavior;
- `--patch_cert_cluster_basis_mode scaled` fits one shared three-corner SH basis
  plus one positive scale per face in the carrier;
- `--patch_cert_cluster_basis_max_scale` bounds those face scales;
- the same train-only fit-regression, policy-val, shrink, and patch crossfold
  certificates still decide whether the carrier survives.

This is still a carrier-level representation prior, not a topology or UV-chart
method.  The intended effect is to preserve cross-face support sharing while
allowing different residual amplitudes on neighboring faces.

### v10c Audited Scaled Carrier

A focused implementation review found that v10b's numerical path was mostly
self-consistent, but the evidence trail was not yet strong enough for a paper
claim.  The following audit hardening was added before the next pilot:

- `--patch_cert_cluster_basis_max_scale <= 0` is rejected by both the
  materializer and runner;
- scaled carriers now record `face_scales`, `effective_max_scale`,
  `coeff_clamped_count`, `coeff_total_count`,
  `coeff_clamped_fraction`, and `coeff_max_clamp_excess`;
- strict certified plan replay now rejects non-finite coefficients and
  coefficients outside the saved DC/SH bounds;
- candidate-plan metadata now stores cluster mode, steps, learning rate, min
  samples, max scale, fit-regression threshold, init mode, DC bound, and SH
  bound;
- row payloads now include explicit `pre_cluster_policy_val_proxy`,
  `post_cluster_policy_val_proxy`, and `post_cluster_patch_certificate` fields
  while preserving legacy names for compatibility.

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  docs/car_model/5-14-PhaseS-V6Multifold-V7V8-FoldAware-PatchCert-Log.md \
  docs/car_model/SPCarNet_research_log.md
```

A small direct unit check verified that strict plan replay rejects an
out-of-bounds coefficient row with `delta_coeff_out_of_strict_bounds`.

Scene pilots launched:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10c_scaledcluster_audit_20260514_flowers
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10c_scaledcluster_audit_20260514_bicycle
```

W&B group:

```text
phase_s_patchcert_v10c_scaledcluster_audit_20260514
```

Decision: `NOT COMPLETE`.  v10c supersedes v10b as the claimable scaled-carrier
pilot because it has the required carrier-scale and strict-replay audit fields.

v10c `flowers` then exposed a real shape bug in the scaled predictor:
face-local samples carry three local corner ids per pixel, so the per-face scale
must broadcast over both the corner and SH dimensions.  The predictor now uses
a rank-aware scale view instead of a fixed `(-1, 1, 1)` view.

Validation after the fix:

- `py_compile` passed again;
- `git diff --check` passed again;
- a direct shape smoke with `sample_vertex_ids` shaped `[N, 3]` verified that
  scaled mode applies, records two `face_scales`, and does not clamp the test
  coefficients.

The fixed `flowers` rerun is:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10d_scaledcluster_shapefix_20260514_flowers
```

W&B group:

```text
phase_s_patchcert_v10d_scaledcluster_shapefix_20260514
```

## Honest Read

v6 remains the latest completed Phase-S evidence row. v7/v8 final summaries are
negative ablations. v8.1/v8.2/v8.3 are useful implementation and integrity
evidence, and v8.4 is the intended strict-validator result row because it was
launched after the final row-level replay hardening landed.

The v8.4 operator result is scientifically mixed before the final decision:
`flowers` proves the hardened strict direct path can materialize a tiny
certificate-carrying representation edit, while `bicycle` proves the same
policy can safely no-op when certificates are insufficient.  That is good
method hygiene, but it does not solve the paper goal by itself.  The open
question is whether the candidate row after the normal Phase-J appearance
adapter can beat the Phase-J fallback under the fixed train-val gate.  If not,
the honest conclusion is that fold-aware PatchCert is an integrity upgrade and
ablation, not the next headline method.

The v9 shared-basis carrier is the current active method upgrade. It is more
research-relevant than another gate tweak because it changes how residual
capacity is tied across a certified patch carrier, but it is not yet evidence
of a paper-ready gain. It should become a claim only if the fixed scene pilots
produce accepted decisions and visible/quantitative improvements over the
Phase-J fallback.
