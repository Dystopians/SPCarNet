# Phase-S PatchCert v14-v15 Closed-Loop Method Log

Date: 2026-05-14

## Scope

This log records the current Phase-S direct PatchCert carrier line after the
v10d-v13 negative evidence.  It separates real method changes from diagnostic
or legacy ablations, and records the exact artifact roots needed for fair
review.

## Current Method Modules

1. Face-local residual SH carrier

   The method duplicates only the three vertices of accepted mesh faces and
   writes bounded SH residual coefficients to the local copies.  It preserves
   triangle count and avoids changing vertices shared by unrelated faces.

   Implementation: `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`

2. Train-only policy validation

   Residual support is split into fit views and policy-val train views.  A face
   or carrier must improve the residual proxy on held-out train evidence before
   it can be materialized.  No held-out test residuals are used for selection.

3. Strict PatchCert carrier

   A carrier is a whole connected patch of faces.  It must pass patch-level
   proxy gain, four-fold train-only PatchCert crossfold, neighbor admission
   crossfold, post-shrink policy validation, and strict whole-carrier replay
   checks.  The replay path now validates cluster-basis metadata and carrier
   integrity instead of allowing row-level slicing.

4. Cluster/chart carrier basis

   Instead of giving every face independent local coefficients, PatchCert can
   refit a shared carrier basis.  Current modes include shared, scaled, rank2,
   chart_linear, and chart_quad.  v14/v15 use `chart_quad`.

5. Seed rescue

   v14 adds fixed group-first seed rescue for scenes where strict single-face
   intersections leave too few PatchCert seeds.  Rescue candidates still need
   policy-val eligibility and auxiliary train-only witnesses, and they still go
   through the full PatchCert carrier gate.  This fixes the earlier bicycle
   hard no-op failure mode, but does not by itself solve render-metric tail
   risk.

6. Risk-floor + global shrink

   v15 keeps the v14 carrier family but adds global policy-val shrink and a
   stricter post-shrink PatchCert proxy-gain floor.  This is meant to remove
   carriers that pass basic certification but are too weak or risky in rendered
   SSIM/tail metrics.

7. Legacy metric-aware selector ablation

   The `patchrisk_metricaware_v14_legacy` run is explicitly labeled legacy
   because it allows old candidate-plan rows that lack strict PatchCert carrier
   metadata.  It is useful as a diagnostic/ablation, not as the paper-facing
   strict method.

8. v16 whole-carrier holdout selector

   v16 changes the promotion unit from a face/patch list to a whole
   PatchCert carrier.  After strict PatchCert growth, each carrier is scored on
   deterministic train-only view-holdout groups.  A carrier must pass the fixed
   group count and MSE-regression checks before materialization, and replay
   validation now rejects a strict plan that omits a passing holdout
   certificate.  Held-out test metrics remain report-only.

## Completed Evidence

### v14 Seed Rescue + ChartQuad

Artifact root:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14_seedrescue_chartquad_20260514`

W&B group:
`phase_s_patchcert_v14_seedrescue_chartquad_20260514`

Summary:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14_seedrescue_chartquad_20260514_summary/summary_2scene.md`

Qualitative panels:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14_seedrescue_chartquad_20260514_qualitative/patchcert_qualitative_contact_sheet.png`

Result:

| scene | train-val dPSNR | train-val dSSIM | train-val dLPIPS | train-val balanced | report-only test balanced | decision |
|---|---:|---:|---:|---:|---:|---|
| bicycle | +0.000891 | -0.000281 | -0.000018 | -0.004373 | -0.000466 | rejected |
| flowers | +0.000036 | -0.000013 | +0.000004 | -0.000307 | +0.000007 | rejected |
| garden control | +0.000032 | +0.000000 | -0.000000 | +0.000037 | +0.000023 | accepted |

Interpretation:

v14 is a real method change because bicycle no longer collapses to `accepted_faces=0`; however, it is not a paper-level result.  The added carriers can improve PSNR locally but still damage SSIM/tail stability.  Garden passes only with a tiny effect size.

### v14b Seed Rescue Min-Aux=2 Ablation

Artifact root:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14b_seedrescue2_chartquad_20260514_bicycle`

Status:
completed for bicycle and rejected as expected.  It is too strict: 7 rescue
seeds were proposed, but zero carriers survived PatchCert, so it falls back to
a no-op model with zero metric delta.

### v15 Risk-Floor + Global Shrink

Artifact root:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v15_riskfloor_shrink_chartquad_20260514`

W&B group:
`phase_s_patchcert_v15_riskfloor_shrink_chartquad_20260514`

Status:
bicycle completed; flowers is still running as of this update.

Current operator evidence for bicycle:

| field | value |
|---|---:|
| strict face candidates | 0 |
| rescued face candidates | 16 |
| accepted faces | 11 |
| vertices added | 33 |
| accepted patches | 2 |
| global shrink scale | 0.959138 |

Interpretation:

v15 has reduced v14's accepted bicycle support from 16 to 11 faces and removed
the weak low-proxy carrier.  This fixed the large mean-metric failure:
bicycle train-val balanced improved to `+0.000828`, with `dPSNR=+0.000027`,
`dSSIM=+0.000013`, and `dLPIPS=-0.000027`.  It still rejected because the
train-val balanced CVaR tail was `-0.000115`, just below the `-0.0001` gate.
Report-only held-out test remained negative (`-0.000459` balanced).  The
remaining bottleneck is therefore not seed starvation but carrier tail risk.

### v16 Whole-Carrier Holdout Selector

Artifact root:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v16_carrierholdout_chartquad_20260514`

W&B group:
`phase_s_patchcert_v16_carrierholdout_chartquad_20260514`

Status:
running on bicycle/flowers.

Fixed policy:

- carrier holdout groups: `4`
- minimum passing groups: `3`
- minimum group relative gain: `0`
- maximum group MSE regression: `0`
- CVaR fraction/weight: `0.25 / 1.0`
- selection unit: whole PatchCert carrier, never sliced rows

Rationale:

v15 showed positive mean train-val quality but a small negative view-tail CVaR.
v16 directly attacks that failure by requiring each carrier to be stable on
train-only view groups before it can affect rendering.

Post-launch review found that the first v16 implementation grouped all train
views, which meant three of four carrier-holdout groups could come from views
also used by the coefficient fit.  This is not test leakage, but it is not a
clean holdout certificate.  v16 is therefore retained only as a diagnostic run,
not as the paper-facing method.

### v17 Policy-Val-Only Whole-Carrier Holdout

Artifact roots:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v17_policyholdout_chartquad_key_20260514`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v17_policyholdout_chartquad_controls_20260514`

W&B group:
`phase_s_patchcert_v17_policyholdout_chartquad_20260514`

Status:
running in two parallel fixed-policy jobs:

- key stress scenes: `bicycle,flowers`;
- control scenes: `garden,counter,bonsai`.

Method correction:

- the carrier holdout cache now uses only the train policy-val split
  (`policy_val_train_split`), not the train-fit views;
- strict PatchCert carrier mode now requires the carrier holdout selector;
- strict replay validates `patch_cluster_basis.passed`, carrier face agreement,
  and multi-face `applied` status on the normal passing path;
- the whole-carrier selector rejects a carrier if an upstream stage has only
  accepted part of that carrier, instead of silently redefining a smaller
  carrier.

Interpretation:

v17 is the current main method line.  It may be stricter and reject more often,
but it closes the audit gap that made v16 unsuitable as a final research claim.

### Legacy PatchRisk Metric-Aware Selector

Artifact root:
`outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchrisk_metricaware_v14_legacy_20260514_counter`

W&B group:
`phase_s_patchrisk_metricaware_v14_legacy_20260514`

Known completed row:

| trial | scene | train-val dPSNR | train-val dSSIM | train-val dLPIPS | train-val balanced | report-only test balanced | decision |
|---|---|---:|---:|---:|---:|---:|---|
| patchrisk2_s0p75 | counter | +0.000080 | +0.000001 | -0.000000 | +0.000100 | +0.000090 | accepted |

Interpretation:

The result is useful as an ablation showing that train-val metric-aware
selection can find tiny positive edits on counter.  It cannot be used as the
strict paper method because it relies on explicitly allowed uncertified legacy
plan rows.

## Diagnostics

Starvation scan archive:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14_seedrescue_chartquad_20260514_summary/starvation_v14_v15_probe.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v14_seedrescue_chartquad_20260514_summary/starvation_v14_v15_probe.json`

Key finding:

The main bottleneck moved from hard seed starvation to render-metric tail risk.
v14 fixed bicycle's zero-seed problem but did not produce stable rendered
quality.  v15 reduced the failure to a small train-val tail-CVaR miss.  v16
introduced whole-carrier holdout selection, and v17 repairs its audit flaw by
restricting the certificate to the train policy-val split only.

## Current Stop/Go Assessment

This is not complete.  The strict method still has not shown multi-scene,
fair, train-val accepted gains over the Phase-J/MeshSplat-derived baseline.
The active gate is v17:

- If v17 accepts bicycle and flowers, collect summaries, build qualitative
  panels, and promote the fixed policy to the next multi-scene table.
- If v17 still rejects bicycle/flowers, inspect policy-val-only
  carrier-holdout rows to decide
  whether the bottleneck is no stable carrier, proxy/render mismatch, or too
  coarse a surface-attached residual basis.
