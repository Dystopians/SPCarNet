# Phase-S Shared Residual-Field Operator

Date: 2026-05-16

Status: `NOT COMPLETE - PROXY_TO_RENDER_MISMATCH CONFIRMED`. This note records the first real method change after
the v21/v22 PatchCert carrier bottleneck. The goal is to stop treating Phase-S
as a carrier-size or hyperparameter search problem and instead change the
representation update itself.

## Motivation

The previous Phase-S evidence showed a consistent failure mode:

- v21 `rank2` PatchCert did not improve the strict portfolio.
- v22 coverage-aware auto-prefix produced real non-noop edits on
  `garden`, `bonsai`, and `room`, but strict effect-aware promotion accepted
  `0 / 3`.
- The report-only render deltas were mostly `1e-6` to `1e-5`, so full-frame
  visual changes were weak even when the train proxy looked positive.

This suggests that the local face/patch coefficients are too sparse and too
low-amplitude. The next operator therefore fits a shared residual field across
the train-certified face pool, then bakes that field back into the existing
face-local checkpoint representation.

## Implemented Method

File:

```text
scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
```

New operator path:

```text
--shared_residual_field
--shared_residual_field_anchors
--shared_residual_field_sigma
--shared_residual_field_weight_l2
--shared_residual_field_view_hinge_weight
--shared_residual_field_view_hinge_min_samples
--shared_residual_field_duplicate_smooth_weight
```

Instead of solving one independent SH residual coefficient row for every
duplicated local vertex slot, the operator builds deterministic RBF features
over the selected train-evidence local mesh slots:

```text
feature(v) = [1, normalized_xyz(v), rbf_1(v), ..., rbf_K(v)]
delta_SH(v) = bounds * tanh(feature(v) @ theta)
```

The learned field is still materialized by the existing
`materialize_facelocal()` path, so no renderer changes are required:

```text
shared residual field -> per-corner delta_coeff -> duplicated local vertices
```

The fitting loss remains train-only:

```text
weighted residual MSE
+ lambda_mag * DC^2
+ lambda_sh1_mag * SH-rest^2
+ lambda_smooth * local edge smoothness
+ field_weight_l2 * theta^2
+ view_hinge_weight * per-train-view regression hinge
+ duplicate_smooth_weight * same-source-vertex coefficient disagreement
```

The audit now records:

- `operator=surface_residual_facelocal_shared_field_delta`
- RBF anchor count, feature count, sigma, and parameter count
- view-hinge and duplicate-source smoothness settings
- policy-val proxy and final accepted proxy
- `accepted`, `policy_pass`, and `no_op_copy`

Phase-K runner and auto-visual pipeline now forward the shared-field arguments:

```text
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

The 2026-05-16 continuation also added an opt-in carrier selector guard:

```text
--patch_cert_carrier_holdout_auto_prefix_positive_tail_safe
--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe
```

When enabled, auto-prefix stops before the first carrier whose individual
holdout certificate has a negative score, negative tail group, or nonzero CVaR
loss. This prevents the coverage floor from forcing a risky carrier into the
materialized checkpoint only to satisfy a face-count target. The audit records
the blocked carrier, whether the requested minimum face count was relaxed, and
the effective selected face count.

## Smoke Evidence

Low-cost direct operator smoke on `garden`:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/shared_residual_field_smoke_20260516/garden/model/surface_residual_facelocal_sh1_delta_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_s/shared_residual_field_smoke_20260516/garden/shared_field_candidate_plan.json
```

Result:

| field | value |
|---|---:|
| operator | `surface_residual_facelocal_shared_field_delta` |
| selected faces | 123 |
| accepted faces | 39 |
| vertices added | 117 |
| shared-field anchors | 8 |
| shared-field params | 576 |
| policy-val relative gain | +0.003621842 |
| final accepted policy-val relative gain | +0.076231383 |
| no-op copy | false |
| degenerate faces | 0 |
| invalid indices | 0 |

The shell wrapper returned nonzero only because an outer `tee` target directory
did not exist; the Python operator wrote the checkpoint and audit successfully.

## Full Render-Gate Evidence

All full pilots below used online W&B and the same rule that held-out test
metrics are report-only; selection is controlled by train-val decision JSON.

| run | operator | accepted faces | train-val dPSNR | train-val dSSIM | train-val dLPIPS | balanced delta | tail neg frac | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v1 global shrink | non-noop shared field | 18 | +0.000003815 | -0.000000179 | +0.000000417 | -0.000008106 | 0.341463 | rejected |
| v2 face gain | non-noop shared field | 15 | +0.000003815 | -0.000000179 | +0.000000313 | -0.000006020 | 0.341463 | rejected |
| v3 positive-tail-safe | non-noop shared field | 6 | -0.000003815 | -0.000000060 | +0.000000015 | -0.000005305 | 0.170732 | rejected |

Artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_sharedfield_v1_garden_retry_20260516/decisions/garden_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_sharedfield_v2_facegain_garden_20260516/decisions/garden_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_sharedfield_v3_tail_safe_garden_20260516/decisions/garden_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_sharedfield_v3_tail_safe_garden_20260516/garden/model/surface_residual_facelocal_sh1_delta_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_sharedfield_v3_tail_safe_garden_20260516_qualitative/qualitative_summary.md
```

The v3 positive-tail-safe selector changed the failure mode in a useful way:
it reduced train-val tail negative fraction from `0.341463` to `0.170732`,
blocked a carrier with score `-0.008111864` and CVaR loss `0.079199173`, and
materialized only the safest carrier. It still did not create a positive
full-render delta. The best report-only qualitative rows remain at `1e-6` to
`1e-5` metric deltas, which is not visually meaningful.

## Promotion Rule

This operator should only be treated as progress if it beats the strict v22
failure mode, not merely if it writes a non-noop checkpoint. Minimum useful
evidence:

- `selection_uses_test=false`
- operator audit available
- `policy_pass=true`
- `no_op_copy=false`
- strict train-val gate accepts at least one hard pilot scene, or the full9
  strict effect-aware portfolio increases from `3 / 9` to at least `4 / 9`
  with non-noise deltas

## Current Assessment

This is a real method change in the train/eval pipeline. It changes the fitted
representation update from independent local rows to a shared residual field,
while preserving the existing checkpoint, renderer, train-val gate, W&B, and
portfolio machinery.

It is not yet a completed scientific result. The repeated garden pilots now
show the central bottleneck directly: strong local policy-val proxy gains do
not transfer to measurable full-frame render gains after Phase-J ELA. The next
method step should not be another carrier threshold sweep. It should introduce
a render-space trust-region certificate: surface proxy proposes a certified
plan, but materialization or residual scale is accepted only when train-val
render metrics improve under the same balanced objective used by the final
gate. Strict plan replay currently forbids arbitrary coefficient scaling, so
that mechanism must be implemented as an audited render-certified replay path,
not by using an uncertified ablation flag.

## Render-Trust Replay Hook

The first infrastructure hook for that next step is now implemented:

```text
scripts/car_model/ecsr_write_render_trust_certificate.py
--materialize_plan_render_trust_json
--delta_facelocal_materialize_plan_render_trust_json
```

The certificate converts a Phase-K train-val decision into an explicit replay
authorization. A non-unit strict `--materialize_plan_scale` is allowed only if
the render-trust certificate is accepted, `selection_uses_test=false`, the
scale matches, and the candidate plan sha256 matches. A negative smoke using
the rejected v3 `garden` decision correctly produced `accepted=false` and strict
materialization failed with
`render_trust_certificate_not_accepted`:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_trust_cert_smoke_20260516/garden_rejected_scale050.json
```

This does not yet create a winning Phase-S row. It closes the safety/interface
gap needed for the next experiment: render-space scale or region search must
prove train-val render improvement before a scaled representation edit can be
replayed as a strict certified checkpoint.
