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

## Render-Trust Scale Replay Result

The first scale replay tested the v3 positive-tail-safe `garden` carrier at
scale `0.5` as an explicitly uncertified ablation, then converted the decision
into a render-trust certificate:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_trust_scale_search_20260516/scale050_trial/decisions/garden_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_trust_scale_search_20260516/garden/scale050_render_trust_certificate.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_trust_scale_search_20260516/scale050_trial_qualitative/qualitative_summary.md
```

Result:

| field | value |
|---|---:|
| candidate | `phase_s_rendertrust_scale050_trial_20260516` |
| accepted | `false` |
| selected fallback | `phasej_guarded_adaptedge` |
| selection uses test | `false` |
| materialized faces | `6` |
| materialize scale | `0.5` |
| train-val dPSNR | `-0.000003815` |
| train-val dSSIM | `-0.000000060` |
| train-val dLPIPS | `-0.000000045` |
| train-val balanced delta | `-0.000004113` |
| train-val tail negative fraction | `0.170732` |
| report-only test balanced delta | `+0.000000596` |
| certificate accepted | `false` |

The qualitative contact sheet exists, but it is diagnostic rather than
paper-facing:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_trust_scale_search_20260516/scale050_trial_qualitative/patchcert_qualitative_contact_sheet.png
```

This confirms that shrinking the same certified residual direction is not
enough. The proxy-to-render mismatch is not just an amplitude problem.

## Next Method Pivot: Render-Visible Region Carriers

The next implemented interface starts from visible render residual blobs rather
than from face-score rows:

```text
scripts/car_model/ecsr_build_render_visible_region_carriers.py
```

It reads train-only surface evidence `views/*.npz`, extracts high-residual
connected image regions, projects those regions back to face ids, merges
multi-view regions into carriers by face overlap, and writes a region-ranked
`top_residual_supports.csv` plus a symlinked `views/` directory. That output is
compatible with the existing face-local fitter, so the next full runner can
change the proposal prior without changing the renderer or final train-val
gate.

The claim boundary is strict: this proposal generator is not a final method row
until a rendered checkpoint passes the same Phase-K train-val gate with
`selection_uses_test=false`. It is the next real method step because v1/v2/v3
and scale `0.5` show that surface proxy ranking alone is not aligned with
visible image-space improvement.

First `garden` proposal/smoke outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/garden/render_visible_region_carriers.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/garden/render_visible_region_carriers.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence/garden/top_residual_supports.csv
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/direct_smoke/garden/model/surface_residual_facelocal_sh1_delta_audit.json
```

The proposal builder scanned `8` train evidence views, extracted `64`
high-residual connected regions, merged them into `49` render-visible carriers,
and exported `1776` region-ranked evidence faces. The direct smoke consumed
that evidence without changing the renderer: it selected `512` faces, accepted
`274`, added `822` local vertices, and wrote a non-noop shared-field checkpoint
plus candidate plan. Its policy-val proxy was positive
(`+0.075074986` final accepted relative gain), but fit proxy was still negative
(`-0.002445628` final accepted relative gain), so it is intentionally treated
only as an interface/proposal smoke. The full Phase-K render-gate validation is
recorded separately under:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_garden
```

Full `garden` Phase-K render-gate result:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_garden/decisions/garden_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_garden/garden/model/surface_residual_facelocal_sh1_delta_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_garden_qualitative/patchcert_qualitative_contact_sheet.png
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_garden/garden/render_visible_region_v1_render_trust_certificate.json
```

| field | value |
|---|---:|
| candidate | `phase_s_rvregion_sharedfield_v1_20260516` |
| accepted | `true` |
| selection uses test | `false` |
| train-val dPSNR | `+0.000070572` |
| train-val dSSIM | `+0.000000000` |
| train-val dLPIPS | `-0.000000611` |
| train-val balanced delta | `+0.000082791` |
| report-only test dPSNR | `+0.000043869` |
| report-only test dSSIM | `-0.000000417` |
| report-only test dLPIPS | `-0.000000089` |
| report-only test balanced delta | `+0.000037313` |
| carriers in proposal | `49` |
| accepted faces | `183` |
| vertices added | `549` |
| final policy-val proxy gain | `+0.173052862` |
| final fit proxy gain | `+0.074261867` |
| render-trust certificate | `accepted=true` |

This is the first Phase-S shared-field variant that passes the strict render
gate after changing the proposal prior. It is meaningful because it validates a
non-noop representation edit selected without test metrics. It is still not a
paper-level visual breakthrough: the held-out test improvement is positive but
near metric-noise scale, and the qualitative panel is diagnostic rather than a
strong full-frame example.

The same fixed carrier policy has also been generated for all nine available
M360 scenes:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/{scene}/render_visible_region_carriers.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence/{scene}/top_residual_supports.csv
```

A collector was added so the multi-scene run can be summarized without manual
copying once the long jobs finish:

```text
scripts/car_model/ecsr_collect_phase_s_regionprior_summary.py
```

The multi-scene validation launched with W&B online in two fixed groups:
`bicycle,flowers,stump,treehill` on GPU 1 and
`bonsai,counter,kitchen,room` on GPU 6. These jobs are the required next
fairness gate before this proposal prior can be promoted beyond the `garden`
single-scene result.

## Full9 Region-Prior Result And Robust Promotion

The full fixed-policy multi-scene validation completed on all nine available
Mip-NeRF360 scenes:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phase_s_regionprior_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phase_s_regionprior_full9_robust_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_indoor_qualitative/patchcert_qualitative_contact_sheet.png
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phasek_regionprior_outdoor_qualitative/patchcert_qualitative_contact_sheet.png
```

Default Phase-K accepts `4 / 9` scenes (`bonsai`, `garden`, `kitchen`,
`treehill`), but this is not safe as a final policy: `bonsai` is accepted by
train-val mean metrics yet has a large negative report-only test delta. The
collector therefore now reports a train-val-only robust promotion layer:

- the original decision must accept and `selection_uses_test=false`;
- mean train-val LPIPS must not regress;
- train-val balanced tail CVaR must be at least `-0.0001`;
- the worst stratified train-val group balanced delta must be at least
  `-0.00001`.

This robust promotion keeps `2 / 9` scenes: `garden` and `kitchen`. It rejects
the default accepted `bonsai` and `treehill` rows without reading test metrics.

| policy | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---:|---:|---:|---:|---|
| default Phase-K fallback | `4 / 9` | `-0.000460307` | `+0.000079963` | `+0.000064320` | unsafe, `bonsai` dominates the negative mean |
| robust promotion fallback | `2 / 9` | `+0.000298606` | `+0.000006563` | `-0.000020499` | safer and all-axis positive, but still small |

Interpretation: this is real progress relative to the failing shared-field
v1/v2/v3 rows because it completes a full9 train-only, non-noop
representation-level policy with positive effective RGB deltas after fallback.
It is not a complete paper-level solution: the gains remain much smaller than
Phase-J-over-clean and the qualitative sheets are diagnostic rather than
visually obvious full-frame wins. The next research step should move from
face-score saliency weighting to true per-view region core/context fitting or a
masked render-space objective.
