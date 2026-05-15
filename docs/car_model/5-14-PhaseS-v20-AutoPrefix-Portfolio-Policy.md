# Phase-S v20 Auto-Prefix And Fixed Portfolio Policy

Date: 2026-05-14

Status: `NOT COMPLETE` as a final paper endpoint. This is a real method and
audit milestone, but it does not yet solve broad Phase-S coverage.

## Scope

This note updates the earlier v18/v19 disjoint-carrier log with the completed
v19b/v20 rows and the train-val-only portfolio selector. It records what was
implemented, what was run, where the artifacts live, and what can be claimed
without using held-out test metrics for selection.

## Method Delta

v19b and v20 both operate on fixed Phase-F compact checkpoints and materialize
real Mesh Splatting checkpoint edits through face-local SH1 residual carriers.
The held-out test split is used only after selection for reporting.

The new parts are:

- **Disjoint sample carrier holdout.** The policy-val evidence is split into a
  tuning half and a carrier-certification holdout half. v19b/v20 use
  `sample_balanced` grouping, so carrier holdout does not fall back to fitted
  train views.
- **Strict carrier integrity.** Strict replay now validates carrier holdout and
  cluster-basis metadata on the passing path, not only when a row fails.
- **Carrier auto-prefix.** v20 removes the manual top-k carrier game. It sorts
  carrier rows by train-only carrier-holdout score, scans deterministic
  prefixes, and selects the best cumulative prefix whose certificate passes and
  whose cumulative score is nonnegative.
- **Train-val portfolio policy.** `ecsr_select_phase_s_policy_portfolio.py`
  selects among already-run candidates using train-val balanced delta only.
  Missing/rejected candidates fall back to Phase-J with zero effective delta.
  Test deltas are report-only after selection. After the code review, the
  selector was tightened so a candidate must explicitly record
  `selection_uses_test=false`; missing selection provenance is ineligible
  instead of being silently trusted.

Relevant implementation files:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_select_phase_s_policy_portfolio.py`
- `scripts/car_model/ecsr_fit_facelocal_plan_alphas.py`
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_analyze_patchcert_starvation.py`

## Key Evidence Roots

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top1_bicycle_20260514
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top2_bicycle_20260514
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_portfolio_policy_v1_20260514
outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative
```

## v19b/v20 Quantitative Results

The table reports deltas against the Phase-J fallback used by the Phase-K gate.
Positive PSNR/SSIM and negative LPIPS are better. `accepted` means accepted by
the train-val gate; held-out test deltas are report-only.

| row | scenes | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v19b full, bicycle | 1 | no | +0.000063 | +0.000024 | -0.000010 | -0.000076 | -0.000007 | +0.000012 | real edit, rejected; test regresses slightly |
| v19b full, flowers | 1 | no | +0.000002 | +0.000000 | +0.000000 | +0.000004 | +0.000000 | -0.000000 | near no-op, rejected |
| v19b top1, bicycle | 1 | no | +0.000008 | -0.000001 | -0.000000 | +0.000000 | +0.000000 | -0.000000 | all-metric test non-regression, but train-val rejects |
| v19b top2, bicycle | 1 | no | +0.000008 | n/a | n/a | -0.000076 | -0.000007 | +0.000012 | rejected; similar to full row |
| v20 auto-prefix, bicycle | 1 | no | +0.000008 | -0.000001 | -0.000000 | +0.000000 | +0.000000 | -0.000000 | fixed policy selects top1-like prefix, train-val rejects |
| v20 auto-prefix, flowers | 1 | no | +0.000002 | +0.000000 | +0.000000 | +0.000004 | +0.000000 | -0.000000 | near no-op, train-val rejects |

Primary summaries:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/phasek_barycentric_gate_summary.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top1_bicycle_20260514/phasek_barycentric_gate_summary.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top2_bicycle_20260514/phasek_barycentric_gate_summary.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514/phasek_barycentric_gate_summary.md`

## v20 Carrier Evidence

v20 materializes real checkpoint edits even though the final decision falls back
to Phase-J.

| scene | accepted faces | vertices added | policy-val all samples | tuning samples | disjoint holdout samples | selected carriers | auto-prefix score |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 4 | 12 | 22672 | 11336 | 11336 | 1 | 0.873029 |
| flowers | 2 | 6 | 29402 | 14701 | 14701 | 1 | 0.631411 |

Audit files:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514/bicycle/model/surface_residual_facelocal_sh1_delta_audit.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514/flowers/model/surface_residual_facelocal_sh1_delta_audit.json`

Both audits report:

```text
patch_cert_carrier_holdout_auto_prefix: true
carrier_holdout_disjoint_from_policy_tuning: true
selection_unit: patchcert_carrier
test_usage: none
grouping: sample_balanced
groups: 4
min_passing_groups: 3
```

## Fixed Portfolio Policy Result

The fixed portfolio selector was run over 7 scenes and existing candidate
families:

- `georisk`
- `patchrisk`
- `v19b`
- `v19b_top1`
- `v19b_top2`
- `v20_auto`

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_select_phase_s_policy_portfolio.py \
  --scenes garden,bicycle,room,kitchen,bonsai,flowers,counter \
  --candidate georisk=outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_{scene}/{scene}/coupled_selector_decision.json \
  --candidate patchrisk=outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_{scene}/{scene}/coupled_selector_decision.json \
  --candidate v19b=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/decisions/{scene}_decision.json \
  --candidate v19b_top1=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top1_bicycle_20260514/decisions/{scene}_decision.json \
  --candidate v19b_top2=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top2_bicycle_20260514/decisions/{scene}_decision.json \
  --candidate v20_auto=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514/decisions/{scene}_decision.json \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_portfolio_policy_v1_20260514/portfolio_summary.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_portfolio_policy_v1_20260514/portfolio_summary.md \
  --output_csv outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_portfolio_policy_v1_20260514/portfolio_summary.csv
```

Portfolio summary:

| scene | selected policy | accepted | train-val balanced | effective test dPSNR | effective test dSSIM | effective test dLPIPS |
|---|---|---:|---:|---:|---:|---:|
| garden | Phase-J fallback | no | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| bicycle | Phase-J fallback | no | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| room | Phase-J fallback | no | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| kitchen | Phase-J fallback | no | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| bonsai | Phase-J fallback | no | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| flowers | georisk | yes | +0.000019789 | +0.005418777 | +0.000470877 | -0.000586182 |
| counter | georisk | yes | +0.000101507 | +0.000055313 | +0.000000417 | -0.000001699 |

Mean effective report-only delta over the 7-scene portfolio:

```text
dPSNR:  +0.000782013
dSSIM:  +0.000067328
dLPIPS: -0.000083983
accepted: 2 / 7
selection_uses_test: false
```

The portfolio selector was rerun after the explicit provenance guard was added;
the accepted count remains `2 / 7`, so this fairness hardening did not change
the current result.

## Qualitative Evidence

The strongest current qualitative panels are still the GeoRisk/CVaR accepted
portfolio rows. They show local error-change panels rather than full-frame
visual improvements.

| scene | mode | view | dPSNR | dSSIM | dLPIPS | panel |
|---|---|---|---:|---:|---:|---|
| flowers | accepted positive | `00019.png` | +0.016326904 | +0.001679182 | -0.001603484 | `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/flowers_georisk4_s1_00019_georisk_cvar_panel.png` |
| counter | accepted positive | `00002.png` | +0.000520706 | +0.000003517 | +0.000004441 | `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/counter_georisk4_s1_00002_georisk_cvar_panel.png` |
| garden | rejected false positive | `00006.png` | -0.000154495 | -0.000001371 | +0.000004120 | `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/garden_georisk8_s0p5_00006_georisk_cvar_panel.png` |

README-ready copies:

```text
assets/spcarnet_phase_s_portfolio_flowers_georisk_panel.png
assets/spcarnet_phase_s_portfolio_counter_georisk_panel.png
```

## What This Means

The positive interpretation:

- v20 is no longer manual carrier-count tuning. It is a deterministic,
  train-only auto-prefix policy.
- The carrier evidence is cleaner than v16/v18 because policy tuning samples
  and carrier holdout samples are disjoint.
- The portfolio policy is fixed and explicitly refuses candidates that do not
  pass train-val gates.
- The existing GeoRisk rows give small but real accepted improvements on
  `flowers` and `counter`.

The hard negative interpretation:

- v20 does not beat the Phase-J fallback on the fair train-val gate.
- The best fixed portfolio only accepts `2 / 7` scenes.
- The average gain is tiny and not visually obvious at full-frame scale.
- v20/top1 edits are often near the metric-noise floor: bicycle v20 adds only
  `4` accepted faces and `12` vertices on an approximately `8.31M` triangle
  model, and flowers adds only `2` faces and `6` vertices.
- The dominant failure is tail instability rather than mean quality: rejected
  rows repeatedly trip balanced-CVaR, negative-fraction, or LPIPS-positive-tail
  gates.
- Hard scenes such as `bicycle`, `garden`, `room`, `kitchen`, and `bonsai`
  still fall back to Phase-J.
- This does not yet support a claim that representation-level Phase-S broadly
  dominates Phase-J or the underlying MeshSplatting baseline by itself.

## Completion Checklist

| item | status | evidence |
|---|---|---|
| real method change in train/eval pipeline | done | v20 auto-prefix flags and portfolio selector script |
| baseline/current/improved/ablation run | partial | Phase-J fallback, GeoRisk/PatchRisk, v19b, v19b top1/top2, v20 auto-prefix; not all scenes have v20 decisions |
| metrics saved | done for completed rows | summary JSON/MD/CSV roots listed above |
| qualitative outputs saved | partial | GeoRisk/CVaR panels only; no broad v20 qualitative win |
| commands/configs/errors documented | done in this note and logs | `phasek_barycentric_gate.log`, portfolio command |
| paper story written honestly | partial | usable for slides as audit-clean representation attempt, not final claim |
| weaknesses marked | done | sparse acceptance and tiny visual gain called out explicitly |

## Next Required Work

The next useful step is not another manual top-k scan. It should test a stronger
representation operator that can improve hard scenes without relaxing the
train-val gate:

1. Build a carrier operator with explicit image-space residual target but
   representation-space capacity greater than the current tiny face-local SH1
   edit.
2. Keep v20-style disjoint carrier holdout and train-val portfolio selection.
3. Require the fixed policy to improve more than `2 / 7` scenes before promoting
   it above Phase-J.
4. Generate qualitative panels from accepted rows only and include rejected
   false-positive panels as safety evidence.

Final status for this stage: `NOT COMPLETE`.
