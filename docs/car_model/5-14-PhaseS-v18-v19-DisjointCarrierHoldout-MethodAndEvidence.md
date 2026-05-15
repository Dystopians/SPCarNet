# Phase-S v18/v19 Disjoint Carrier Holdout Method And Evidence Log

Date: 2026-05-14

Scope: documentation-only audit of the current SPCarNet Phase-S v18/v19 bicycle evidence under `outputs/carnet/meshsplatopt/ecsr_phase_s`.  This log does not claim a completed method win.  The only completed decision rows found here still select the Phase-J fallback.

Update note: this file was first written while v19b/top1 were still incomplete.
The completed v19b/top1/top2/v20 auto-prefix results and fixed portfolio policy
are now recorded in
[`5-14-PhaseS-v20-AutoPrefix-Portfolio-Policy.md`](5-14-PhaseS-v20-AutoPrefix-Portfolio-Policy.md).
That newer note supersedes stale "pending" statements in this file.

## Source Set

Primary source log:

- `docs/car_model/SPCarNet_research_log.md`, especially the 2026-05-14 20:55 PDT v17 entry.  It records the policy-val carrier-holdout repair and states v17 was still `NOT COMPLETE`, needing decisions, metric summaries, qualitative panels, and comparison against Phase-J / clean MeshSplat baselines.

Primary artifact roots:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19_disjoint_sampleholdout_chartquad_key_20260514`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top1_bicycle_20260514`

## Method Modules

`surface_residual_facelocal_sh_delta`

- Applies a checkpoint-level surface-attached residual carrier to a fixed Phase-F compact model:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`.
- Uses train-only surface evidence from:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16/bicycle`.
- Writes a real Mesh Splatting checkpoint copy plus `surface_residual_facelocal_sh1_delta_audit.json`.
- The audit explicitly reports `test_usage: none`; held-out test metrics are report-only downstream.

Face and patch evidence filters:

- Start from high-error, view-supported faces (`selected_faces=3397` for bicycle in v18/v19b).
- Require minimum view hits, pixel count, consistency, face-view gain certificates, consensus, and patch crossfold checks.
- Use seed rescue to recover candidate patches when ordinary support is too sparse.

Cluster-basis carrier:

- Fits a shared `chart_quad` SH carrier basis for accepted patch clusters.
- Main config fields include `patch_cert_cluster_basis=true`, `patch_cert_cluster_basis_mode=chart_quad`, `patch_cert_cluster_basis_steps=240`, `patch_cert_cluster_basis_lr=0.025`, `patch_cert_cluster_basis_min_samples=32`, and `patch_cert_cluster_basis_max_scale=2`.
- This is a representation edit on the checkpoint, not an image-space postprocess.

Validation shrink:

- Uses global validation shrink (`validation_shrink_mode=global`) on policy-val evidence before final carrier materialization.
- v18 full bicycle reports `global_scale=0.9591384360470306`; v19b reports `global_scale=0.9579285889172244`.

Strict PatchCert carrier:

- Requires the final applied carrier to correspond to certified support instead of replaying an uncertified subset.
- In the launch commands this is `--strict_patchcert_carrier`.

Carrier holdout selector:

- v17 introduced policy-val-only carrier holdout after a v16 audit issue: v16 grouped all train views, allowing fitted train views to vote in the holdout certificate.
- v18 uses sample-balanced carrier holdout from the policy-val train split, but not a disjoint split between policy tuning and holdout.
- v19/v19b add disjoint sample holdout: separate policy-val tuning samples from carrier holdout samples.

Render/ELA/gate module:

- Renders the edited checkpoint and applies the Phase-J-style evidence lumigraph adapter for train-val and report-only held-out test comparison.
- The gate compares candidate train-val metrics against `ours_26000_phasej_trainval_gate_rendercalib_v1_top1_s2_fair`.
- Held-out test comparison uses `ours_26000_phasej_guarded_adaptedge_ela_replay_rendercalib_v1_top1_s2_fair` only as report-only evidence.

Compact stratified gate:

- Adds compactness and stratified risk checks on top of the standard train-val gate.
- For bicycle v18 full, the standard tail gate fails and compact gate also fails due stratified PSNR.

## Version Distinctions

| Version | Main change | Evidence status | Current interpretation |
|---|---|---:|---|
| v17 | Policy-val carrier holdout repair after v16 all-train grouping flaw. Strict carrier mode requires carrier holdout. | Research log only in current inspected source set. | Audit repair, not completed evidence. |
| v18 full/chartquad | Sample-balanced carrier holdout over policy-val train split; chart-quadratic cluster basis; can apply a multi-face carrier. | Completed bicycle decision JSON/MD. Flowers artifacts exist but are not summarized here. | Real operator edit, but bicycle final gate rejects and falls back to Phase-J. |
| v18 top1 | Same v18 family but restricts carrier holdout selector to one carrier (`--patch_cert_carrier_holdout_max_carriers 1`). | Completed bicycle summary and decision. | Even smaller real edit; final gate rejects and falls back to Phase-J. |
| v19 | Adds disjoint sample holdout request (`--patch_cert_carrier_holdout_disjoint`). | Incomplete/crashed before usable metrics. | Not evidence. The log ends with `TypeError: summarize_crossfold_face_gain() got an unexpected keyword argument 'holdout_samples'`. |
| v19b | Disjoint sample holdout after the v19 crash fix. Uses `policy_val_tuning_samples=11336` and `carrier_holdout_disjoint_samples=11336`. | Partial bicycle operator and held-out test metrics exist; train-val gate appears unfinished in the log and no decision JSON was found. | Promising only as an audit-cleaner design step; final decision pending. |
| v19b top1 | Intended top1 disjoint sample-holdout variant. | Only `bicycle/model/cameras.json` found under the inspected root. | Pending/no result. |

## Evidence Cleanliness Notes

- Test leakage: the operator audit reports `test_usage: none`; selection/gate decisions are based on train-val policy evidence, with held-out test metrics marked report-only.
- Phase-J references are explicit in decision JSONs:
  - train-val baseline: `ours_26000_phasej_trainval_gate_rendercalib_v1_top1_s2_fair`
  - held-out test baseline: `ours_26000_phasej_guarded_adaptedge_ela_replay_rendercalib_v1_top1_s2_fair`
- v18 does not solve carrier-holdout independence completely: it uses policy-val train split samples for both tuning/certification style logic and holdout grouping.  It is cleaner than v16 all-train grouping but weaker than disjoint holdout.
- v19b improves cleanliness by splitting policy-val evidence into `policy_val_tuning_samples=11336` and `carrier_holdout_disjoint_samples=11336`, with `carrier_holdout_disjoint_from_policy_tuning=true`.
- v19 has no usable evidence because the run crashed before producing operator metrics.
- v19b has no final decision JSON in the inspected artifact root.  Its train-val ELA step appears in the log but no completed train-val metrics or final gate decision were found.
- Existing v18/v18 top1 decisions set `selection_uses_test=false`; report-only test deltas must not be used to promote an accepted row.

## Commands And Config

Runtime:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python
```

v18 full/chartquad operator command is recorded in:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/bicycle/phasek_barycentric_gate.log
```

Key operator settings from the command/audit:

```text
--source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model
--evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16/bicycle
--iteration 26000
--top_k 16384
--strength 0.035
--max_abs_delta_rgb 0.05
--lambda_mag 0.03
--lambda_smooth 0.1
--lambda_sh1_mag 0.06
--sh_degree 3
--uniform_barycentric
--validation_shrink_mode global
--patch_cert_cluster_basis
--patch_cert_cluster_basis_mode chart_quad
--patch_cert_cluster_basis_steps 240
--patch_cert_neighbor_crossfold
--patch_cert_shrink
--patch_cert_seed_rescue
--patch_cert_carrier_holdout_selector
--strict_patchcert_carrier
```

v18 top1 differs in the logged command by:

```text
--patch_cert_carrier_holdout_max_carriers 1
```

v19 disjoint command additionally includes:

```text
--patch_cert_carrier_holdout_disjoint
```

but this run failed with:

```text
TypeError: summarize_crossfold_face_gain() got an unexpected keyword argument 'holdout_samples'
```

v19b disjoint audit confirms:

```text
policy_val_all_samples: 22672
policy_val_tuning_samples: 11336
carrier_holdout_disjoint_samples: 11336
carrier_holdout_disjoint_from_policy_tuning: true
crossfold_face_gain_certificate.certificate_type: reserved_policy_val_sample_fold_consistency
patch_cert_carrier_holdout_disjoint: true
```

## Result Paths

v18 full/chartquad bicycle:

- decision JSON: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/decisions/bicycle_decision.json`
- decision MD: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/decisions/bicycle_decision.md`
- operator audit: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/bicycle/model/surface_residual_facelocal_sh1_delta_audit.json`
- candidate test metrics: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/bicycle/model/test_results.json`
- Phase-J train-val/test refs: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/bicycle/phasej_trainval_gate_results.json`, `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_chartquad_key_20260514/bicycle/phasej_test_results.json`

v18 top1 bicycle:

- summary JSON/MD: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514/phasek_barycentric_gate_summary.json`, `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514/phasek_barycentric_gate_summary.md`
- decision JSON/MD: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514/decisions/bicycle_decision.json`, `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514/decisions/bicycle_decision.md`
- operator audit: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v18_sampleholdout_top1_bicycle_20260514/bicycle/model/surface_residual_facelocal_sh1_delta_audit.json`

v19 bicycle:

- failed log: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19_disjoint_sampleholdout_chartquad_key_20260514/bicycle/phasek_barycentric_gate.log`
- only partial model setup files found: `bicycle/model/cameras.json`, `bicycle/model/cfg_args`, `bicycle/model/input.ply`

v19b bicycle:

- operator audit: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/bicycle/model/surface_residual_facelocal_sh1_delta_audit.json`
- candidate test metrics: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/bicycle/model/test_results.json`
- Phase-J refs: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/bicycle/phasej_trainval_gate_results.json`, `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/bicycle/phasej_test_results.json`
- incomplete gate log: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_chartquad_key_20260514/bicycle/phasek_barycentric_gate.log`

v19b top1:

- only found: `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v19b_disjoint_sampleholdout_top1_bicycle_20260514/bicycle/model/cameras.json`

## Quantitative Comparison: Bicycle Rows Available Now

All deltas below are candidate minus Phase-J reference.  Test rows are report-only unless a completed decision says otherwise.

| Row | Final decision | Applied carrier | Train-val dPSNR | Train-val dSSIM | Train-val dLPIPS | Train-val balanced delta | Gate/tail status | Report-only test dPSNR | Report-only test dSSIM | Report-only test dLPIPS |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| v18 full/chartquad bicycle | rejected, selected Phase-J | 11 faces / 33 vertices | +0.000026703 | +0.000013411 | -0.000026673 | +0.000828385 | fails tail CVaR and compact stratified PSNR | -0.000083923 | -0.000006795 | +0.000011981 |
| v18 top1 bicycle | rejected, selected Phase-J | 1 face / 3 vertices | +0.000000000 | +0.000000000 | -0.000000298 | +0.000005960 | fails tail negative fraction and LPIPS positive fraction; compact PSNR also below threshold | +0.000000000 | +0.000000000 | -0.000000089 |
| v19b partial/chartquad bicycle | pending, no decision JSON found | 10 faces / 30 vertices | pending | pending | pending | pending | pending; train-val gate log appears incomplete | -0.000076294 | -0.000006557 | +0.000011981 |

Absolute held-out test metrics for the relevant Phase-J reference on bicycle:

| Reference | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J test reference `ours_26000_phasej_guarded_adaptedge_ela_replay_rendercalib_v1_top1_s2_fair` | 24.021543503 | 0.702356577 | 0.266087502 |

Absolute candidate held-out test metrics:

| Candidate | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v18 full/chartquad candidate test | 24.021459579 | 0.702349782 | 0.266099483 |
| v18 top1 candidate test | 24.021543503 | 0.702356577 | 0.266087413 |
| v19b partial/chartquad candidate test | 24.021467209 | 0.702350020 | 0.266099483 |

Operator-side proxy evidence is much stronger than final rendered evidence, which is the key failure mode:

| Row | Final accepted policy-val proxy samples | Proxy relative gain | Final rendered/gated outcome |
|---|---:|---:|---|
| v18 full/chartquad | 353 | 0.869068 | rejected by train-val tail/compact gate |
| v18 top1 | 51 | 0.920441 | rejected by train-val tail/compact gate |
| v19b partial/chartquad | 166 | 0.822869 | final train-val gate pending |

## Explicit Failures Versus Phase-J And MeshSplat Baselines

Against Phase-J:

- v18 full is not a win.  It has small positive aggregate train-val PSNR/SSIM/LPIPS deltas but fails the tail CVaR gate.  Its held-out report-only test metrics regress on all three axes: PSNR lower, SSIM lower, LPIPS higher.
- v18 top1 is not a win.  It is essentially metric-noise on full-frame render metrics and still fails tail gates.
- v19 is not a result.  It crashed before metrics.
- v19b is not yet a result.  It improves evidence cleanliness through disjoint sample holdout, but the available held-out test row is still slightly worse than Phase-J on PSNR/SSIM/LPIPS and the train-val decision is pending.

Against clean MeshSplatting / MeshSplat baseline:

- No row here directly demonstrates superiority over clean MeshSplatting.  These rows are evaluated primarily as deltas over the strong Phase-J fallback.
- Because Phase-J is already the stronger previous endpoint in the Phase-S notes, failing to beat Phase-J means the v18/v19 rows cannot be promoted as the new method endpoint.
- v18/v19 should therefore be described as representation-cleanliness and carrier-risk experiments, not as a MeshSplat baseline improvement.
- Any clean MeshSplat comparison remains pending for these v18/v19 rows unless a separate same-protocol collector is run and recorded.

## Qualitative Output Status

Qualitative contact sheets/panels for v18/v19/v19b were not found in the inspected v18/v19 roots.  Mark as pending:

- v18 full/chartquad bicycle qualitative panel: pending.
- v18 top1 bicycle qualitative panel: pending.
- v19 disjoint bicycle qualitative panel: not applicable until the crash is fixed and rerun.
- v19b disjoint bicycle qualitative panel: pending after completed train-val decision.
- v19b top1 qualitative panel: pending/no result.

## Current Honest Status

`NOT COMPLETE`.

The current evidence supports this narrow statement only: v18/v19b are real attempts to clean up Phase-S carrier certification, and v19b specifically makes carrier holdout disjoint from policy tuning samples.  The available completed bicycle decisions do not beat Phase-J.  v19b has only partial evidence and its currently available held-out test metrics are slightly negative against Phase-J.  There is no basis here to claim a Phase-J or MeshSplatting baseline win.

Next placeholders to fill, if the work continues:

- v19b train-val ELA completion and final decision JSON/MD.
- v19b top1 real operator audit, train-val metrics, held-out report-only metrics, and decision.
- Qualitative panels for accepted or near-accepted bicycle rows.
- A same-protocol clean MeshSplat comparison table only after final v19b decisions exist.
