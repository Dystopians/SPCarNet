# Fresh Phase-J Replay and Local Gate Audit

Date: 2026-05-13

## Purpose

This note records a fairness and concurrency repair in the Phase-S
representation loop.  Earlier Phase-S rows compared candidate renders against
the default Phase-J test method name:

`ours_26000_phasej_guarded_adaptedge_ela`

That name can reuse stale renders/results already present inside the selected
Phase-J compact model.  This matters because a no-op or near-no-op candidate
can appear to regress only because the baseline row came from an older, more
favorable replay.  The fix is to force a fresh Phase-J ELA replay under a
unique method name for each experiment and to write the train-val/test Phase-J
metrics into the experiment output root, not into a shared compact-model file.

## Code Changes

Implemented in:

- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_collect_phasek_barycentric_gate_summary.py`

Runner changes:

- added `--phasej_test_method`;
- always replays/evaluates Phase-J test under the requested method name;
- writes Phase-J train-val metrics to
  `{output_root}/{scene}/phasej_trainval_gate_results.json`;
- writes Phase-J report-only test metrics to
  `{output_root}/{scene}/phasej_test_results.json`;
- passes those local files into the decision gate.

Collector changes:

- adds `--decision_path_template` for per-scene/per-GPU output roots;
- reports operator-audit missing/rejected/no-op counts;
- reports whether held-out test deltas used a fresh Phase-J replay or the
  default potentially stale reference.

## Triggering Bug

While running parallel fair-replay face-local experiments, the old runner wrote
multiple Phase-J train-val methods into the shared file:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/{scene}/ratio_0200/compact_model/trainval_gate_results.json`

This caused `v1` face-local runs to fail with:

```text
KeyError: 'ours_26000_phasej_trainval_gate_gaincert_v1_fair not found in .../compact_model/trainval_gate_results.json'
```

The failed rows were rerun after the local-gate fix.  The old shared-file
`v2` decisions were preserved as `*.sharedrace.json` / `*.sharedrace.md` and
then regenerated with the fixed runner.

## Experiments

### DC Patch-Cluster v6, Fresh Replay

Output summaries:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v6_fairreplay_highconf_20260513_combined/phasek_barycentric_gate_summary_collected.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v6_fairreplay_highconf_20260513_bicycle`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v6_fairreplay_highconf_20260513_flowers`

| scene | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | false | +0.000101 | +0.000003 | +0.000033 | +0.000597 | +0.000038 | -0.000038 |
| flowers | false | +0.000011 | -0.000000 | +0.000002 | -0.000002 | +0.000000 | -0.000002 |

Interpretation: the fresh reference fixes a measurement problem.  `flowers`
is no longer a large negative against Phase-J; it is essentially neutral.
However, both DC patch-cluster rows still reject under train-val.  This is not
a closed method result.

### Face-Local GainCert v1, Fresh Replay

Output summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_fairreplay_20260513_combined/phasek_barycentric_gate_summary_collected.md`

| scene | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | false | -0.000006 | +0.000000 | +0.000001 | +0.000374 | +0.000035 | -0.000115 |
| flowers | true | +0.000044 | +0.000001 | -0.000001 | +0.005426 | +0.000471 | -0.000588 |

Effective two-scene deltas with train-val fallback:

- mean dPSNR: `+0.002713`
- mean dSSIM: `+0.000235`
- mean dLPIPS: `-0.000294`

### Face-Local GainCert v2 Face-Shrink, Fresh Replay

Output summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v2_faceshrink_fairreplay_20260513_combined/phasek_barycentric_gate_summary_collected.md`

| scene | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | false | -0.000006 | -0.000000 | +0.000000 | +0.000376 | +0.000035 | -0.000115 |
| flowers | true | +0.000044 | +0.000001 | -0.000002 | +0.005426 | +0.000471 | -0.000587 |

The face-shrink policy improves the local residual proxy and slightly improves
the `flowers` train-val balanced score, but it does not fix `bicycle`.  The
result is useful evidence that amplitude calibration alone is not enough for
the hard outdoor scene.

## Current Decision

Status: `NOT COMPLETE`.

The fair-replay repair is important: it prevents stale Phase-J references and
makes multi-GPU experiments reproducible.  Scientifically, the best fresh
result in this batch is `flowers`: face-local GainCert passes train-val and
has a nontrivial report-only test improvement.  `bicycle` remains the blocker:
both v1 and v2 improve report-only test metrics but fail the train-only gate by
a tiny PSNR/balanced margin.

The next method change should target support alignment on `bicycle`, not
another global strength sweep.  A credible next branch is a train-only
per-view/face support-risk analyzer that identifies which accepted local faces
cause the negative train-val PSNR tail, then materializes only the subset whose
gain is stable across held-out train views.
