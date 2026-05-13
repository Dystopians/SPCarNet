# 2026-05-12 Representation Upgrade Loop

This note records the post full9-collector attempt to move Phase-S beyond
threshold tuning.  The goal was a real train/eval-pipeline method change that
could close the remaining `bicycle`, `counter`, and `treehill` Phase-S weakness
under the same train-only held-out gate protocol.

## Implemented Method Changes

### 1. Face-Local Full-SH Residuals

Code paths:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

Changes:

- added `--sh_degree {1,2,3}` to the face-local residual operator;
- kept default `sh_degree=1` so historical SH1 runs remain reproducible;
- added SH2/SH3 basis evaluation in the same coefficient order as
  `utils/sh_utils.py`;
- materialized non-DC coefficients into stored `features_rest` channels;
- exposed the runner-side flag as `--delta_sh_degree`.

### 2. Surface Subdivision Residuals

Code paths:

- `scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

Changes:

- added `--delta_operator subdivision` to the Phase-K barycentric gate runner;
- replaced selected high-error triangles with four coplanar sub-triangles;
- added midpoint vertices with bounded train-only residual DC deltas;
- kept topology validity checks (`degenerate_face_count`, `invalid_index_count`);
- added audit-side `policy_pass`;
- added a view-level gain certificate for subdivision candidates:
  `--min_view_gain_views`, `--min_view_gain_relative_gain`,
  `--min_view_gain_samples`, and `--min_view_gain_fraction`;
- added train-only multi-offset face validation:
  `--policy_val_offsets`, `--min_policy_val_offsets`, and
  `--min_policy_val_offset_fraction`;
- each candidate face is now fit/evaluated across requested train-only support
  partitions before materialization; passing folds are averaged into the final
  residual delta, while held-out test views remain report-only;
- after code review, the final averaged delta is re-evaluated on every
  train-only validation offset before `policy_pass` is set; this supersedes the
  initial v8/v9 evidence where fold-specific deltas were certified before
  averaging;
- the runner now forces barycentric evidence for subdivision operators even if
  legacy `--delta_uniform_barycentric` is present;
- recorded the subdivision budget cap as `max_faces_to_apply` in new audits.

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check
```

Both passed after the implementation updates.

## SH3 Evidence

SH3 smoke on `counter` produced a valid checkpoint-level representation change:

- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/operator_smoke/facelocal_sh3_20260512/counter/model`
- accepted faces: `163`
- vertices added: `489`
- topology: triangles unchanged, degenerate faces `0`, invalid indices `0`
- fit proxy relative gain: `0.126139`
- policy-val proxy relative gain: `0.188982`

Hard-scene single-gate results:

| scene | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | true | +0.000002 | +0.000000 | +0.000000 | +0.000000 | -0.000000 | +0.000000 | accepted but negligible |
| counter | false | -0.000004 | -0.000000 | -0.000001 | -0.000299 | -0.000014 | +0.000050 | rejected by train-val PSNR |
| treehill | false | -0.000687 | +0.000000 | -0.000005 | -0.000481 | -0.000002 | -0.000003 | rejected by train-val PSNR |

Decision: SH3 is a correct implementation milestone, but not a scientific
closure.  More coefficient capacity did not produce meaningful held-out gains.

## Subdivision Evidence

Shared evidence root for v1-v6:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/surface_evidence_subdivision_v1_20260512`

Dense evidence root for v7:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/surface_evidence_subdivision_dense48_v7_20260512`

### Treehill

| variant | single gate | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS | strict 4-offset |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v1 aggressive | false | +0.001389 | -0.000067 | -0.000026 | -0.000690 | -0.000032 | -0.000044 | not run |
| v2 conservative | true | +0.001535 | -0.000046 | -0.000015 | -0.000040 | -0.000028 | -0.000030 | false: offsets 2/3 fail |
| v3 ssimsafe | true | +0.001379 | -0.000047 | -0.000020 | +0.000284 | -0.000027 | -0.000027 | false: offsets 2/3 fail |
| v4 viewcert | true | +0.000444 | -0.000001 | +0.000002 | +0.000158 | -0.000001 | -0.000005 | false: offsets 2/3 fail |
| v7 dense48 viewcert | false | -0.000008 | -0.000001 | -0.000002 | -0.000002 | +0.000000 | -0.000001 | not run |

Strict v4 details:

- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v4_viewcert_20260512_treehill/treehill/multifold_trainval_gate.md`
- offset 0: dPSNR `+0.002197`, dSSIM `+0.000431`, dLPIPS `-0.000782`
- offset 1: dPSNR `+0.001064`, dSSIM `+0.000103`, dLPIPS `-0.000911`
- offset 2: dPSNR `-0.000437`, dSSIM `-0.000499`, dLPIPS `-0.001115`
- offset 3: dPSNR `-0.001875`, dSSIM `-0.000483`, dLPIPS `+0.000058`

Interpretation: subdivision creates real local effects and can improve some
train splits, especially LPIPS, but the gains are not robust across policy-val
offsets.  Dense support plus stricter view certificates made the method nearly
no-op instead of solving the instability.

### Counter

| variant | single gate | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v1 aggressive | false | -0.004698 | -0.000178 | +0.000392 | +0.005713 | +0.000040 | -0.000891 | train-val fail, test improves |
| v2 conservative | false | +0.001545 | -0.000057 | -0.000233 | +0.002878 | +0.000017 | -0.000183 | failed by SSIM threshold |
| v3 ssimsafe | false | -0.000343 | -0.000134 | -0.000181 | +0.002457 | +0.000313 | -0.000381 | train-val PSNR/SSIM fail |
| v4 viewcert | false | -0.000521 | -0.000099 | -0.000119 | +0.012033 | -0.000039 | -0.000869 | train-val PSNR/SSIM fail |
| v5 viewcert budget16 | false | -0.000608 | -0.000070 | -0.000239 | -0.000025 | -0.000001 | +0.000002 | train-val PSNR/SSIM fail |
| v6 viewcert budget8 | false | -0.000257 | -0.000038 | +0.000088 | -0.000044 | -0.000000 | +0.000000 | train-val PSNR fail |
| v7 dense48 viewcert | false | -0.000462 | -0.000107 | -0.000366 | -0.028734 | +0.000134 | -0.000349 | train-val PSNR/SSIM fail |

The strongest report-only counter test row was v4:

- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/subdivision_v4_viewcert_20260512_counter`
- accepted subdivision faces: `34`
- triangles: `9644247 -> 9644349`
- test delta: PSNR `+0.012033`, SSIM `-0.000039`, LPIPS `-0.000869`

It is not selectable under the current fair gate because train-val PSNR and
SSIM regress.

### Recovery Attempt

Because counter v4 had a real report-only test signal, a topology-frozen 500
iteration recovery run was launched:

- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/recovery_subdivision_v4_viewcert_counter_500iters_20260512`
- W&B run:
  `counter_subdivision_v4_viewcert_recovery_500iters_20260512`
- command log:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/recovery_subdivision_v4_viewcert_counter_500iters_20260512/logs/train_recovery.log`
- train flags included:
  `--freeze_topology_updates`, `--skip_restricted_delaunay`,
  `--feature_lr 0.0008`, `--weight_lr 0.015`,
  `--lr_triangles_points_init 0.0002`

Recovery test ELA result:

- recovery test ELA: PSNR `28.473688`, SSIM `0.895481`, LPIPS `0.184238`
- Phase-J counter test baseline: PSNR `28.449171`, SSIM `0.893731`,
  LPIPS `0.186472`
- test deltas: PSNR `+0.024517`, SSIM `+0.001750`,
  LPIPS `-0.002235`

Recovery train-val gate result:

- recovery train-val: PSNR `29.451817`, SSIM `0.902312`, LPIPS `0.174200`
- Phase-J train-val: PSNR `29.477457`, SSIM `0.902502`, LPIPS `0.171491`
- train-val deltas: PSNR `-0.025640`, SSIM `-0.000190`,
  LPIPS `+0.002709`

Decision: recovery confirms that the subdivision signal can improve the held-out
test split, but it fails the train-only policy-val gate badly.  It is therefore
not a valid paper-claim result under the current protocol.

## Multi-Offset Robust Policy Evidence

The next change was not another parameter sweep.  The subdivision operator was
upgraded so every face-level residual must pass multiple train-only validation
offsets before it is applied.  This directly targets the observed failure mode:
single-offset improvements on `counter` and `treehill` did not survive strict
four-offset validation.

Important correction: the initial v8/v9 implementation certified each
offset-specific delta before averaging the passing deltas.  Code review found
that this did not certify the actual final materialized delta.  The authoritative
v10 runs below use the corrected policy: fit per-offset deltas, average the
selected deltas, then re-evaluate that final delta on every requested
train-only validation offset before allowing materialization.

New command/config interface:

- runner flags:
  `--delta_policy_val_offsets`, `--delta_min_policy_val_offsets`,
  `--delta_min_policy_val_offset_fraction`
- operator flags:
  `--policy_val_offsets`, `--min_policy_val_offsets`,
  `--min_policy_val_offset_fraction`
- strict policy used for `counter` v8:
  offsets `0,1,2,3`, minimum offsets `4`, fraction `1.0`
- relaxed diagnostic policy used for `treehill` v9:
  offsets `0,1,2,3`, minimum offsets `3`, fraction `0.75`
- corrected final-delta diagnostic policy used for `treehill` v10:
  offsets `0,1,2,3`, minimum offsets `3`, fraction `0.75`

Key output paths:

- `counter` v10 single gate:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/subdivision_v10_finaldelta_20260512_counter`
- `counter` v10 strict gate:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v10_finaldelta_20260512_counter/counter/multifold_trainval_gate.json`
- `treehill` v10 relaxed 3/4 diagnostic:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/subdivision_v10_finaldelta3of4_gain0_20260512_treehill`
- `treehill` v10 strict gate:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v10_finaldelta3of4_gain0_20260512_treehill/treehill/multifold_trainval_gate.json`

Qualitative render outputs were also generated with the metric runs:

- `counter` v10 test renders:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/subdivision_v10_finaldelta_20260512_counter/counter/model/test/ours_26000_subdivision_v10_finaldelta_phasej_ela/renders`
- `treehill` v10 test renders:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/subdivision_v10_finaldelta3of4_gain0_20260512_treehill/treehill/model/test/ours_26000_subdivision_v10_finaldelta_phasej_ela/renders`

Single-gate results:

| scene/variant | policy | accepted | faces | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter v10 | 4/4 offsets, final-delta certified | true | 1 | +0.000006 | +0.000000 | -0.000001 | +0.000006 | -0.000000 | -0.000000 |
| treehill v10 | 3/4 offsets, final-delta certified | true | 59 | +0.000639 | -0.000032 | -0.000015 | +0.000008 | -0.000021 | -0.000033 |

Strict four-offset train-val gates:

| scene/variant | accepted | offset 0 | offset 1 | offset 2 | offset 3 | interpretation |
|---|---:|---|---|---|---|---|
| counter v10 | true | dPSNR `+0.000006`, dSSIM `+0.000000`, dLPIPS `-0.000001` | dPSNR `+0.000008`, dSSIM `-0.000000`, dLPIPS `-0.000001` | dPSNR `+0.000000`, dSSIM `+0.000000`, dLPIPS `-0.000001` | dPSNR `+0.000000`, dSSIM `+0.000000`, dLPIPS `-0.000001` | robust but effectively no-op |
| treehill v10 | false | fail PSNR: dPSNR `-0.001774`, dSSIM `+0.000266`, dLPIPS `+0.000061` | fail SSIM: dPSNR `+0.000198`, dSSIM `-0.000457`, dLPIPS `-0.000269` | fail PSNR/SSIM: dPSNR `-0.000071`, dSSIM `-0.000270`, dLPIPS `-0.000556` | pass: dPSNR `+0.000319`, dSSIM `+0.000112`, dLPIPS `-0.000160` | final-delta policy still not split-stable |

Decision: the multi-offset policy is a real reliability improvement, and v10
fixes the final-delta certification weakness.  It prevents the earlier invalid
`counter` recovery-style selection and gives a strictly passing `counter`
candidate.  However, the strict candidate changes only one face and yields
1e-6 scale metric deltas, so this is not a paper-level method win.  On
`treehill`, a 59-face update passes the single gate but still fails strict
four-offset validation.  The method family is now better audited, but the core
operator remains too weak: robustness collapses it to no-op on `counter`, while
stronger local edits remain split-unstable on `treehill`.

## Current Scientific Assessment

The project remains `NOT COMPLETE`.

What improved:

- Phase-S now has real representation-level mechanisms, not only hyperparameter
  changes: SH3 face-local residuals, topology-changing local subdivision,
  a subdivision view-gain certificate, and multi-offset train-only face
  validation.
- The new subdivision branch can create non-trivial test improvements on
  `counter`, and recovery amplified this to a sizeable report-only test gain.
- The audit trail is stronger: commands used W&B, result paths are fixed,
  qualitative render outputs exist, and strict multi-offset evidence was run for
  the hard-scene candidates.
- The v8 policy directly addresses overfitting risk by requiring per-face
  residual gains on multiple train-only partitions before writing topology.

What is still blocking:

- `counter` now passes strict validation only by applying a one-face near-no-op;
  this is reliable, but not scientifically strong.
- `treehill` can pass the relaxed single gate, but fails strict four-offset
  validation because offsets 0/1/2 expose PSNR or SSIM regressions.
- Dense evidence and stricter view certificates reduced risk but also collapsed
  the effective update into a near no-op.
- Recovery training generated the best counter test result, but it worsened the
  train-only gate, so it cannot be used as a fair selected method.

The hard lesson is that the current local residual/subdivision family can find
localized test improvements, but it does not yet learn a split-robust correction
policy with enough effect size.  The next credible method change should move
from DC-only midpoint materialization to a more expressive but still
train-certified representation, such as subdivision-local SH residuals or
view-support clustered surface residual codes trained against multiple
train-only support partitions.

## SH1/Luma/Anchor and Render-Calibrated Prefix Attempts

This follow-up round upgraded the subdivision branch from DC-only midpoint
residuals to subdivision-local SH1 residuals, added explicit luma-preserving
projection, and added a candidate-plan replay interface for render-calibrated
prefix tests.  These are real pipeline changes, but they still do not close the
paper-level Phase-S claim.

New code interfaces:

- `--feature_mode sh1` and `--max_abs_sh_coeff` in
  `scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py`;
- runner mirrors `--delta_subdivision_feature_mode` and
  `--delta_subdivision_max_abs_sh_coeff`;
- SH1 midpoint deltas write DC to `features_dc` and the first three SH channels
  to `features_rest`;
- `--luma_preserve`, `--luma_shrink_grid`, and
  `--luma_shrink_selection` add train-only SH1 DC-luma projection;
- `--anchor_support` adds low-error in-face anchors to constrain updates on
  already-stable pixels;
- `--candidate_plan_out` and `--materialize_plan_in` allow replaying a selected
  candidate subset into a checkpoint before running the real render gate.

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py

git diff --check
```

Key evidence paths:

- `counter` v11b strict SH1:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v11b_sh1_boundsfix_finaldelta_20260512_counter/counter/multifold_trainval_gate.json`
- `treehill` v11b strict SH1:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v11b_sh1_boundsfix_finaldelta3of4_20260512_treehill/treehill/multifold_trainval_gate.json`
- `treehill` v12 SH1 luma-max:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v12_sh1_lumamax_finaldelta3of4_20260512_treehill/treehill/multifold_trainval_gate.json`
- `treehill` v13 SH1 luma + anchor:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v13_sh1_anchor_lumamax_finaldelta3of4_20260512_treehill/treehill/multifold_trainval_gate.json`
- `treehill` v14/v15/v16 render-calibrated prefix replays:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v14_rendercalib_v12top8_20260512_treehill/treehill/multifold_trainval_gate.json`,
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v15_rendercalib_v12top4_20260512_treehill/treehill/multifold_trainval_gate.json`,
  and
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/subdivision_v16_rendercalib_v12top1_20260512_treehill/treehill/multifold_trainval_gate.json`.

Strict four-offset results:

| variant | accepted | faces | mean dPSNR | mean dSSIM | mean dLPIPS | main failure |
|---|---:|---:|---:|---:|---:|---|
| counter v11b SH1 | true | 1 | +0.000003 | +0.000000 | -0.000001 | passes but near no-op |
| treehill v11b SH1 | false | 64 | +0.000660 | +0.000014 | -0.000519 | offset PSNR/SSIM failures |
| treehill v12 luma-max | false | 64 | +0.000672 | -0.000004 | -0.000443 | offset 0/1 PSNR and offset 1/2 SSIM |
| treehill v13 anchor | false | 11 | -0.000038 | -0.000546 | -0.001203 | SSIM fails on all offsets |
| treehill v14 v12-top8 replay | false | 8 | -0.000247 | -0.000090 | -0.000585 | offsets 2/3 fail PSNR/SSIM |
| treehill v15 v12-top4 replay | false | 4 | -0.000023 | +0.000026 | -0.000155 | offsets 0/3 fail |
| treehill v16 v12-top1 replay | false | 1 | -0.000354 | -0.000018 | -0.000013 | offset 3 fails PSNR/SSIM |

Important observations:

- SH1 residuals produce real LPIPS improvements on `treehill`, often much
  larger than the DC-only branch, so the added representation capacity is not a
  no-op.
- Luma projection improves some report-only/single-gate behavior but does not
  fix strict split instability.
- Low-error anchors make the update more conservative in single-gate testing,
  but under strict four-offset rendering they still introduce SSIM regressions.
- Render-calibrated prefix replay is useful diagnostically: top-8 passes offsets
  0/1 but fails 2/3; top-4 passes 1/2 but fails 0/3; top-1 passes 0/1/2 but
  still fails offset 3.  This proves a simple top-prefix policy is insufficient.

Updated decision: the current method remains `NOT COMPLETE`.  We now have a
clearer blocker, not merely missing experiments: local face-prefix selection
cannot provide a nontrivial treehill update that passes all strict train-only
offsets.  The next credible step is a true render-calibrated combinatorial or
greedy acceptance policy that uses real train-val render feedback to accept or
reject candidate groups, or a stronger structure-preserving representation loss
that directly targets SSIM instead of relying on local RGB/luma proxies.
