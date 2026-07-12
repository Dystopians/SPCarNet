# v49b Fair-Noop L1-Risk Atlas and v50 Portfolio Log

Date: 2026-06-23

Status: `CLOSED DIAGNOSTIC` / `NOT COMPLETE`.

This log records the fair-noop correction after the v49 L1-risk diagnostic, the
new v49b full9 replay, and the v50 fixed portfolio decision. The
purpose is not to replace the current Phase-J headline. The purpose is to
separate a fair representation-level surface-atlas result from an unsafe
fallback artifact.

## Why v49b exists

The first v49 L1-risk auto-noop replay found an important fairness problem:
when `--write_noop_on_reject` rejected a surface atlas, the fallback path copied
renders and metrics from `source_model/test/<base_method>`. For scenes such as
`garden`, that made the rejected v49 row inherit a stronger source-model render
instead of the same target-evidence no-op compact baseline. The resulting gain
was therefore artificial and must not be used as final evidence.

v49b fixes this by making rejected rows materialize from the target evidence:

```text
--write_noop_on_reject
--noop_fallback_source target_evidence
```

The code path now records the fallback provenance in
`target_apply.fallback_source`, and the default CLI behavior is the fair target
evidence fallback. The older source-model fallback remains available only as an
explicit diagnostic option:

```text
--noop_fallback_source source_model
```

## Code changes

Changed files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
  - added `--noop_fallback_source {target_evidence,source_model}`;
  - changed rejected fallback default to `target_evidence`;
  - records fallback provenance in the audit JSON;
  - keeps source-model fallback as an explicit diagnostic mode only.
- `scripts/car_model/run_l1risk_fairnoop_scene.py`
  - added a fixed scene runner for v49b;
  - reconstructs the v49b command from the existing v48 audit;
  - runs apply and then `metrics.py`;
  - writes logs under `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs`.
  - later parameterized the L1 positive-view fraction and target changed-fraction
    gates, so the same runner can execute the locked v50 policy without copying
    a second script.
- `scripts/car_model/summarize_l1risk_surface_atlas_full9.py`
  - summarizes v49/v49b results against same-evidence no-op and v48;
  - reports aggregate strict/nonregressive wins and per-scene policy diagnostics.

Local compile checks already run before the replay:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  scripts/car_model/summarize_l1risk_surface_atlas_full9.py
```

## Completed v49b replay

Root:

```text
/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623
```

Tag:

```text
v49b_l1risk_fairnoop_autosupport_autocap_guarded_v42calib_region_texture_adapter
```

All nine scenes have now been replayed under the fair target-evidence fallback.
The launch was done in waves on GPUs 1, 4, and 5; the jobs were mostly
CPU-heavy during candidate atlas fitting and policy-val evaluation.

First-wave logs:

| scene | GPU | log |
|---|---:|---|
| `room` | 5 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_room.log` |
| `counter` | 4 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_counter.log` |
| `bonsai` | 1 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_bonsai.log` |
| `bicycle` | 5 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_bicycle.log` |
| `flowers` | 4 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_flowers.log` |
| `garden` | 1 | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_garden.log` |

Remaining-scene logs:

| scene | log |
|---|---|
| `stump` | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_stump.log` |
| `treehill` | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_treehill.log` |
| `kitchen` | `/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/logs/apply_metrics_kitchen.log` |

## v49b full9 summary

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_l1risk_surface_atlas_full9.py \
  --v49_root /dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623 \
  --v49_tag v49b_l1risk_fairnoop_autosupport_autocap_guarded_v42calib_region_texture_adapter \
  --output_json /dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/v49b_fairnoop_full9_summary.json \
  --output_md /dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/v49b_fairnoop_full9_summary.md
```

Official summary paths:

```text
/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/v49b_fairnoop_full9_summary.json
/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/v49b_fairnoop_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623/v49b/v49b_fairnoop_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623/v49b/v49b_fairnoop_full9_summary.md
```

Aggregate:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v49b vs same-evidence no-op | 9 | 2 | 3 | +0.000697 | +0.00001307 | -0.00002279 |
| v49b vs v48 | 9 | 0 | 1 | -0.000765 | -0.00001467 | +0.00001674 |

Accepted scenes:

```text
kitchen, bonsai
```

Main diagnostic:

- v49b is scientifically useful because it fixes rejected fallback provenance.
- It is too conservative as a method endpoint: `room` and `counter` were
  rejected only because `image_l1_positive_view_fraction` was `0.916667`
  instead of `1.0`, despite passing the mean, min-view, CVaR, and SSIM gates.
- `bicycle`, `flowers`, `stump`, and `treehill` have real tail/SSIM/L1
  warnings and should remain rejected by this policy family.

## v50 locked decision rule

v50 should be a fixed train-evidence-only portfolio, not a per-scene parameter
game. The decision can use audit fields produced before held-out test metrics:

- policy-val relative gain;
- policy-val positive-view fraction;
- policy-val min-view and CVaR20 gain;
- policy-val image SSIM mean/min/positive-view gain;
- policy-val image L1 mean/min/CVaR/positive-view gain;
- fit-summary fields: fit/policy-val view counts, fit samples, atlas faces,
  selected support/texture/fill;
- carrier-summary support-expansion fields, because the expansion code skips
  policy-val stride views when mining extra support faces;
- policy-val candidate rows and candidate score ordering.

The decision must not use held-out test PSNR/SSIM/LPIPS. Test metrics are only
allowed for final reporting.

The decision also must not use target/test-camera support as a policy input:

- no `results.json` or `per_view.json`;
- no v48/v49/v49b summary metric deltas;
- no `target_apply.changed_fraction`;
- no target coverage gate;
- no `target_evidence_dir` or target split metadata;
- no post-fallback `accepted/effective_policy/fallback_written` state if target
  coverage or fallback logic has modified the original train-policy decision.

When in doubt, recompute the candidate decision from `policy_val.best`,
`policy_val.rows`, `policy_val.samples`, `fit_summary`, and
`carrier_summary.support_expansion`.

The locked v50 structure is:

1. accept strict v49b rows that pass the L1/SSIM/tail risk gate;
2. use same-evidence no-op fallback for rejected rows;
3. optionally promote v48 rows only if a fixed train-only relaxed-evidence tier
   passes across the audit diagnostics;
4. report any relaxed promotion as a separate ablation unless it is fixed before
   looking at test metrics.

Concrete fixed v50 candidate settings:

```text
--texture_size 16
--texture_size_candidates 8,16,24,32
--support_expansion_mode fit_residual_topk
--support_expansion_max_extra_faces 2048
--support_expansion_min_face_samples 128
--support_expansion_min_mean_l1 0.003
--policy_val_stride 4
--alpha_grid 0,0.015625,0.03125,0.0625,0.125
--min_l1 0.001
--min_atlas_face_samples 32
--atlas_confidence_mode count_var_sign
--atlas_confidence_count_scale 2.0
--atlas_confidence_empty_bin 0.5
--atlas_confidence_variance_scale 0.004
--atlas_confidence_sign_power 0.5
--atlas_confidence_face_sample_scale 256
--min_atlas_confidence 0.02
--atlas_lowpass_passes 1
--atlas_empty_bin_fill_mode auto_policy
--select_alpha_by_risk_gate
--min_policy_val_samples 1024
--min_policy_val_relative_gain 0.0002
--min_policy_val_positive_view_fraction 1.0
--min_policy_val_cvar20_relative_gain 0.0
--min_policy_val_min_view_relative_gain 0.0
--enable_policy_val_image_ssim_gate
--min_policy_val_ssim_mean_gain 0.0
--min_policy_val_ssim_positive_view_fraction 0.75
--min_policy_val_ssim_min_view_gain -0.000005
--enable_policy_val_image_l1_gate
--min_policy_val_l1_mean_gain 0.0
--min_policy_val_l1_positive_view_fraction 0.5
--min_policy_val_l1_min_view_gain -0.000005
--min_policy_val_l1_cvar20_view_gain -0.000005
--min_target_changed_fraction 0.0
--write_noop_on_reject
--noop_fallback_source target_evidence
```

Equivalent runner invocation:

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <SCENE> \
  --gpu <GPU> \
  --output_root /dev/shm/peilincai_spcarnet_v50_l1risk_locked_20260623 \
  --tag v50_l1risk_locked_l1pos05_trainpolicy_fairnoop_region_texture_adapter \
  --min_policy_val_l1_positive_view_fraction 0.5 \
  --min_target_changed_fraction 0.0 \
  --force
```

The only intentional relaxation relative to v49b is
`--min_policy_val_l1_positive_view_fraction 0.5` instead of `1.0`. The mean L1,
min-view L1, and CVaR20 L1 vetoes stay active, so clear L1 regressions remain
rejected. This is intended to avoid over-rejecting tiny/noisy image-L1 cases
while preserving a train-only risk gate.

## v50 full9 summary

Root:

```text
/dev/shm/peilincai_spcarnet_v50_l1risk_locked_20260623
```

Tag:

```text
v50_l1risk_locked_l1pos05_trainpolicy_fairnoop_region_texture_adapter
```

Official summary paths:

```text
/dev/shm/peilincai_spcarnet_v50_l1risk_locked_20260623/v50_full9_summary.json
/dev/shm/peilincai_spcarnet_v50_l1risk_locked_20260623/v50_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623/v50/v50_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623/v50/v50_full9_summary.md
```

Aggregate:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v50 vs same-evidence no-op | 9 | 6 | 6 | +0.001264 | +0.00002174 | -0.00003405 |
| v50 vs v48 | 9 | 0 | 2 | -0.000198 | -0.00000600 | +0.00000548 |

Per-scene effective policy:

| scene | v49b effective | v50 effective | v50 accepted | dPSNR v50-v49b | dSSIM | dLPIPS |
|---|---|---|---:|---:|---:|---:|
| bicycle | fallback_noop | accepted_atlas | 1 | +0.000143 | +0.00000215 | +0.00000048 |
| flowers | fallback_noop | fallback_noop | 0 | +0.000000 | +0.00000000 | +0.00000000 |
| garden | fallback_noop | accepted_atlas | 1 | +0.000137 | +0.00000221 | -0.00000334 |
| stump | fallback_noop | fallback_noop | 0 | +0.000000 | +0.00000000 | +0.00000000 |
| treehill | fallback_noop | fallback_noop | 0 | +0.000000 | +0.00000000 | +0.00000000 |
| room | fallback_noop | accepted_atlas | 1 | +0.001656 | +0.00003940 | -0.00001846 |
| counter | fallback_noop | accepted_atlas | 1 | +0.003172 | +0.00003427 | -0.00008005 |
| kitchen | accepted_atlas | accepted_atlas | 1 | +0.000000 | +0.00000000 | +0.00000000 |
| bonsai | accepted_atlas | accepted_atlas | 1 | +0.000000 | +0.00000000 | +0.00000000 |

Interpretation:

- v50 successfully repairs the v49b over-rejection on `room` and `counter`, and
  also promotes a small safe `garden`/`bicycle` atlas row.
- v50 remains a representation-level safety/policy improvement, not a new
  headline endpoint: it is still below v48 on mean full9 metrics and has fewer
  nonregressive/tie scenes than v48.
- v48 remains the strongest current representation-level no-op comparison
  result; Phase-J remains the paper-safe visual headline.

## Representation lesson carried into v50

Several completed read-only audits converged on the same diagnosis:

- the current representation-level branch is not limited mainly by SH degree or
  one more alpha/strength parameter;
- the key limitation is action footprint: only a small fraction of target pixels
  land on atlas-supported faces, and the residual is then attenuated by
  confidence and safety gates;
- ELA is strong because it acts like a dense image-space residual transfer,
  while the persistent representation path is still sparse and surface-local;
- v50 must therefore be evaluated as a surface-support and atlas-policy
  improvement, not as another parameter sweep.

The practical consequence is that v50 should prefer fixed policy decisions that
increase trustworthy support coverage under train-only gates. If v49b simply
rejects many scenes to fair no-op, that is useful evidence, but not a paper-level
method improvement by itself.

## Current interpretation boundary

v49b is a fairness and safety diagnostic. It may reduce the apparent gains of
the old v49 because rejected rows now fall back to the correct no-op evidence.
That is expected and scientifically necessary.

The current paper-safe headline remains Phase-J:

- full9 local selected-clean MeshSplatting baseline: `9 / 9` strict RGB wins;
- mean deltas: `+1.3311` PSNR, `+0.0347` SSIM, `-0.0634` LPIPS;
- average triangle reduction: `7.6479%`;
- honest caveat: major RGB gain still comes from render-time ELA, while v48/v49b
  are representation-level internalization evidence.

v49b/v50 strengthen the representation-level story as safety diagnostics:
v49b fixed the fallback fairness flaw, and v50 showed that the L1-positive gate
can be relaxed in a fixed train-policy way to recover `room/counter`. They do
not yet replace v48 as the strongest representation-level result or Phase-J as
the paper-safe endpoint.

## Baseline wording boundary

The current safe public claim remains:

> Phase-J beats the local selected clean MeshSplatting baseline under the same
> local evaluation protocol.

The MeshSplatting paper-table comparison is useful only as a sanity check,
because preprocessing, implementation, resolution, masks, and evaluation scripts
may differ. v48/v50 should not be described as beating clean MeshSplatting;
their claim is only a small positive representation-level delta against the
same-evidence no-op compact baseline.

For geometry, the precise wording is:

```text
9 / 9 geometry-safe; 6 / 9 strict sparse-geometry wins; mean triangle reduction 7.6479%.
```

Do not say that geometry strictly improves on every scene.

## Remaining gap

- v50 is better than v49b, but it is not better than v48.
- The persistent surface-atlas route still has a small action footprint and
  small visual effect size compared with Phase-J ELA.
- The next method step should not be another scalar threshold tweak. It should
  increase trustworthy surface support or residual capacity while keeping the
  train-policy tail gates.
- Durable small artifacts for both v49b and v50 are archived at
  `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v49b_v50_l1risk_small_artifacts_20260623`.

## Early partial v49b health check

Partial summary path:

```text
/dev/shm/peilincai_spcarnet_v49b_fairnoop_20260623/v49b_partial_summary.md
```

As of the first completed two scenes, `bicycle` and `flowers` both reject the
atlas and write fair no-op fallback from `target_evidence`.

| scene | accepted | effective | fallback source | copied from source | v50 L1-pos05 implication |
|---|---:|---|---|---:|---|
| `bicycle` | 0 | `fallback_noop` | `target_evidence` | 0 | still rejected by min-view / SSIM-tail gates |
| `flowers` | 0 | `fallback_noop` | `target_evidence` | 0 | still rejected by min-view / SSIM / L1 mean gates |

This is not a full9 result. It only confirms that the fair fallback path is
working and that the first two outdoor scenes do not become v50 promotions by
relaxing the L1 positive-view fraction alone.
