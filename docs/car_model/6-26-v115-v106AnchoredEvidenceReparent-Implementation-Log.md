# v115 v106-Anchored Evidence Reparent Implementation Log

Date: 2026-06-26

## Status

This is a method-infrastructure milestone, not a quality-promotion result.

The previous vNext full9 effective-margin run is reliable but weak: it completed
`9 / 9` scenes with `9 / 9` protocol pass, but accepted only `1 / 9` nonzero
rows and averaged `25.067410 / 0.741259 / 0.306695`, below both local clean
MeshSplatting and v106 POD-MoE base-preserve. The root cause is now clearer:
the vNext evidence path was still anchored to the Phase-F compact parent, so
fallback/no-op and residual application were not using the stronger v106 parent.

This update adds the missing pipeline interface needed for v115:

```text
v106 parent render + Phase-J teacher residual -> reparented evidence cache -> certified residual surface texture
```

## Implemented Code

New script:

- `scripts/car_model/ecsr_reparent_surface_evidence_cache.py`

Updated runner:

- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

The reparent tool copies an existing ECSR surface evidence cache, strictly
matches each NPZ view with a parent render image, replaces `rgb_render`, and
recomputes `residual_rgb` / `residual_l1` against the new parent whenever
`rgb_gt` is present. It writes an auditable JSON report:

```text
surface_evidence_reparent_audit.json
```

The scene runner now has explicit v115 anchor hooks:

```text
--reparent_fit_parent_render_dir
--reparent_target_parent_render_dir
--reparent_parent_label
--reparent_allow_resize
```

Default behavior is unchanged. Reparenting is only active when the new flags are
provided.

## Verified Behavior

### Unit Smoke

Synthetic evidence cache:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_reparent_surface_evidence_cache.py \
  --base_evidence_dir "$TMP/base" \
  --parent_render_dir "$TMP/parent" \
  --out_dir "$TMP/out" \
  --parent_label smoke_parent \
  --force
```

Observed:

```text
rgb_render_mean 0.2509765625
residual_l1_mean 0.4990234375
processed 1 residual_recomputed 1 skipped 0
```

### Runner Dry-Run

The vNext scene runner was dry-run on `counter` with both fit and target
reparent flags enabled.

Observed manifest command order:

```text
reparent_fit_evidence
reparent_target_evidence
strip_target_evidence_no_gt
apply_certified_residual_texture
```

The dry-run protocol audit still passed, and the manifest recorded:

```text
effective_fit_evidence_dir: /dev/shm/spcarnet_v115_runner_reparent_dryrun_20260626/counter/fit_evidence_reparented
effective_target_evidence_dir: /dev/shm/spcarnet_v115_runner_reparent_dryrun_20260626/counter/target_evidence_reparented
adapter_target_evidence_dir: /dev/shm/spcarnet_v115_runner_reparent_dryrun_20260626/counter/target_evidence_no_gt
```

## v114 Blocker

v114 OOF-refit is not a usable positive or negative result yet.

Current evidence:

- field exists for garden:
  `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden/fields/ours_26000_v114_oof_refit_podmoe_garden_field.pt`
- original detached model package is missing `cfg_args`;
- retrying with the available v100 recovery model is correctly rejected by
  endpoint hash safety:
  `RuntimeError: v102 surface residual field endpoint_report_sha256 mismatch`.

This is an artifact/provenance blocker. It should not be reported as a completed
metric result.

## Subagent Consensus

Three read-only subagents converged on the same diagnosis:

- v106 POD-MoE base-preserve is the current verified quality parent and main
  comparison line, but it is a sidecar field, not a drop-in model directory.
- Phase-J is the teacher / upper bound, not a baked representation parent.
- vNext effective-margin gate is a safety/no-op control, not a quality endpoint.
- v115 must prove the candidate representation is stronger under the same
  locked train-only certificate, not just loosen gates or tune per-scene
  thresholds.

## Required Next Experiment

The next real experiment is a three-scene v115 pilot on:

```text
flowers, garden, counter
```

These cover weak generalization, outdoor/frame-contract risk, and the only
previous nonzero accepted vNext scene.

Before that pilot can be fully fair, one of these must be true:

1. restore the original v101 detached model package used by the v106 fields; or
2. rebuild/render v106-compatible parent outputs from an available model and
   matching endpoint certificate.

Then run vNext with:

```text
--reparent_fit_parent_render_dir <v106 train parent renders>
--reparent_target_parent_render_dir <v106 target/test parent renders>
--teacher_render_dir <Phase-J train teacher renders>
--strict_no_target_gt_apply
--enable_policy_val_effective_margin_gate
```

Promotion requires, at minimum:

- full9 protocol pass `9 / 9`;
- exact no-target-GT apply;
- non-regressive versus v106 on all scenes;
- multiple nonzero accepted scenes, not mostly fallback/no-op;
- mean better than v106 by more than numerical noise;
- changed-region qualitative panels with native frame-contract hashes.

Until those are achieved, final status remains `NOT COMPLETE`.
