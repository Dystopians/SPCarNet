# MeshPrior Parking Patch Proposal Test Report

Date: 2026-05-01

## Scope

This step runs before/after proposal tests on copied parking mesh patches. It is a gate and rollback stress test, not a full-scene geometry update.

The source training checkpoint is not modified.

## Inputs

- Mesh patch summary: `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- Patches tested: `8`
- Total source patch faces: `10826`

## Implementation

Added:

- `scripts/car_model/meshprior_test_parking_patch_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_patch_proposals.py`

For each local patch, the test creates three copied before/after proposals:

- `protect_noop`: unchanged copied patch; expected to be rejected by the M9 scene gate because no scene metric improves.
- `component_cleanup_candidate`: removes the smallest `5%` of copied patch triangles; expected to be accepted by the topology gate on disconnected triangle-splat patches.
- `floater_reject`: adds one isolated copied triangle; expected to be rejected by the topology gate.

The `component_cleanup_candidate` result is a stress-test acceptance on copied patch data. It is not yet an approved full-scene edit because render/photometric validation and checkpoint writeback are still missing.

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_test_parking_patch_proposals.py --mesh_patch_summary outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json --output_dir outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/proposal_meshes/*/*.npz`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/rollback_snapshots/*.npz`

Results:

- patches tested: `8`
- proposal tests: `24`
- accepted: `8`
- rejected: `16`
- protect_noop_rejected: `8`
- cleanup_accepted: `8`
- floater_rejected: `8`
- source model edited: `false`
- copied patch geometry edited: `true`

Representative metrics:

| proposal | accepted | triangle delta | component delta | boundary delta | reasons |
| --- | --- | ---: | ---: | ---: | --- |
| `parking_region_0000_protect_noop` | false | 0 | 0 | 0 | `no_scene_metric_improved` |
| `parking_region_0000_component_cleanup_candidate` | true | -113 | -113 | 339 | `accepted_by_scene_evidence` |
| `parking_region_0000_floater_reject` | false | 1 | 1 | -3 | `component_count_increased`, `no_scene_metric_improved` |

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_patch_proposals.py
```

Result: PASS.

Smoke checks:

- `2` patches produce `6` copied-patch proposal tests;
- cleanup candidates are accepted;
- floater proposals are rejected;
- no-op protect proposals are rejected by the M9 improvement gate;
- source model remains unmodified.

## Gate

Stage gate: SOFT PASS.

The gate plumbing behaves correctly on copied real-scene patches:

- it rejects no-op proposals that do not improve scene metrics;
- it rejects new isolated components;
- it accepts topology cleanup on copied disconnected triangle-splat patches.

This is a soft pass because the accepted cleanup candidates are not yet validated by rendering or by a full checkpoint writeback/recovery loop. The next high-priority step is a checkpoint-copy application test that writes accepted copied-patch face removals into a duplicated triangle state, evaluates bookkeeping integrity, and only then considers a short render/geometry evaluation.
