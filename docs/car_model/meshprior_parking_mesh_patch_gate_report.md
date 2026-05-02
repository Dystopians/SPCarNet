# MeshPrior Parking Mesh Patch Gate Report

Date: 2026-05-01

## Scope

This step runs a no-op/protect readiness gate over extracted parking mesh patches. It verifies that each local patch can pass through gate plumbing and rollback snapshot creation before any deformation, fill, snap, or pruning is attempted.

No source model geometry is modified.

## Inputs

- Mesh patch summary: `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- Patch count: `8`
- Total patch faces: `10826`

## Implementation

Added:

- `scripts/car_model/meshprior_gate_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_gate.py`

The gate:

1. loads each extracted local patch;
2. evaluates no-op geometry and topology deltas;
3. writes a rollback snapshot for each patch;
4. marks the patch `protect_ready` if no-op geometry is stable and the patch has at least `50` faces.

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_gate_parking_mesh_patches.py --mesh_patch_summary outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json --output_dir outputs/carnet/meshprior/parking_phone_tiny/patch_gate
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/rollback_snapshots/*.npz`

Results:

- patches evaluated: `8`
- protect_ready: `8`
- deferred: `0`
- failed: `0`
- rollback snapshots: `8`
- geometry edited: `false`

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_mesh_patch_gate.py
```

Result: PASS.

Smoke checks:

- all `8` patches are evaluated;
- all `8` patches are `protect_ready`;
- no patch is deferred or failed;
- every patch has a rollback snapshot;
- JSON, CSV, and Markdown reports are written.

## Gate

Stage gate: PASS.

The real-scene parking pipeline now has stable local mesh patch assets and rollback snapshots. The next step should be a copied-patch before/after proposal test, starting with conservative snap/protect diagnostics before any full-scene training run.
