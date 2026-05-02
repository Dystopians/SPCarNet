# MeshPrior Parking Metadata Gate Report

Date: 2026-05-01

## Scope

This step gates parking cluster proposal metadata into an explicit local mesh-extraction action plan. It does not run the full M9 before/after scene mesh gate because these parking proposals do not yet have editable mesh patches or stable face IDs.

The gate is intentionally conservative:

- `protect`, `snap_candidate`, and `fill_candidate` can become `candidate_extract` only when cluster support is high.
- `prune` is always deferred until real scene mesh evidence exists.
- high-uncertainty clusters become diagnostics rather than edits.

## Inputs

- Proposal metadata: `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json`
- Proposal count: `45`
- Eligible source clusters: `9`

## Implementation

Added:

- `scripts/car_model/meshprior_gate_parking_metadata_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_metadata_gate.py`

Default gate thresholds:

- minimum candidate support: `0.80`
- minimum candidate views: `4`
- maximum candidate uncertainty: `0.25`
- uncertainty diagnostic threshold: `0.25`

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_gate_parking_metadata_proposals.py --proposals outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json --output_dir outputs/carnet/meshprior/parking_phone_tiny/metadata_gate
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.md`

Results:

- proposals evaluated: `45`
- candidate_extract: `24`
- deferred: `17`
- diagnostic: `1`
- rejected: `3`
- mesh extraction targets: `8`
- diagnostic targets: `1`
- geometry edited: `false`

Top mesh-extraction targets:

| region | proposal types | views | sparse points | support | uncertainty |
| --- | --- | ---: | ---: | ---: | ---: |
| `parking_region_0000` | `protect`, `snap_candidate`, `fill_candidate` | 32 | 3851 | 1.0000 | 0.0000 |
| `parking_region_0001` | `protect`, `snap_candidate`, `fill_candidate` | 20 | 2203 | 1.0000 | 0.0000 |
| `parking_region_0003` | `protect`, `snap_candidate`, `fill_candidate` | 14 | 1028 | 1.0000 | 0.0000 |
| `parking_region_0006` | `protect`, `snap_candidate`, `fill_candidate` | 11 | 548 | 1.0000 | 0.0000 |
| `parking_region_0004` | `protect`, `snap_candidate`, `fill_candidate` | 8 | 851 | 1.0000 | 0.0000 |

Diagnostic target:

- `parking_region_0008`: uncertainty `0.4800`, only `2` views and `39` sparse points.

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_metadata_gate.py
```

Result: PASS.

Smoke checks:

- all `45` metadata proposals are evaluated
- at least one mesh-extraction target is produced
- prune is not included in mesh-extraction targets
- action-plan JSON, gate report JSON/Markdown, and CSV are written

## Gate

Stage gate: PASS.

This is not a geometry gate. It is a readiness gate for local mesh extraction. The next missing technical bridge is to map each accepted parking cluster to a local editable scene mesh patch with stable face IDs, after which the existing before/after scene-gate and rollback machinery can become authoritative.
