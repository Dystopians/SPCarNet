# MeshSplatOpt Stage R17.01 CSEF Local Snap Report

Date: 2026-05-03

## Decision

`LOCAL_CSEF_SNAP_SELECTOR_PASS`.

This stage strengthens the previously weak `SNAP_VERTICES` path. The old snap proposal generator fit one global plane to the whole mesh and did not explicitly reject moves through negative free-space evidence. The revised selector uses local one-ring/two-ring plane support per candidate vertex, records CSEF-style evidence in every edit, and rejects snaps whose negative free-space evidence exceeds the configured threshold.

## Implementation

Changed:

- `ss3dm_prior/meshsplatopt/snap_proposals.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py`

Key behavior:

- builds mesh vertex adjacency from faces;
- fits the snap target from local neighbor support while excluding the candidate vertex;
- caps displacement by scene scale as before;
- keeps boundary vertices conservative;
- accepts optional `vertex_evidence` with `positive_surface_evidence`, `negative_free_space_evidence`, and `uncertainty`;
- writes selector/evidence/risk metadata into the resulting `MeshEdit`;
- rejects unsupported vertices and high free-space-risk vertices before generating a move.

## Smoke Results

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py --output_dir outputs/carnet/meshsplatopt/stageR17_01_csef_local_snap_smoke
```

Result:

| check | status |
|---|---:|
| dent error reduced | `true` |
| floater rejected without support | `true` |
| negative free-space rejected | `true` |
| misalignment error reduced | `true` |
| rollback exact | `true` |

Metrics:

- dent plane error: `0.03072 -> 0.019831720797113993`
- misalignment plane error: `0.019200000000000002 -> 0.0096`

Compile:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q
```

Exit code: `0`.

## Interpretation

This does not yet prove a real-scene quality gain over the full freeze baseline, but it fixes the main scientific weakness of the previous snap selector: target evidence was global and risk-blind. The selector is now compatible with the CSEF paper story because the edit carries explicit local surface support and free-space rejection metadata.

The next gate is a real-checkpoint diagnostic: generate CSEF-local snap proposals on a public scene, run the existing render-backed counterfactual gate, then test a W&B-logged short recovery only if the gate accepts the edit.

## Artefacts

- `outputs/carnet/meshsplatopt/stageR17_01_csef_local_snap_smoke/snap_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR17_01_csef_local_snap_smoke/snap_outputs/snap_proposals.json`
- `outputs/carnet/meshsplatopt/stageR17_01_csef_local_snap_smoke/snap_outputs/snap_debug_before_after.ply`
