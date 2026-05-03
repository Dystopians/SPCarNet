# MeshSplatOpt Stage R14.8 Checkpoint Topology Evidence Audit

Date: 2026-05-02

## Gate

`PASS` for the audit, with a hard sub-gate failure:

```text
FAIL_EDGE_CSEF_INVALID_TRIANGLE_SOUP
```

The real Mesh Splatting checkpoint is a triangle-soup representation. Shared-edge topology is not a valid signal for real checkpoint proposal selection.

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_audit_checkpoint_topology_evidence.py \
  --checkpoint_path outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt \
  --output_dir outputs/carnet/meshsplatopt/stageR14_8_checkpoint_topology_evidence_audit
```

## Result

| metric | value |
|---|---:|
| vertices | `193491` |
| triangles | `64497` |
| connected components | `64497` |
| largest component faces | `1` |
| largest component fraction | `0.000015504597113044017` |
| single-face component fraction | `1.0` |
| shared edges | `0` |
| boundary face fraction | `1.0` |
| repeated vertex refs | `0` |

## Decision

Do not use shared-edge boundary-loop CSEF to select real checkpoint edits. It will misclassify every triangle as boundary evidence and hallucinate holes.

The next real proposal selector must use spatial adjacency, render residuals, sparse COLMAP evidence, or explicit checkpoint/raster evidence instead of mesh edge connectivity.

## Artefacts

- `scripts/car_model/meshsplatopt_audit_checkpoint_topology_evidence.py`
- `outputs/carnet/meshsplatopt/stageR14_8_checkpoint_topology_evidence_audit/checkpoint_topology_evidence_audit.json`
- `outputs/carnet/meshsplatopt/stageR14_8_checkpoint_topology_evidence_audit/checkpoint_topology_evidence_audit.md`
