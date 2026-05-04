# Final Stage F38 - Counterfactual Gate Ablation Report

Decision: `F38_SYNTHETIC_COUNTERFACTUAL_GATE_PASS_REAL_SCENE_FULL_ABLATION_STILL_OPEN`.

This is a mechanism-level counterfactual ablation. It applies the same edit proposals with the MeshSplatOpt gate/rollback path and with an unsafe no-gate/no-rollback path.

Status: `PASS`.

## Checks

| check | result |
| --- | --- |
| `supported_fill_survives_gate` | `True` |
| `all_unsafe_rejected_cases_rollback_exact` | `True` |
| `all_no_gate_rejected_cases_commit_damage` | `True` |
| `floater_would_add_unobserved_face` | `True` |
| `snap_would_move_vertex_by_5m` | `True` |
| `delete_would_remove_supported_face` | `True` |

## Cases

| case | gated accepted | gated reasons | gated state expected | no-gate topology delta | no-gate max vertex displacement |
| --- | --- | --- | --- | --- | --- |
| `accepted_supported_fill` | `True` | `none` | `True` | `V 16, F 18` | `0.000000` |
| `rejected_bad_floater` | `False` | `free_space_gate_failed, fill_boundary_certificate_failed` | `True` | `V 3, F 1` | `0.000000` |
| `rejected_snap_through_free_space` | `False` | `free_space_gate_failed, snap_free_space_rejected` | `True` | `V 0, F 0` | `5.000000` |
| `rejected_delete_supported_surface` | `False` | `delete_supported_surface_rejected` | `True` | `V 0, F -1` | `0.000000` |

## Interpretation

The ablation shows that the safety gate is load-bearing: the floater, free-space snap, and supported-surface deletion are all rejected and exactly rolled back, while the no-gate/no-rollback path commits each damaging topology or geometry mutation. This closes the implementation-level counterfactual gap, but it is not a substitute for a full real-scene render/geometry no-gate training ablation.

## Artifacts

- JSON: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/final_stageF38_counterfactual_gate_ablation/counterfactual_gate_ablation.json`
