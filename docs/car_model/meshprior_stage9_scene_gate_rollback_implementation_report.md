# MeshPrior Stage 9 Scene Gate and Rollback — Implementation Report

| Field | Value |
|---|---|
| Stage | M9 / scene gate and rollback |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/scene_gate.py` | Dry-run scene evidence metrics, accept/reject gate, rollback snapshot save/restore. |
| `scripts/car_model/meshprior_evaluate_proposals.py` | CLI for dry-run proposal evaluation. |
| `scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py` | Synthetic accept/reject and rollback smoke test. |
| `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md` | Stage design. |

## 2. Implementation Summary

M9 implements the first scene-side gate for MeshPrior proposals.

Implemented functions:

- `ProposalGateResult`;
- `evaluate_proposal_geometry_delta(...)`;
- `evaluate_proposal_free_space_delta(...)`;
- `evaluate_proposal_topology_delta(...)`;
- `accept_or_reject(...)`;
- `save_rollback_snapshot(...)`;
- `restore_rollback_snapshot(...)`.

The gate is dry-run only at this stage. It works on before/after mesh arrays and optional object evidence.

## 3. Gate Behavior

Hard rejects:

- free-space violation increases;
- connected component count increases;
- triangle growth exceeds threshold;
- object uncertainty is too high;
- no scene metric improves.

Scene evidence improvements currently include:

- boundary edge count decreasing;
- free-space violation decreasing;
- connected component count decreasing.

Object evidence can only tighten or explain the decision. It cannot accept a proposal without scene-side support.

## 4. CLI

Implemented:

```bash
python scripts/car_model/meshprior_evaluate_proposals.py \
  --scene_source <colmap_scene> \
  --scene_model <trained_scene_model> \
  --proposals <proposals.json> \
  --output_dir outputs/carnet/meshprior/scene_gate/<run_name> \
  --mode dry_run
```

For dry-run, each proposal row points to before/after mesh NPZ files:

```json
{
  "proposal_id": "fill_good",
  "proposal_type": "fill",
  "before_npz": "before.npz",
  "after_npz": "after.npz",
  "object_evidence": {"uncertainty": 0.1}
}
```

Outputs:

- `gate_report.json`;
- `gate_report.md`;
- `<proposal_id>_rollback_snapshot.npz`.

## 5. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage8_fill.py
```

Smoke result:

- one topology-improving fill proposal accepted;
- one disconnected-floater proposal rejected;
- rollback snapshot restored vertices, faces, and metadata;
- CLI produced `gate_report.json` and `gate_report.md`.

## 6. Known Limitations

- Rendering, COLMAP sparse depth, sparse normal proxy, and FPS proxy are not connected yet.
- Free-space is function-hook based in dry-run; full scene free-space evidence remains for later integration.
- Proposal input format is minimal NPZ before/after mesh pairs.

## 7. Stage Gate

| Gate | Result |
|---|---|
| Dry-run gate works | PASS |
| Rollback snapshot and restore works | PASS |
| Gate report explains accepted/rejected proposals | PASS |
| Object prior cannot accept proposals without scene evidence | PASS |
| M8 fill regression still passes | PASS |

Decision: `PASS`. The next allowed stage is M10 scene-level optimization integration.
