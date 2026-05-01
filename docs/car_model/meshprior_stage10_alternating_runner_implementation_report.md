# MeshPrior Stage 10 Alternating Runner — Implementation Report

| Field | Value |
|---|---|
| Stage | M10 / alternating runner |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage10_alternating_runner_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `scripts/car_model/meshprior_run_pipeline.py` | Dry-run MeshPrior orchestration runner. |
| `scripts/car_model/smoke_test_meshprior_stage10_pipeline.py` | Synthetic dry-run pipeline smoke test. |
| `docs/car_model/meshprior_stage10_alternating_runner_design.md` | Stage design. |

## 2. Implementation Summary

M10 wires prior stages into an end-to-end dry-run pipeline:

1. synthetic scene setup or placeholder scene input;
2. region mining artifact;
3. posterior summary artifact;
4. protect/prune score export;
5. optional snap proposal;
6. optional fill proposal;
7. scene gate evaluation;
8. accepted proposal export;
9. markdown report generation.

The runner is artifact-only. It does not modify scene geometry, and `--apply` currently raises an error.

## 3. CLI

Implemented:

```bash
python scripts/car_model/meshprior_run_pipeline.py \
  --scene_source <colmap_scene> \
  --scene_model <trained_scene_model> \
  --posterior_checkpoint <stage3_checkpoint> \
  --output_dir outputs/carnet/meshprior/pipeline/<run_name> \
  --proposal_types protect prune \
  --mode dry_run
```

Supported resume and safety flags:

- `--skip_region_mining`;
- `--regions_json`;
- `--posterior_dir`;
- `--proposals_json`;
- `--eval_only`;
- `--dry_run`;
- `--no_geometry_write`;
- `--max_regions`;
- `--max_proposals`;
- `--require_gate_pass`;
- `--apply` rejected in M10.

## 4. Artifacts

The smoke run wrote:

```text
run_config.json
synthetic_scene/damaged_mesh.npz
regions.json
posterior/posterior_summary.json
proposals/protect_prune_scores.json
proposals/proposals.json
scene_gate/gate_report.json
accepted_proposals.json
pipeline_report.md
pipeline_status.json
```

Rollback snapshots are written by the scene gate under `scene_gate/`.

## 5. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py
```

Smoke result:

```text
status: PASS
accepted_count: 1
rejected_count: 0
```

The accepted proposal was a guarded fill on the synthetic local-hole mesh.

## 6. Real Scene Status

Real-scene dry-run is not exercised in M10 because no concrete scene source/model path was selected in this stage. The runner accepts the required CLI shape and keeps scene geometry read-only by default.

## 7. Stage Gate

| Gate | Result |
|---|---|
| Synthetic dry-run pipeline completes end-to-end | PASS |
| Accepted proposal export is written | PASS |
| Report generation works | PASS |
| Runner does not modify mesh geometry by default | PASS |
| M9 gate regression still passes | PASS |

Decision: `PASS`. The next allowed stage is M11 actual scene training/evaluation and wandb.
