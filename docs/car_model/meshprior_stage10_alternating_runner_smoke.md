# MeshPrior Stage 10 Alternating Runner — Smoke Report

| Field | Value |
|---|---|
| Stage | M10 / alternating runner smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Command

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
```

The smoke runs:

```bash
python scripts/car_model/meshprior_run_pipeline.py \
  --scene_source synthetic \
  --scene_model synthetic \
  --posterior_checkpoint "" \
  --output_dir <tmp>/pipeline \
  --proposal_types protect prune fill \
  --mode dry_run \
  --require_gate_pass
```

## Output Summary

```text
status: PASS
accepted_count: 1
rejected_count: 0
```

Required artifacts verified:

- `pipeline_status.json`;
- `scene_gate/gate_report.json`;
- `accepted_proposals.json`;
- `pipeline_report.md`;
- `regions.json`;
- `posterior/posterior_summary.json`.

## Safety Check

The runner stayed in dry-run artifact mode. It did not write back to a scene model or source mesh.

## Gate Verdict

`PASS`
