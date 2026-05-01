# MeshPrior Stage 9 Scene Gate and Rollback — Smoke Report

| Field | Value |
|---|---|
| Stage | M9 / scene gate and rollback smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Commands

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage8_fill.py
```

## Synthetic Cases

Accepted case:

- proposal: `fill_good`;
- type: `fill`;
- boundary edges: `4 -> 0`;
- component count: `1 -> 1`;
- free-space violation delta: `0`;
- result: accepted.

Rejected case:

- proposal: `floater_bad`;
- type: `fill`;
- component count: `1 -> 2`;
- floater count delta: `1`;
- result: rejected.

Rollback:

- snapshot written as NPZ;
- vertices restored exactly;
- faces restored exactly;
- metadata restored.

## CLI Report

The smoke also ran:

```bash
python scripts/car_model/meshprior_evaluate_proposals.py \
  --scene_source synthetic \
  --scene_model synthetic \
  --proposals proposals.json \
  --output_dir gate \
  --mode dry_run
```

The generated report contained:

```text
accepted_count: 1
rejected_count: 1
```

## Gate Verdict

`PASS`
