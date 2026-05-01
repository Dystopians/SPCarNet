# MeshPrior Stage 2 Region Mining — Smoke Report

| Field | Value |
|---|---|
| Stage | M2 / region mining smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Commands

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage2_region_mining.py
```

Output summary:

```text
regions: 2
eligible_for_posterior: 1
```

The smoke test creates an ASCII PLY with:

- one box-like component,
- one single-triangle small component.

The miner finds both connected components. The box-like component is eligible for posterior inference; the single-triangle component is retained in diagnostics but rejected for posterior inference because it has too few triangles.

## Dry-Run Missing-Data Check

```bash
micromamba run -n mesh_splatting python scripts/car_model/meshprior_mine_regions.py \
  --scene_model /tmp/meshprior_missing_model \
  --scene_source /tmp/meshprior_missing_scene \
  --output_dir /tmp/meshprior_stage2_dry_run \
  --mode dry_run
```

Output summary:

```text
regions: 0
eligible_for_posterior: 0
mesh_path: null
```

The CLI exited with code 0 and wrote the expected artifacts.

## Gate Verdict

`PASS`
