# MeshPrior Stage 5 Optimizer Adapter — Smoke Report

| Field | Value |
|---|---|
| Stage | M5 / optimizer adapter smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Command

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py
```

## Verified Behavior

- Loaded a synthetic `triangle_scores.npz`.
- Normalized protect/prune scores per region.
- Checked all normalized values are finite.
- Verified bounded-add score combination cannot change base scores by more than `0.25`.
- Exported both:
  - `meshprior_scores.npz`,
  - `meshprior_prism_scores.json`.
- Reloaded the generic NPZ.
- Detected PRISM as present in this repository.

## Output Summary

```text
rows: 4
prism_present: true
```

## Gate Verdict

`PASS`
